"""Training data: the extractor's shards (run/src/dataset.rs) as one indexed
store, and a batch builder for every head.

A sample is one game state. `prev` links each to the sample 4 frames earlier
on its path; `next` (built here) inverts it, and `chain[i, j]` is the sample
4 * (j + 1) frames after i (-1 when not recorded). That gives k-step pairs
for k in {4, 16, 64} with the pads between them.

Hindsight pairs (a, g) for distance, reachability and policy:
- tree: g in the same edge after a, or in a recorded descendant edge
  (ancestry from nodes.csv); frame(g) - frame(a) is an upper bound on the
  frames needed.
- TAS: any later TAS sample; near-optimal, so a tight bound.
Reachability negatives: g with fewer stars than a (stars never go down: no
budget reaches it), and TAS pairs with budget < half the TAS's frames.
"""

import glob
import os

import numpy as np
import pandas as pd

from .schema import BUTTON_BITS, Schema

HOPS = 16  # chain length: 16 * 4 = 64 frames
KS = (4, 16, 64)
N_STATIC_GEO = 96
GOAL_GEO = 24  # of N_STATIC_GEO, reserved along the direction to the goal
N_GEO = N_STATIC_GEO + 32


# Goal ids for "collect this": course * 8 + star bit (course 0: the castle's
# own stars, save-flag bits 24-30), KEY_GOAL + k for the k-th Bowser key.
N_GOALS = 26 * 8 + 3
KEY_GOAL = 26 * 8


def parse_star_flags(text):
    """nodes.csv `star_flags` ("course:hexflags ...") -> set of goal ids."""
    out = set()
    for part in text.split():
        c, _, f = part.partition(":")
        try:
            c, f = int(c), int(f, 16) & 0x7F
        except ValueError:
            continue
        out |= {c * 8 + b for b in range(7) if f >> b & 1}
    return out


# Learned objectives (v6): what the edge did to an object, and its style.
VERBS = ("none", "touch", "stand", "pound", "grab", "defeat")
VERB_NONE, VERB_TOUCH, VERB_STAND, VERB_POUND, VERB_GRAB, VERB_DEFEAT = range(6)
ACT_GROUND_POUND_LAND = 0x0080023C
STYLES = ("reach", "exit", "event", "speed", "object")
# Event bits worth an objective of their own: ledge grab, enter water, coin,
# star, key, red coin, grab, dialog, cannon.
SALIENT_EVENTS = (1 << 2) | (1 << 5) | (0x7F << 7)


def edge_style(zone_change, ev, speed, verb=VERB_NONE):
    if zone_change:
        return 1
    if verb != VERB_NONE:
        return 4
    if ev & SALIENT_EVENTS:
        return 2
    if abs(speed) >= 30:
        return 3
    return 0


def goal_id(m):
    """Goal id of a reference milestone (refs.Reference.milestones)."""
    if m["kind"] == "key":
        return KEY_GOAL + m["keys"]
    if m["course"] >= 0 and m["bit"] >= 0:
        return m["course"] * 8 + m["bit"]
    return None


class Shard:
    def __init__(self, d, sch):
        self.dir = d
        meta = np.fromfile(f"{d}/meta.i64", np.int64).reshape(-1, 6)
        self.n = len(meta)
        self.meta = meta
        self.gi = np.fromfile(f"{d}/gi.i32", np.int32).reshape(self.n, sch.n_gi)
        self.gf = np.fromfile(f"{d}/gf.f32", np.float32).reshape(self.n, sch.n_gf)
        self.pads = np.fromfile(f"{d}/pads.u32", np.uint32).reshape(self.n, 4)
        on = np.fromfile(f"{d}/on.u16", np.uint16).astype(np.int64)
        tn = np.fromfile(f"{d}/tn.u16", np.uint16).astype(np.int64)
        self.on, self.tn = on, tn
        self.ooff = np.concatenate([[0], np.cumsum(on)])
        self.toff = np.concatenate([[0], np.cumsum(tn)])
        self.oi = np.memmap(f"{d}/oi.i32", np.int32, "r", shape=(int(self.ooff[-1]), sch.n_oi))
        self.of = np.memmap(f"{d}/of.f32", np.float32, "r", shape=(int(self.ooff[-1]), sch.n_of))
        self.tri = np.memmap(f"{d}/tri.f32", np.float32, "r", shape=(int(self.toff[-1]), sch.n_tri))


class Store:
    """All samples of the given sources (directories holding s*/ shards and
    zones/), indexed globally."""

    def __init__(self, sources, nodes_csv=None, sch=None):
        """`sources`: data directories, or (directory, nodes.csv) pairs for
        trees (each tree's node ids are offset so they don't collide); a bare
        directory uses `nodes_csv`. TAS sources have no nodes (node -1)."""
        self.sch = sch or Schema()
        sch = self.sch
        self.shards, self.zones = [], {}
        self.trees = []  # (nodes.csv, first shard, last shard + 1)
        for src in sources:
            src, csv = (src, nodes_csv) if isinstance(src, str) else src
            first = len(self.shards)
            for d in sorted(glob.glob(f"{src}/s*"), key=lambda p: int(os.path.basename(p)[1:])):
                self.shards.append(Shard(d, sch))
            if csv is not None and any((sh.meta[:, 0] >= 0).any() for sh in self.shards[first:]):
                self.trees.append((csv, first, len(self.shards)))
            for z in glob.glob(f"{src}/zones/*.f32"):
                lv, ar = map(int, os.path.basename(z)[:-4].split("_"))
                t = np.fromfile(z, np.float32).reshape(-1, sch.n_tri)
                if len(t) > len(self.zones.get((lv, ar), ())):
                    self.zones[(lv, ar)] = t
        self.zone_cent = {k: (t[:, 0:3] + t[:, 3:6] + t[:, 6:9]) / 3 for k, t in self.zones.items()}
        sizes = [s.n for s in self.shards]
        self.base = np.concatenate([[0], np.cumsum(sizes)])
        self.N = int(self.base[-1])
        self.shard_of = np.repeat(np.arange(len(sizes)), sizes)
        self.local = np.arange(self.N) - self.base[self.shard_of]
        meta = np.concatenate([s.meta for s in self.shards])
        prev = meta[:, 2].copy()
        ok = prev >= 0
        prev[ok] += self.base[self.shard_of[ok]]
        self.node, self.frame, self.prev, self.events = meta[:, 0], meta[:, 1], prev, meta[:, 3].astype(np.int64)
        self.npad_valid, self.source = meta[:, 4], meta[:, 5]
        self.gi = np.concatenate([s.gi for s in self.shards])
        self.gf = np.concatenate([s.gf for s in self.shards])
        self.pads = np.concatenate([s.pads for s in self.shards])
        self._build_chain()
        self._load_nodes()

    # -- structure -----------------------------------------------------------
    def _build_chain(self):
        nxt = np.full(self.N, -1, np.int64)
        has = self.prev >= 0
        nxt[self.prev[has]] = np.flatnonzero(has)
        ch = np.full((self.N, HOPS), -1, np.int64)
        ch[:, 0] = nxt
        for j in range(1, HOPS):
            p = ch[:, j - 1]
            ch[:, j] = np.where(p >= 0, nxt[np.maximum(p, 0)], -1)
        self.chain = ch
        self.chain_len = (ch >= 0).sum(1)

    def _load_nodes(self):
        """Every tree's nodes in one numbering: node ids are offset per tree
        (self.node is rewritten accordingly); parents, depths, frames, the
        objective kind that made each node, the winning route, and the stars
        (goal ids, `goal_id`) and keys each node has."""
        parents, depths, frames, vias, routes, stars, keys = [], [], [], [], [], [], []
        off = 0
        self.tree_of_node = []
        for t, (csv, a, b) in enumerate(self.trees):
            d = pd.read_csv(csv, usecols=["id", "parent", "via", "depth", "stars", "frame", "keys", "star_flags"],
                            dtype={"star_flags": str})
            par = d.parent.values.astype(np.int64)
            parents.append(np.where(par >= 0, par + off, -1))
            depths.append(d.depth.values.astype(np.int64))
            frames.append(d.frame.values.astype(np.int64))
            vias.append(d.via.fillna("root").str.replace("~d", "", regex=False).values)
            best = int(d[d.stars == d.stars.max()].sort_values("frame").id.iloc[0])
            r = np.zeros(len(d), bool)
            v = best
            while v >= 0:
                r[v] = True
                v = int(par[v])
            routes.append(r)
            stars.append([parse_star_flags(x) for x in d.star_flags.fillna("").values])
            keys.append(d["keys"].values.astype(np.int64))
            self.tree_of_node.append(np.full(len(d), t))
            for sh in self.shards[a:b]:
                m = sh.meta[:, 0] >= 0
                sh.meta[m, 0] += off
            lo, hi = self.base[a], self.base[b]
            tn = self.node[lo:hi] >= 0
            self.node[lo:hi][tn] += off
            off += len(d)
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
        self.parent = cat(parents, np.int64)
        self.depth = cat(depths, np.int64)
        self.node_frame = cat(frames, np.int64)
        self.on_route = cat(routes, bool)
        self.node_keys = cat(keys, np.int64)
        self.node_stars = [s_ for ss in stars for s_ in ss]
        self.tree_of_node = cat(self.tree_of_node, np.int64)
        via = np.concatenate(vias) if vias else np.zeros(0, object)
        self.kind_names = ["TAS"] + sorted(set(via.tolist()))
        kidx = {k: i for i, k in enumerate(self.kind_names)}
        node_kind = np.array([kidx[v] for v in via], np.int64)
        tree = self.node >= 0
        self.kind = np.zeros(self.N, np.int64)
        self.kind[tree] = node_kind[self.node[tree]]

    def star_labels(self, tas_name="tas120", class_min=False):
        """Time-to-star labels (CSR over samples: self.lab_off, lab_x goal ids,
        lab_t frames, and self.lab_tight per sample): for a tree sample, each
        star or key it doesn't have that a descendant collects, with the
        frames until the earliest such descendant (an upper bound); for a TAS
        sample, each one the TAS collects later, with the TAS's frames (tight)."""
        n_nodes = len(self.parent)
        # Per goal, the earliest frame it is first collected at a node or below.
        goals = sorted({g for ss in self.node_stars for g in ss} |
                       {KEY_GOAL + k for k in range(1, 3) if (self.node_keys >= k).any()})
        gi_of = {g: i for i, g in enumerate(goals)}
        best = np.full((n_nodes, len(goals)), 1 << 30, np.int32)
        for v in range(n_nodes):
            p = self.parent[v]
            have = self.node_stars[v] | {KEY_GOAL + k for k in range(1, int(self.node_keys[v]) + 1)}
            had = (self.node_stars[p] | {KEY_GOAL + k for k in range(1, int(self.node_keys[p]) + 1)}) if p >= 0 else set()
            for g in have - had:
                best[v, gi_of[g]] = self.node_frame[v]
        for v in np.argsort(-self.depth, kind="stable"):
            p = self.parent[v]
            if p >= 0:
                np.minimum(best[p], best[v], out=best[p])
        G = self.sch.gi
        cs = np.stack([self.gi[:, G[f"cs{k}"]] for k in range(25)], 1).astype(np.int64)
        flags = self.gi[:, G["save_flags"]].astype(np.int64) & 0xFFFFFFFF
        keys = self.gi[:, G["keys"]].astype(np.int64)
        xs, ts, cnt = [], [], np.zeros(self.N, np.int64)
        tree = np.flatnonzero(self.node >= 0)
        BIG = 1 << 29
        for j, g in enumerate(goals):
            t = best[self.node[tree], j] - self.frame[tree]
            ok = (t >= 0) & (best[self.node[tree], j] < BIG) & ~self._has_goal(g, cs[tree], flags[tree], keys[tree])
            idx = tree[ok]
            xs.append(np.stack([idx, np.full(len(idx), g), t[ok]], 1))
        # TAS: its own collection frames.
        from .refs import build_reference
        ref = build_reference(tas_name)
        tas = np.flatnonzero(self.node < 0)
        for m in ref.milestones:
            g = goal_id(m)
            if g is None:
                continue
            t = m["frame"] - self.frame[tas]
            ok = (t > 0) & ~self._has_goal(g, cs[tas], flags[tas], keys[tas])
            idx = tas[ok]
            xs.append(np.stack([idx, np.full(len(idx), g), t[ok]], 1))
        allr = np.concatenate(xs) if xs else np.zeros((0, 3), np.int64)
        if class_min:
            # Cross-tree minimum (v5): a label becomes the fastest time any
            # tree took to that star from any state of the same class: the
            # exact collected set (course star flags, save flags, keys),
            # zone, 256-unit cell and work under way.
            gf = self.sch.gf
            cell = np.floor(self.gf[:, [gf["pos_x"], gf["pos_y"], gf["pos_z"]]] / 256).astype(np.int64)
            parts = [cs, flags[:, None], keys[:, None], self.gi[:, [G["level"], G["area"], G["wip"]]].astype(np.int64), cell]
            key = np.concatenate(parts, 1)
            _, cls = np.unique(key, axis=0, return_inverse=True)
            cls = cls.reshape(-1)
            ck = cls[allr[:, 0]] * (N_GOALS + 1) + allr[:, 1]
            order = np.argsort(ck, kind="stable")
            uk, start = np.unique(ck[order], return_index=True)
            mins = np.minimum.reduceat(allr[order, 2], start)
            allr[:, 2] = mins[np.searchsorted(uk, ck)]
            self.class_min_classes = int(cls.max()) + 1
        allr = allr[np.argsort(allr[:, 0], kind="stable")]
        np.add.at(cnt, allr[:, 0], 1)
        self.lab_off = np.concatenate([[0], np.cumsum(cnt)])
        self.lab_x = allr[:, 1].astype(np.int64)
        self.lab_t = allr[:, 2].astype(np.float32)
        self.lab_tight = self.node < 0
        return self

    def edges(self, beta=150.0):
        """Edge dataset (v6), after star_labels: for each recorded tree edge
        with states near both ends, its start sample, end sample, frames,
        objective kind, and per star the credit weight exp(A / beta), A =
        T(start, X) - (frames + T(end, X)) (0 if the edge collects X) from
        the time-to-star labels: about 1 for edges on the fastest known way
        to X, near 0 for detours. Stored as edge_* arrays, one row per
        (edge, star) with a label at the start."""
        rows = []
        tree = np.flatnonzero(self.node >= 0)
        order = tree[np.lexsort((self.frame[tree], self.node[tree]))]
        nodes, first, cnt = np.unique(self.node[order], return_index=True, return_counts=True)
        lab = {}
        for n, f0, c in zip(nodes, first, cnt):
            n = int(n)
            p = self.parent[n]
            if p < 0:
                continue
            a, b = order[f0], order[f0 + c - 1]
            # Start sample within 8 frames of the edge's start; end sample
            # within 8 of its end.
            if self.frame[a] - self.node_frame[p] > 8 or self.node_frame[n] - self.frame[b] > 8 or a == b:
                continue
            frames = int(self.node_frame[n] - self.node_frame[p])
            la = {int(x): float(t) for x, t in zip(self.lab_x[self.lab_off[a]:self.lab_off[a + 1]], self.lab_t[self.lab_off[a]:self.lab_off[a + 1]])}
            lb = {int(x): float(t) for x, t in zip(self.lab_x[self.lab_off[b]:self.lab_off[b + 1]], self.lab_t[self.lab_off[b]:self.lab_off[b + 1]])}
            got = self.node_stars[n] - (self.node_stars[p] if p >= 0 else set())
            # Event bits over the edge (each sample carries those since the
            # previous one, so the start sample's own are left out).
            ev = int(np.bitwise_or.reduce(self.events[order[f0 + 1 : f0 + c]]))
            verb, slot = self.edge_object(order[f0 : f0 + c])
            for x, ta in la.items():
                tb = 0.0 if x in got else lb.get(x)
                if tb is None:
                    continue
                A = min(0.0, ta - (frames + tb))
                rows.append((a, b, frames, x, np.exp(A / beta), ev, verb, slot))
        r = np.array(rows, np.float64).reshape(-1, 8)
        self.edge_a, self.edge_b = r[:, 0].astype(np.int64), r[:, 1].astype(np.int64)
        self.edge_frames, self.edge_x = r[:, 2].astype(np.float32), r[:, 3].astype(np.int64)
        self.edge_w = r[:, 4].astype(np.float32)
        self.edge_ev = r[:, 5].astype(np.int64)
        self.edge_verb, self.edge_slot = r[:, 6].astype(np.int64), r[:, 7].astype(np.int64)
        ok = self.mask[self.edge_a]
        for k in ("edge_a", "edge_b", "edge_frames", "edge_x", "edge_w", "edge_ev", "edge_verb", "edge_slot"):
            setattr(self, k, getattr(self, k)[ok])
        return self

    def edge_object(self, chain):
        """The object an edge (its samples, start first) did something to, in
        hindsight: (verb, pool slot), verb one of VERBS (0 = none, slot -1).
        Priority grab > ground pound > defeat > stand > touch."""
        sch = self.sch
        G, O, OF = sch.gi, sch.oi, sch.of
        oi, of, om = self.objects(chain)
        mp = np.stack([self.gf[chain, sch.gf[k]] for k in ("pos_x", "pos_y", "pos_z")], -1)
        op = np.stack([of[..., OF[k]] for k in ("pos_x", "pos_y", "pos_z")], -1)
        d = np.where(om, np.linalg.norm(op - mp[:, None], axis=-1), np.inf)
        dxz = np.where(om, np.linalg.norm((op - mp[:, None])[..., [0, 2]], axis=-1), np.inf)
        slots = oi[..., O["slot"]]
        gi = self.gi[chain]
        if (gi[0, G["level"]], gi[0, G["area"]]) != (gi[-1, G["level"]], gi[-1, G["area"]]):
            return VERB_NONE, -1
        if gi[-1, G["holding"]] and gi[0, G["held_slot"]] != gi[-1, G["held_slot"]] or (gi[-1, G["holding"]] and not gi[0, G["holding"]]):
            return VERB_GRAB, int(gi[-1, G["held_slot"]])
        gp = np.flatnonzero(np.isin(gi[1:, G["action"]], (ACT_GROUND_POUND_LAND,))) + 1
        for t in gp:
            j = int(np.argmin(dxz[t]))
            if dxz[t, j] < 250:
                return VERB_POUND, int(slots[t, j])
        # Defeat: a (slot, behaviour) near Mario at the start, gone at the end.
        end = set(zip(slots[-1][om[-1]].tolist(), oi[-1, :, O["bhv"]][om[-1]].tolist()))
        for j in np.argsort(d[0]):
            if d[0, j] > 600:
                break
            if (int(slots[0, j]), int(oi[0, j, O["bhv"]])) not in end:
                return VERB_DEFEAT, int(slots[0, j])
        if gi[-1, G["floor_is_obj"]]:
            j = int(np.argmin(dxz[-1]))
            if dxz[-1, j] < 400:
                return VERB_STAND, int(slots[-1, j])
        later = d[1:] if len(chain) > 1 else d
        t, j = np.unravel_index(int(np.argmin(later)), later.shape)
        if later[t, j] < 120:
            return VERB_TOUCH, int(slots[t + (len(chain) > 1), j])
        return VERB_NONE, -1

    @staticmethod
    def _has_goal(g, cs, flags, keys):
        if g >= KEY_GOAL:
            return keys >= g - KEY_GOAL
        c, b = divmod(g, 8)
        if c == 0:
            return ((flags >> (24 + b)) & 1).astype(bool)
        return ((cs[:, c - 1] >> b) & 1).astype(bool)

    def val_subtrees(self, frac=0.05, seed=0, lo=0.002, hi=0.02):
        """Validation nodes: whole subtrees (a root and every descendant), so
        validation states have no near-duplicate parent or child in training.
        Roots are random nodes off the winning route whose subtree holds
        lo..hi of the recorded nodes; picked until frac of them are covered."""
        n = len(self.parent)
        rec = np.zeros(n, np.int64)
        tree = self.node >= 0
        np.add.at(rec, np.unique(self.node[tree]), 1)
        sub = rec.copy()
        for v in np.argsort(-self.depth, kind="stable"):
            p = self.parent[v]
            if p >= 0:
                sub[p] += sub[v]
        total = rec.sum()
        cand = np.flatnonzero((sub >= lo * total) & (sub <= hi * total) & ~self.on_route)
        rng = np.random.default_rng(seed)
        rng.shuffle(cand)
        val = np.zeros(n, bool)
        kids = [[] for _ in range(n)]
        for v in range(n):
            if self.parent[v] >= 0:
                kids[self.parent[v]].append(v)
        covered = 0
        for r in cand:
            if covered >= frac * total:
                break
            # Skip roots inside or above an already chosen subtree.
            a, inside = r, False
            while a >= 0:
                if val[a]:
                    inside = True
                    break
                a = self.parent[a]
            if inside:
                continue
            stack, members = [r], []
            while stack:
                x = stack.pop()
                members.append(x)
                stack.extend(kids[x])
            if val[members].any():
                continue
            val[members] = True
            covered += sub[r]
        self.n_val_roots = int(covered)
        return val

    def setup_pairs(self, split, val_tas_levels=(11, 22)):
        """Splits samples into train / val and builds the pair indexes.
        Tree: whole subtrees (val_subtrees) are validation. TAS: the levels
        in `val_tas_levels` are validation, never trained on."""
        g = self.sch.gi
        level = self.gi[:, g["level"]]
        tree = self.node >= 0
        val_node = self.val_subtrees()
        val = np.where(tree, val_node[np.maximum(self.node, 0)], np.isin(level, val_tas_levels))
        self.split = split
        self.mask = val if split == "val" else ~val
        self.stars = self.gi[:, g["stars"]].astype(np.int64)
        self.tree_idx = np.flatnonzero(self.mask & tree)
        self.tas_idx = np.flatnonzero(self.mask & ~tree)
        # Tree: samples per node, recorded descendant nodes per node.
        ti = self.tree_idx
        order = ti[np.lexsort((self.frame[ti], self.node[ti]))]
        nodes, start, count = np.unique(self.node[order], return_index=True, return_counts=True)
        self.rec_nodes = nodes
        self.node_samples = order
        self.node_start = dict(zip(nodes.tolist(), start.tolist()))
        self.node_count = dict(zip(nodes.tolist(), count.tolist()))
        recorded = np.zeros(len(self.parent), bool)
        recorded[nodes] = True
        anc, desc = [], []
        cur = self.parent[nodes]
        rows = np.arange(len(nodes))
        while len(rows):
            ok = cur >= 0
            rows, cur = rows[ok], cur[ok]
            hit = recorded[cur]
            anc.append(cur[hit])
            desc.append(nodes[rows[hit]])
            cur = self.parent[cur]
        anc, desc = np.concatenate(anc), np.concatenate(desc)
        o = np.argsort(anc, kind="stable")
        self.desc_of = desc[o]
        a_sorted = anc[o]
        u, s, c = np.unique(a_sorted, return_index=True, return_counts=True)
        self.desc_start = dict(zip(u.tolist(), s.tolist()))
        self.desc_count = dict(zip(u.tolist(), c.tolist()))
        # The same pairs indexed by descendant (ancestor-first anchors).
        o2 = np.argsort(desc, kind="stable")
        self.anc_of = anc[o2]
        u2, s2, c2 = np.unique(desc[o2], return_index=True, return_counts=True)
        self.anc_nodes = u2
        self.anc_start, self.anc_cnt = s2, c2
        # Tree samples whose edge has a recorded descendant edge: the only
        # anchors with long-horizon tree goals (~22 % of tree samples).
        hd = np.array([self.desc_count.get(int(n), 0) > 0 for n in self.node[self.tree_idx]], bool)
        self.tree_idx_desc = self.tree_idx[hd] if hd.any() else self.tree_idx
        # TAS: samples sorted by frame.
        self.tas_sorted = self.tas_idx[np.argsort(self.frame[self.tas_idx])]
        self.tas_frames = self.frame[self.tas_sorted]
        return self

    # -- per-sample arrays ----------------------------------------------------
    def objects(self, idx):
        sch = self.sch
        B = len(idx)
        oi = np.zeros((B, sch.max_obj, sch.n_oi), np.int32)
        of = np.zeros((B, sch.max_obj, sch.n_of), np.float32)
        om = np.zeros((B, sch.max_obj), bool)
        for b, i in enumerate(idx):
            s = self.shards[self.shard_of[i]]
            li = self.local[i]
            a, n = s.ooff[li], s.on[li]
            oi[b, :n] = s.oi[a : a + n]
            of[b, :n] = s.of[a : a + n]
            om[b, :n] = True
        return oi, of, om

    def geometry(self, idx, goal_pos=None):
        """N_STATIC_GEO nearest static triangles of the sample's zone (GOAL_GEO
        of them nearest the segment toward `goal_pos` when given) + the
        sample's object triangles."""
        sch = self.sch
        B = len(idx)
        tri = np.zeros((B, N_GEO, sch.n_tri), np.float32)
        tm = np.zeros((B, N_GEO), bool)
        g, f = sch.gi, sch.gf
        for b, i in enumerate(idx):
            key = (int(self.gi[i, g["level"]]), int(self.gi[i, g["area"]]))
            pos = self.gf[i, [f["pos_x"], f["pos_y"], f["pos_z"]]]
            t = self.zones.get(key)
            if t is not None and len(t):
                c = self.zone_cent[key]
                d = np.linalg.norm(c - pos, axis=1)
                take = N_STATIC_GEO
                chosen = []
                if goal_pos is not None and np.isfinite(goal_pos[b]).all():
                    seg = goal_pos[b] - pos
                    L = np.linalg.norm(seg)
                    if L > 50:
                        u = seg / L
                        tpar = np.clip((c - pos) @ u, 0, min(L, 3000))
                        ds = np.linalg.norm(c - pos - tpar[:, None] * u, axis=1) + 0.1 * tpar
                        k = min(GOAL_GEO, len(t))
                        chosen = np.argpartition(ds, k - 1)[:k]
                        d = d.copy()
                        d[chosen] = np.inf
                rest = min(take - len(chosen), len(t) - len(chosen))
                near = np.argpartition(d, rest - 1)[:rest] if rest > 0 else np.zeros(0, np.int64)
                sel = np.concatenate([np.asarray(chosen, np.int64), near])
                tri[b, : len(sel)] = t[sel]
                tm[b, : len(sel)] = True
            s = self.shards[self.shard_of[i]]
            li = self.local[i]
            a, n = s.toff[li], s.tn[li]
            tri[b, N_STATIC_GEO : N_STATIC_GEO + n] = s.tri[a : a + n]
            tm[b, N_STATIC_GEO : N_STATIC_GEO + n] = True
        return tri, tm

    def state(self, idx, goal_pos=None):
        oi, of, om = self.objects(idx)
        tri, tm = self.geometry(idx, goal_pos)
        return dict(gi=self.gi[idx], gf=self.gf[idx], pad_prev=self.pads[idx, 3].astype(np.int64),
                    oi=oi, of=of, om=om, tri=tri, tm=tm)

    def pos(self, idx):
        f = self.sch.gf
        return self.gf[idx][:, [f["pos_x"], f["pos_y"], f["pos_z"]]]


def unpack_pads(p):
    """u32 pads -> (buttons [.., 14] float, stick x, stick y as int)."""
    p = p.astype(np.int64)
    btn = (p >> 16) & 0xFFFF
    bits = np.stack([(btn >> b) & 1 for b in BUTTON_BITS], -1).astype(np.float32)
    sx = ((p >> 8) & 0xFF).astype(np.int8).astype(np.int64)
    sy = (p & 0xFF).astype(np.int8).astype(np.int64)
    return bits, sx, sy


def atan2s(y, x):
    """sm64's atan2s(y, x): 0 along +y, 0x4000 along +x (65536 per turn)."""
    return np.arctan2(x, y) * (65536 / (2 * np.pi))


class Batcher:
    """Builds training batches: anchors a, k-step futures b, goals g."""

    def __init__(self, store, batch=128, tas_frac=0.25, seed=0, desc_frac=0.4, star_frac=0.3):
        self.s, self.B, self.tas_frac, self.desc_frac = store, batch, tas_frac, desc_frac
        # Share of tree anchors drawn from states with time-to-star labels
        # (Store.star_labels; few tree states have one).
        self.star_frac = star_frac
        lab = getattr(store, "lab_off", None)
        self.labelled = store.tree_idx[np.diff(lab)[store.tree_idx] > 0] if lab is not None else np.zeros(0, np.int64)
        self.rng = np.random.default_rng(seed)

    def anchors(self, n):
        s, r = self.s, self.rng
        nt = r.binomial(n, self.tas_frac) if len(s.tas_idx) else 0
        if not len(s.tree_idx):
            return r.choice(s.tas_idx, n)
        # Tree anchors: desc_frac from edges with recorded descendants (long
        # tree goals), the rest uniform.
        nd = r.binomial(n - nt, self.desc_frac)
        ns = r.binomial(n - nt - nd, self.star_frac) if len(self.labelled) else 0
        a = np.concatenate([r.choice(s.tas_idx, nt), self.ancestor_anchors(nd),
                            r.choice(self.labelled, ns) if ns else np.zeros(0, np.int64),
                            r.choice(s.tree_idx, n - nt - nd - ns)])
        r.shuffle(a)
        return a

    def ancestor_anchors(self, n):
        """Samples of a recorded ancestor of a random recorded node: anchors
        high in the tree, whose descendants reach far (long tree goals)."""
        s, r = self.s, self.rng
        if not len(s.anc_nodes):
            return r.choice(s.tree_idx, n)
        k = r.integers(len(s.anc_nodes), size=n)
        anc = s.anc_of[s.anc_start[k] + (r.random(n) * s.anc_cnt[k]).astype(np.int64)]
        out = np.empty(n, np.int64)
        for i, v in enumerate(anc.tolist()):
            st, c = s.node_start[v], s.node_count[v]
            out[i] = s.node_samples[st + r.integers(c)]
        return out

    def goal_for(self, a):
        """A later sample on a's path: (g, frames, tight)."""
        s, r = self.s, self.rng
        if s.node[a] < 0:
            fa = s.frame[a]
            j0 = np.searchsorted(s.tas_frames, fa, "right")
            if j0 >= len(s.tas_frames):
                return -1, 0, True
            # Log-uniform gap from 4 frames to 30k.
            gap = np.exp(r.uniform(np.log(4), np.log(30000)))
            j = min(np.searchsorted(s.tas_frames, fa + gap), len(s.tas_frames) - 1)
            j = max(j, j0)
            g = s.tas_sorted[j]
            return g, int(s.frame[g] - fa), True
        n = int(s.node[a])
        # Short (same chain) 30%, else a descendant node's sample or later in this edge.
        if r.random() < 0.3 and s.chain_len[a] > 0:
            j = r.integers(s.chain_len[a])
            g = s.chain[a, j]
            if s.mask[g]:
                return g, 4 * (j + 1), False
        cands = []
        st, c = s.node_start.get(n), s.node_count.get(n, 0)
        own = s.node_samples[st : st + c] if st is not None else np.zeros(0, np.int64)
        own = own[s.frame[own] > s.frame[a]]
        dc = s.desc_count.get(n, 0)
        if dc and (len(own) == 0 or r.random() < 0.85):
            d = s.desc_of[s.desc_start[n] + r.integers(dc)]
            ds, dn = s.node_start[int(d)], s.node_count[int(d)]
            cands = s.node_samples[ds : ds + dn]
        else:
            cands = own
        if len(cands) == 0:
            return -1, 0, False
        g = cands[r.integers(len(cands))]
        return g, int(s.frame[g] - s.frame[a]), False

    def impossible_goal(self, a):
        """A sample with fewer stars than a: unreachable from a."""
        s, r = self.s, self.rng
        if s.stars[a] <= 0:
            return -1
        pool = s.tree_idx if r.random() < 0.7 else s.tas_idx
        for _ in range(8):
            g = pool[r.integers(len(pool))]
            if s.stars[g] < s.stars[a]:
                return g
        return -1

    def batch(self):
        s, r, B = self.s, self.rng, self.B
        sch = s.sch
        a = self.anchors(B)
        # k-step future.
        k = np.zeros(B, np.int64)
        b = a.copy()
        for i in range(B):
            avail = [kk for kk in KS if s.chain_len[a[i]] >= kk // 4 and s.mask[s.chain[a[i], kk // 4 - 1]]]
            if avail:
                k[i] = avail[r.integers(len(avail))]
                b[i] = s.chain[a[i], k[i] // 4 - 1]
        has_k = k > 0
        # Pads between a and b: 4 per hop, up to 64.
        ch = s.chain[a]  # [B, 16]
        pads = np.zeros((B, 64), np.uint32)
        pmask = np.zeros((B, 64), bool)
        for j in range(HOPS):
            ok = (ch[:, j] >= 0) & (4 * (j + 1) <= k)
            pads[ok, 4 * j : 4 * j + 4] = s.pads[ch[ok, j]]
            pmask[ok, 4 * j : 4 * j + 4] = True
        # Events within h frames, h in KS.
        ev = np.zeros((B, len(KS), len(sch.events)), np.float32)
        evm = np.zeros((B, len(KS)), bool)
        for hi, h in enumerate(KS):
            hop = h // 4
            ok = s.chain_len[a] >= hop
            acc = np.zeros(B, np.int64)
            for j in range(hop):
                c = ch[:, j]
                acc |= np.where(c >= 0, s.events[np.maximum(c, 0)], 0)
            ev[:, hi] = ((acc[:, None] >> np.arange(len(sch.events))) & 1).astype(np.float32)
            evm[:, hi] = ok
        # Goals.
        g = np.full(B, -1, np.int64)
        dist = np.zeros(B, np.float32)
        tight = np.zeros(B, bool)
        for i in range(B):
            g[i], dist[i], tight[i] = self.goal_for(a[i])
        has_g = g >= 0
        # Reachability. Positives: budget >= frames. Budget negatives use the
        # same (a, g) pairs as positives, so only budget vs distance tells
        # them apart: TAS (near-optimal) with < 50 % of its frames, tree with
        # < 1/8 of its frames (tree routes run ~2.3x the TAS's frames per
        # star, RUNS.md; an assumption, not a certainty). Impossible: a goal
        # with fewer stars (stars never go down); trivially separable from
        # PROGRESS, so reported apart. rtype: 0 positive, 1 impossible,
        # 2 TAS budget, 3 tree budget.
        budget = np.zeros(B, np.float32)
        reach = np.zeros(B, np.float32)
        rmask = np.zeros(B, bool)
        rtype = np.full(B, -1, np.int64)
        impossible = np.zeros(B, bool)
        for i in range(B):
            if r.random() < 0.2:
                gi_ = self.impossible_goal(a[i])
                if gi_ >= 0:
                    g[i], impossible[i], has_g[i] = gi_, True, True
                    budget[i] = np.exp(r.uniform(np.log(16), np.log(40000)))
                    rmask[i], rtype[i] = True, 1
                    continue
            if not has_g[i]:
                continue
            can_neg = dist[i] >= (16 if tight[i] else 32)
            if can_neg and r.random() < 0.5:
                budget[i] = dist[i] * (r.uniform(0.05, 0.5) if tight[i] else r.uniform(0.02, 0.125))
                rtype[i] = 2 if tight[i] else 3
            else:
                budget[i] = max(dist[i], 1) * r.uniform(1.0, 3.0)
                reach[i], rtype[i] = 1.0, 0
            rmask[i] = True
        dist_mask = has_g & ~impossible
        gsafe = np.where(has_g, g, a)
        # Policy: the objective that made a's edge (TAS: its own class) and
        # stick/button summary of the next 16 frames.
        pol = s.kind[a]
        nxt16 = np.zeros((B, 16), np.uint32)
        p16 = s.chain_len[a] >= 4
        for j in range(4):
            c = ch[:, j]
            nxt16[:, 4 * j : 4 * j + 4] = s.pads[np.maximum(c, 0)]
        bits, sx, sy = unpack_pads(nxt16)
        gI = sch.gi
        cam = s.gi[a, gI["cam_yaw"]][:, None]
        yaw = s.gi[a, gI["yaw"]][:, None]
        mag = np.minimum(np.sqrt(sx**2 + sy**2), 64) / 64.0
        rel = (atan2s(-sy, sx) + cam - yaw) * (2 * np.pi / 65536)
        # Magnitude-weighted mean direction; (0, 0) when the stick is idle.
        w = mag / np.maximum(mag.sum(1, keepdims=True), 1e-6)
        theta = np.stack([(np.sin(rel) * w).sum(1), (np.cos(rel) * w).sum(1), mag.mean(1),
                          bits[..., 0].mean(1), bits[..., 1].mean(1), bits[..., 2].mean(1)], -1).astype(np.float32)
        pmask_pol = has_g & ~impossible & p16
        # Object deltas a -> b, matched by pool slot and behaviour.
        sa = s.state(a, goal_pos=np.where(has_g[:, None], s.pos(gsafe), np.nan))
        sb = s.state(b)
        sg = s.state(gsafe)
        oS, oB = sch.oi["slot"], sch.oi["bhv"]
        obj_d = np.zeros((B, sch.max_obj, 3), np.float32)
        obj_alive = np.zeros((B, sch.max_obj), np.float32)
        obj_dm = np.zeros((B, sch.max_obj), bool)
        for i in np.flatnonzero(has_k):
            na, nb = sa["om"][i].sum(), sb["om"][i].sum()
            ka = sa["oi"][i, :na, oS].astype(np.int64) * 1024 + sa["oi"][i, :na, oB]
            kb = sb["oi"][i, :nb, oS].astype(np.int64) * 1024 + sb["oi"][i, :nb, oB]
            o = np.argsort(kb)
            pos_ = np.searchsorted(kb[o], ka)
            pos_ = np.minimum(pos_, max(nb - 1, 0))
            hit = (nb > 0) & (kb[o][pos_] == ka) if nb else np.zeros(na, bool)
            obj_alive[i, :na] = hit
            obj_dm[i, :na] = True
            jb = o[pos_[hit]]
            obj_d[i, :na][hit] = sb["of"][i, jb, 0:3] - sa["of"][i, :na][hit, 0:3]
        # Time to star X (Store.star_labels): one label per anchor that has any.
        star_x = np.zeros(B, np.int64)
        star_t = np.zeros(B, np.float32)
        star_m = np.zeros(B, bool)
        lab = getattr(s, "lab_off", None)
        if lab is not None:
            for i in range(B):
                lo, hi = lab[a[i]], lab[a[i] + 1]
                if hi > lo:
                    q = lo + r.integers(hi - lo)
                    star_x[i], star_t[i], star_m[i] = s.lab_x[q], s.lab_t[q], True
        # Edges (v6): a share of the batch's star slots reused for credited
        # objective-rrt edges: anchor stays a (its own label kept) when it
        # has none; an edge's start replaces nothing, its fields ride along.
        edge_i = np.full(B, -1, np.int64)
        if getattr(s, "edge_a", None) is not None and len(s.edge_a):
            for i in range(B):
                if r.random() < 0.25:
                    edge_i[i] = r.integers(len(s.edge_a))
        # Subgoals: where the anchor's path to its star goal is 64 or 256
        # frames on (tree: 64 along the recorded chain; TAS: either), while
        # still before the star.
        sub_dp = np.zeros((B, 3), np.float32)
        sub_h = np.zeros(B, np.int64)
        sub_zone = np.zeros(B, np.float32)
        sub_m = np.zeros(B, bool)
        F = self.s.sch.gf
        pos = lambda i: s.gf[i, [F["pos_x"], F["pos_y"], F["pos_z"]]]
        Gi = self.s.sch.gi
        for i in np.flatnonzero(star_m):
            ai = a[i]
            hh = 1 if (s.node[ai] < 0 and r.random() < 0.5) else 0
            ahead = (64, 256)[hh]
            if ahead >= star_t[i]:
                continue
            if s.node[ai] < 0:
                j = np.searchsorted(s.tas_frames, s.frame[ai] + ahead)
                if j >= len(s.tas_frames) or s.tas_frames[j] != s.frame[ai] + ahead:
                    continue
                b_ = s.tas_sorted[j]
            else:
                if s.chain_len[ai] < ahead // 4:
                    continue
                b_ = s.chain[ai, ahead // 4 - 1]
            sub_dp[i] = pos(b_) - pos(ai)
            sub_h[i] = hh
            sub_zone[i] = float((s.gi[b_, Gi["level"]], s.gi[b_, Gi["area"]]) != (s.gi[ai, Gi["level"]], s.gi[ai, Gi["area"]]))
            sub_m[i] = True
        # Edge rows: their own start/end states (encoded as extra goals would
        # cost encodes; the edge's start is swapped in as the anchor instead).
        e_m = edge_i >= 0
        e_dp = np.zeros((B, 3), np.float32)
        e_zone = np.zeros(B, np.float32)
        e_frames = np.zeros(B, np.float32)
        e_x = np.zeros(B, np.int64)
        e_w = np.zeros(B, np.float32)
        e_kind = np.zeros(B, np.int64)
        e_ev = np.zeros((B, 16), np.float32)
        e_speed = np.zeros(B, np.float32)
        e_style = np.zeros(B, np.int64)
        e_verb = np.zeros(B, np.int64)
        e_obj = np.full(B, -1, np.int64)  # index in the start state's object tokens
        oi_a, _, om_a = s.objects(np.where(e_m, [s.edge_a[k] if k >= 0 else 0 for k in edge_i], 0))
        for i in np.flatnonzero(e_m):
            k_ = edge_i[i]
            ea, eb = s.edge_a[k_], s.edge_b[k_]
            a[i] = ea
            e_dp[i] = pos(eb) - pos(ea)
            e_zone[i] = float((s.gi[eb, Gi["level"]], s.gi[eb, Gi["area"]]) != (s.gi[ea, Gi["level"]], s.gi[ea, Gi["area"]]))
            e_frames[i], e_x[i], e_w[i], e_kind[i] = s.edge_frames[k_], s.edge_x[k_], s.edge_w[k_], s.kind[eb]
            ev = int(s.edge_ev[k_])
            e_ev[i] = (ev >> np.arange(16)) & 1
            e_speed[i] = s.gf[eb, s.sch.gf["fwd_vel"]]
            e_verb[i] = s.edge_verb[k_]
            if s.edge_slot[k_] >= 0:
                hit = np.flatnonzero(oi_a[i, :, s.sch.oi["slot"]][om_a[i]] == s.edge_slot[k_])
                e_obj[i] = hit[0] if len(hit) else -1
            if e_obj[i] < 0:
                e_verb[i] = VERB_NONE
            e_style[i] = edge_style(bool(e_zone[i]), ev, e_speed[i], e_verb[i])
        if e_m.any():
            sa = s.state(a, goal_pos=np.where(has_g[:, None], s.pos(gsafe), np.nan))
        return dict(
            edge_m=e_m, edge_dp=e_dp, edge_zone=e_zone, edge_frames=e_frames, edge_x=e_x, edge_w=e_w, edge_kind=e_kind,
            edge_ev=e_ev, edge_speed=e_speed, edge_style=e_style, edge_verb=e_verb, edge_obj=e_obj,
            sub_dp=sub_dp, sub_h=sub_h, sub_zone=sub_zone, sub_m=sub_m,
            star_x=star_x, star_t=star_t, star_m=star_m, star_tight=(s.node[a] < 0),
            a=sa, b=sb, g=sg,
            k=k, has_k=has_k, pads=pads.astype(np.int64), pmask=pmask,
            ev=ev, evm=evm,
            dist=dist, tight=tight, dist_mask=dist_mask,
            budget=budget, reach=reach, rmask=rmask, impossible=impossible, rtype=rtype,
            g_same_node=(s.node[gsafe] == s.node[a]) & (s.node[a] >= 0),
            pol=pol, theta=theta, pmask_pol=pmask_pol, g_kind=s.kind[gsafe], has_g=has_g,
            obj_d=obj_d, obj_alive=obj_alive, obj_dm=obj_dm,
        )
