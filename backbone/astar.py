#!/usr/bin/env python3
"""Learned A*: collects stars with no objectives, guided only by the
backbone's frames-to-goal estimate, on savestates in sm64env with the model
on the GPU. Shown live in sm64-viewer.

    .venv/bin/python astar.py [--w 1.5] [--star-secs 600] [--stars 3] [--name astar-...]

- Targets: stars in a fixed order (default: the order objective-rrt's
  14-star route collected them, `--targets course:bit,...` to override),
  skipping any already collected. Each target's goal state is the 2026
  120-star TAS's state as it collects that star. (Choosing the target by
  model distance doesn't work: from states unlike the TAS's, the model's
  long-range estimates are unreliable.)
- Weighted A*: priority g + w * h, g = frames since the search started, h =
  the model's frames from the state to the target's state.
- Moves, `--moves ga` (default): objective-rrt attempts, see `search_ga`.
  `--moves basic`, 16 frames each: run in 8 camera-relative directions, each plain,
  jumping (A held), or diving (A then B); long jump (Z then A) in each
  direction; standing still; 2 samples of the goal-conditioned policy; and
  inverse dynamics toward the target once h < 64. Without control (dialogs,
  cutscenes, warps), the moves are pressing A every 4 frames or nothing.
- Duplicates: a state in an already reached cell (position / 16, action,
  zone, stars) no sooner than its first is dropped.
- Decay of repeated failures (`--decay`, frames): each new state's priority
  gains decay x the number of states already added in its coarse area
  (128 units, same zone), so areas the search keeps returning to without
  reaching the goal lose priority.
- Sub-goals: one far goal doesn't work either (from the castle grounds the
  model points away from the castle door), so each star is reached through
  a chain of waypoints from a known route (`--route`, default the 14-star
  tree route): its state just after each zone change (reached on entering
  that zone), one every 300 frames within a zone (reached within 150
  units, and with at least the route's work under way, `St::wip`), one at
  each frame where the route's work goes up (a post pounded, a boss hit, a
  red coin: reached once ours is as high), and the star. Each is one
  weighted A*; the route only says where
  to go next, every input is the search's own.
- Success: any new star (the target or another); the search continues from
  there toward the next target.
Run directory: objective-rrt/runs/<name>/ (viewer: ?tree=<name>).
"""

import argparse
import heapq
import json
import math
import os
import time

import numpy as np
import torch

from follow import encode_reference
from sm64bb.livewrite import LiveRun
from sm64bb.refs import OBRRT, STAR_BIT, build_reference, make_pool, pack
from sm64bb.runtime import Runtime

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 16
SUB_GAP = 300  # route frames between waypoints in one zone
SUB_R = 150.0  # units: how near a waypoint counts as reaching it
A, B, Z = 0x8000, 0x4000, 0x2000
DEAD = (1 << 0) | (1 << 14)


def status_row(s, i):
    return {k: (v[i].tolist() if v.ndim > 1 else v[i].item()) for k, v in s.items()}


def stick(world_angle, cam_yaw, mag=64):
    """Stick (x, y) that makes Mario go toward world angle (radians, s16
    convention: 0 = +z) given the camera yaw (s16): intendedYaw =
    atan2s(-stickY, stickX) + cam yaw."""
    a = world_angle - cam_yaw * 2 * math.pi / 65536
    return int(round(mag * math.sin(a))), int(round(-mag * math.cos(a)))


def moves(st, rng, n_dirs=8):
    """(label, 16 packed pads) for a controllable state."""
    out = []
    cam = st["cam_yaw"]
    yaw = st["yaw"] * 2 * math.pi / 65536
    for d in range(n_dirs):
        ang = yaw + d * 2 * math.pi / n_dirs
        x, y = stick(ang, cam)
        run = pack(np.zeros(CHUNK, np.int64), np.full(CHUNK, x), np.full(CHUNK, y))
        out.append((f"run{d}", run))
        jump = run.copy()
        jump |= A << 16  # A held: full jumps (and doubles on landing)
        out.append((f"jump{d}", jump))
        dive = run.copy()
        dive[:6] |= A << 16
        dive[6:9] |= B << 16
        out.append((f"dive{d}", dive))
        lj = run.copy()
        lj[0:2] |= Z << 16
        lj[1:8] |= A << 16
        out.append((f"longjump{d}", lj))
    out.append(("stand", np.zeros(CHUNK, np.uint32)))
    return out


class AStar:
    def __init__(self, a):
        self.a = a
        self.rng = np.random.default_rng(a.seed)
        self.rt = Runtime(a.ckpt)
        self.P = a.parents
        self.C = 8 * 4 + 1 + 2 + 1  # moves + policy samples + inverse
        # Games: --moves basic steps every child in one; --moves ga only
        # loads attempt ends (the GA workers have their own games).
        self.N = self.P * a.attempts if a.moves == "ga" else self.P * (self.C + 1)
        # objective-rrt's arsenal as the training tree had it (mgel-lg-pf).
        for k_, v_ in {"OBRRT_MOVES": "1", "OBRRT_MOVE_GA": "1", "OBRRT_MOVE_GA_EXTRA": "1", "OBRRT_CALM": "1",
                       "OBRRT_LAKITU_ARC": "1", "OBRRT_PAINTING_FAST": "1", "OBRRT_TECH": "all",
                       "OBRRT_MOVE_WEIGHT": "0.75", "OBRRT_SWIM_HOLD": "1", "OBRRT_GP_CATCH": "1"}.items():
            os.environ.setdefault(k_, v_)
        self.pool = make_pool(self.N + 1)
        import sm64env
        print(json.dumps({"techniques": sm64env.Pool.configure()}), flush=True)
        self.build_goals()
        self.frames = 0

    def build_goals(self):
        ref = build_reference("tas120")
        zr, rframes = encode_reference(ref, self.rt, make_pool(1), 0)
        g = [m for m in ref.milestones if m["kind"] == "star"]
        idx = [int(np.clip(m["frame"] // 4 - 1, 0, len(rframes) - 1)) for m in g]
        self.gz = zr[torch.as_tensor(idx, device=self.rt.dev)].float()
        self.gmeta = [(m["course"], m["bit"], m["pos"], m["level"]) for m in g]
        self.ref = ref
        # The goal states' raw inputs (for the policy's goal tokens).
        self.goal_frames = [rframes[i] for i in idx]

    def route_work(self, route):
        """The route's `wip` after every 4th frame (cached): posts pounded,
        boss damage, red coins, spawned stars."""
        path = os.path.join(HERE, "refs", f"{route.name}.wip4.npy")
        if os.path.exists(path):
            return np.load(path)
        pool = make_pool(1)
        out = np.zeros(route.T // 4, np.int64)
        f = 0
        for i in range(len(out)):
            pool.step([0], route.pads[f : 4 * (i + 1)][None, :])
            f = 4 * (i + 1)
            out[i] = pool.work([0])[0, 1]
        np.save(path, out)
        return out

    def subgoals(self, route, zr, rframes, rms, course, bit):
        """Waypoints along `route` to its star (course, bit): its state just
        after each zone change, every SUB_GAP frames in between, and the
        star itself. Each: embedding, a `test(row)` for reaching it, frame."""
        end = next((m["frame"] for m in rms if (m["course"], m["bit"]) == (course, bit)), None)
        if end is None:
            raise SystemExit(f"route {route.name} never collects star {course}:{bit}")
        prev = max([m["frame"] for m in rms if m["frame"] < end], default=1646)
        zi = lambda f: int(np.clip(f // 4 - 1, 0, len(rframes) - 1))
        work = self.route_work(route)
        wk = lambda f: int(work[min(len(work) - 1, max(0, f // 4 - 1))])
        out = []
        last = prev
        zone = lambda f: (int(route.level[f - 1]), int(route.area[f - 1]))
        for f in range(prev + 1, end):
            if zone(f) != zone(f - 1):
                lv, ar = zone(f)
                g = min(f + 30, end - 1)
                out.append({"kind": "zone", "frame": g, "level": lv, "z": zr[zi(g)], "pos": route.pos[f - 2].copy(),
                            "test": (lambda lv, ar: lambda r: (r["level"], r["area"]) == (lv, ar))(lv, ar)})
                last = f
            elif f % 4 == 0 and wk(f) > wk(f - 4) and route.level[f - 1] != 0:
                # The route's work went up (a post pounded, a boss hit, a red
                # coin): reached when ours is as high, anywhere in the level.
                lv, ar = zone(f)
                w = wk(f)
                out.append({"kind": "work", "frame": f, "level": lv, "z": zr[zi(f)], "pos": route.pos[f - 1].copy(),
                            "test": (lambda lv, w: lambda r: r["level"] == lv and r.get("wip", 0) >= w)(lv, w)})
                last = f
            elif f - last >= SUB_GAP and route.level[f - 1] != 0:
                p = route.pos[f - 1].copy()
                lv, ar = zone(f)
                w = wk(f)
                out.append({"kind": "waypoint", "frame": f, "level": lv, "z": zr[zi(f)], "pos": p,
                            "test": (lambda lv, ar, p, w: lambda r: (r["level"], r["area"]) == (lv, ar)
                                     and np.linalg.norm(np.array(r["pos"]) - p) < SUB_R and r.get("wip", 0) >= w)(lv, ar, p, w)})
                last = f
        if self.a.subgoals == "zone":
            out = [o for o in out if o["kind"] == "zone"]
        out.append({"kind": "star", "frame": end, "level": int(route.level[end - 1]), "z": zr[zi(end)],
                    "pos": route.pos[end - 1].copy(), "test": lambda r: False})
        return out

    # Courses whose star doors / keys the castle requires before they can be
    # entered: (min stars, keys). Game rules, not a route.
    COURSE_NEEDS = {0: (12, 0), 1: (0, 0), 2: (1, 0), 3: (3, 0), 4: (3, 0), 5: (12, 0), 6: (0, 1), 7: (0, 1), 8: (0, 1),
                    9: (30, 1), 10: (0, 2), 11: (0, 2), 12: (0, 2), 13: (0, 2), 14: (0, 2), 15: (0, 2),
                    16: (8, 0), 17: (30, 1), 18: (70, 2), 19: (1, 0), 20: (0, 0), 21: (10, 0), 22: (0, 1),
                    23: (0, 1), 24: (0, 2), 25: (0, 0)}

    def accessible(self, st):
        """Per goal-bank star: whether its course is open from state st."""
        # Main courses and the slide only: the secret stars are almost absent
        # from the trees the model learned from, so its estimates there
        # aren't trustworthy yet.
        return np.array([(1 <= c <= 15 or c == 19) and st["stars"] >= self.COURSE_NEEDS[c][0] and st["keys"] >= self.COURSE_NEEDS[c][1]
                         for c, _, _, _ in self.gmeta])

    def target_order(self):
        """Goal-bank indices in target order."""
        by = {(c, b): i for i, (c, b, _, _) in enumerate(self.gmeta)}
        if self.a.targets and self.a.targets != "auto":
            want = [tuple(map(int, t.split(":"))) for t in self.a.targets.split(",")]
        else:
            want = [(m["course"], m["bit"]) for m in build_reference("tree14").milestones if m["kind"] == "star"]
        order = [by[w] for w in want if w in by]
        print(json.dumps({"target_order": [list(self.gmeta[i][:2]) for i in order]}), flush=True)
        return order

    def uncollected(self, st):
        cs, fl = st["course_stars"], st["save_flags"]
        out = []
        for i, (c, b, _, _) in enumerate(self.gmeta):
            have = (cs[c - 1] >> b & 1) if c > 0 else ((fl >> (24 + b)) & 1 if b >= 0 else 0)
            out.append(not have)
        return np.array(out)

    def h(self, z, zg):
        """Model frames to the goal: a state embedding (sub-goal mode) or a
        goal id (star mode: frames to collecting that star)."""
        if isinstance(zg, int):
            with torch.no_grad():
                return self.rt.m.distance_star(z, torch.full((len(z),), zg, device=self.rt.dev, dtype=torch.long)).cpu().numpy()
        return self.rt.distance(z, zg[None].expand(len(z), -1)).cpu().numpy()

    def run(self):
        a = self.a
        # Boot like every run: the 2026 TAS's menu inputs until Mario has control.
        R = self.N
        self.pool.reset([R])
        f = 0
        while True:
            self.pool.step([R], self.ref.pads[f : f + 1][None, :])
            f += 1
            if self.pool.status([R])["controllable"][0]:
                break
        live = LiveRun(os.path.join(OBRRT, "runs", a.name), self.ref.pads[:f])
        st = status_row(self.pool.status([R]), 0)
        h0 = self.pool.save([R])[0]
        # Node arrays.
        self.parent, self.pads, self.g, self.handle, self.st, self.wid, self.z = [], [], [], [], [], [], []

        def add(par, pads, g, h, st, via, z, write=False):
            i = len(self.parent)
            self.parent.append(par)
            self.pads.append(pads)
            self.g.append(g)
            self.handle.append(h)
            self.st.append(st)
            self.wid.append(None)
            self.z.append(z)
            self.via.append(via)
            if write:
                self.materialize(i, live)
            return i

        self.via = []
        z0 = self.rt.encode(self.pool.read([R]))
        root = add(-1, np.zeros(0, np.uint32), 0, h0, st, "start", z0[0], True)
        results = []
        start, t_all = root, time.time()
        order = self.target_order()
        route = build_reference(a.route)
        zr, rframes = encode_reference(route, self.rt, make_pool(1), 0)
        zr = zr.float()
        rms = [m for m in route.milestones if m["kind"] == "star"]
        for k in range(a.stars):
            unc = self.uncollected(self.st[start])
            todo = [g for g in order if unc[g]]
            if not todo:
                break
            goal = todo[0]
            if a.targets == "auto":
                # The model picks: the uncollected star it puts fewest frames away.
                from sm64bb.data import goal_id
                with torch.no_grad():
                    gids = torch.as_tensor([goal_id({"kind": "star", "course": c_, "bit": b_}) for c_, b_, _, _ in self.gmeta],
                                           device=self.rt.dev)
                    dz = self.rt.m.distance_star(self.z[start][None].expand(len(gids), -1), gids).cpu().numpy()
                dz[~unc] = np.inf
                dz[~self.accessible(self.st[start])] = np.inf
                goal = int(np.argmin(dz))
            c, b, gpos, glv = self.gmeta[goal]
            if a.guide in ("star", "learned"):
                # Only the star: h = model frames to collecting it.
                from sm64bb.data import goal_id
                gid = goal_id({"kind": "star", "course": c, "bit": b})
                subs = [{"kind": "star", "frame": -1, "level": glv, "z": gid, "pos": np.asarray(gpos, np.float32),
                         "test": lambda r: False}]
            else:
                subs = self.subgoals(route, zr, rframes, rms, c, b)
            print(json.dumps({"target": [c, b], "level": glv, "subgoals": [(s_["kind"], s_["frame"], s_["level"]) for s_ in subs]}), flush=True)
            node, stars0 = start, self.st[start]["stars"]
            t_star = time.time()
            for si, sg in enumerate(subs):
                if self.st[node]["stars"] > stars0:
                    break
                found = self.search(node, sg["z"], sg["test"], live, k, results, a.sub_secs, sg.get("pos"))
                if found is None:
                    break
                node = found
                print(json.dumps({"star": k + 1, "subgoal": si + 1, "of": len(subs), "kind": sg["kind"],
                                  "frame": self.st[node]["frame"], "route_frame": sg["frame"]}), flush=True)
                live.flush(force=True, best=self.wid[node])
            if self.st[node]["stars"] <= stars0:
                print(json.dumps({"star": k + 1, "found": False}), flush=True)
                start = node
                break
            start = node
            results.append({"star": k + 1, "frame": self.st[node]["frame"], "course_bit": [c, b],
                            "stars": self.st[node]["stars"], "level": self.st[node]["level"],
                            "route_frame": rms[k]["frame"] if k < len(rms) else None,
                            "secs": round(time.time() - t_star)})
            print(json.dumps(results[-1]), flush=True)
            live.flush(force=True, best=self.wid[node], race={"ref": "astar", "milestones": results})
        from sm64bb.refs import write_raw
        write_raw(os.path.join(live.path, "best.m64"), live.path_pads(self.wid[start]))
        live.flush(force=True)

    def materialize(self, i, live):
        if self.wid[i] is not None:
            return self.wid[i]
        pw = -1 if self.parent[i] < 0 else self.materialize(self.parent[i], live)
        self.wid[i] = live.add(pw, self.pads[i], self.st[i], self.via[i], self.st[i]["frame"])
        return self.wid[i]

    def key(self, st):
        p = st["pos"]
        return (round(p[0] / 16), round(p[1] / 16), round(p[2] / 16), st["action"], st["level"], st["area"], st["stars"],
                st["controllable"], st.get("wip", 0))

    def search(self, start, zg, reached, live, k, results, budget, goal_pos=None):
        if self.a.moves == "ga":
            return self.search_ga(start, zg, reached, live, k, results, budget, goal_pos)
        return self.search_basic(start, zg, reached, live, k, results, budget)

    def search_ga(self, start, zg, reached, live, k, results, budget, goal_pos):
        """Weighted A* whose moves are objective-rrt attempts (sm64env
        Pool.attempt): from each popped state, `--attempts` attempts, each
        on Mario::PosDist toward the sub-goal's position (half of them) or on
        an objective objective-rrt's registry offers there (techniques
        included), with its GA and movement policies. The attempt's end
        state is the child; its cost is the frames it took."""
        a = self.a
        heap = [(0.0, 0, start)]
        best_g = {self.key(self.st[start]): self.g[start]}
        visits = {}
        cnt, it = 0, 0
        stars0 = self.st[start]["stars"]
        t_end = time.time() + budget
        best_h, t0 = 1e18, time.time()
        gp = np.full(3, np.nan, np.float32) if goal_pos is None else np.asarray(goal_pos, np.float32)
        self.retries = {}
        while heap and time.time() < t_end:
            it += 1
            parents = []
            while heap and len(parents) < self.P:
                _, _, i = heapq.heappop(heap)
                if self.handle[i] is not None:
                    parents.append(i)
            if not parents:
                break
            hs = [self.handle[i] for i in parents for _ in range(a.attempts)]
            pars = [i for i in parents for _ in range(a.attempts)]
            goals = np.repeat(gp[None], len(hs), 0)
            if a.guide == "learned" and isinstance(zg, int):
                # The model's own subgoals: sample the subgoal head's edge
                # mode (where a credited objective-rrt edge toward the star
                # ends) per attempt; PosDist aims there.
                goals = self.learned_goals(pars, zg)
            res = self.pool.attempt(hs, goals, 0.5 if a.guide != "learned" else 0.75, int(self.rng.integers(1 << 62)))
            ends = [(q, r) for q, r in enumerate(res) if r["handle"] is not None and len(r["inputs"])]
            self.frames += sum(r["frames"] for r in res)
            # Parents whose attempts all came back empty (a menu objective
            # that failed, a death) go back in the queue to be tried again.
            got = {pars[q] for q, _ in ends}
            for i in parents:
                if i not in got:
                    self.retries[i] = self.retries.get(i, 0) + 1
                    if self.retries[i] <= 8:
                        cnt += 1
                        heapq.heappush(heap, (self.g[i] + 1e6 * 0 + self.retries[i], cnt, i))
            if it <= 3 or not ends:
                kinds = {}
                for r in res:
                    kinds[r["kind"]] = kinds.get(r["kind"], 0) + (1 if r["handle"] is not None else 0)
                print(json.dumps({"iter": it, "attempt_kinds_with_end": kinds, "ends": len(ends)}), flush=True)
            if not ends:
                continue
            ids = list(range(len(ends)))
            self.pool.load(ids, [r["handle"] for _, r in ends])
            st = self.pool.status(ids)
            wip = self.pool.work(ids)[:, 1]
            keep, found = [], None
            for g, (q, r) in enumerate(ends):
                row = status_row(st, g)
                row["wip"] = int(wip[g])
                i = pars[q]
                gg = self.g[i] + len(r["inputs"])
                if row["stars"] > stars0 or reached(row):
                    j = self.new_node(i, r["inputs"], gg, r["handle"], row, r["kind"], None)
                    self.materialize(j, live)
                    if found is None or self.g[j] < self.g[found]:
                        found = j
                    continue
                # Duplicates are pruned only with Mario in control: through a
                # dialog or menu he stands still while it advances.
                key = self.key(row)
                if row["controllable"] and best_g.get(key, 1 << 30) <= gg:
                    self.pool.free([r["handle"]])
                    continue
                if row["controllable"]:
                    best_g[key] = gg
                keep.append((g, q, r, row, gg))
            if found is not None:
                return found
            if self.a.debug:
                print(json.dumps({"iter": it, "ends": len(ends), "kept": len(keep), "heap": len(heap),
                                  "end_states": [(status_row(st, g_)["frame"], status_row(st, g_)["controllable"],
                                                  [round(v) for v in status_row(st, g_)["pos"]], hex(status_row(st, g_)["action"]))
                                                 for g_ in range(len(ends))][:6]}), flush=True)
            if keep:
                z = self.rt.encode(self.pool.read([g for g, _, _, _, _ in keep]))
                hh = self.h(z, zg)
                for n_, (g, q, r, row, gg) in enumerate(keep):
                    j = self.new_node(pars[q], r["inputs"], gg, r["handle"], row, r["kind"], z[n_])
                    p = row["pos"]
                    area = (round(p[0] / 128), round(p[1] / 128), round(p[2] / 128), row["level"], row["area"])
                    v = visits.get(area, 0)
                    visits[area] = v + 1
                    cnt += 1
                    heapq.heappush(heap, (gg + a.w * float(hh[n_]) + a.decay * v, cnt, j))
                    self.materialize(j, live)
                    if hh[n_] < best_h:
                        best_h, self.best_node = float(hh[n_]), j
            live.flush(frames=self.frames, frontier={"keys": 0, "stars": self.st[start]["stars"]},
                       best=self.wid[self.best_node] if getattr(self, "best_node", None) is not None else None,
                       race={"ref": "astar", "milestones": results,
                             "current": {"milestone": k + 1, "best_frames": None, "ref_frames": None, "iters": it,
                                         "frontier": len(heap), "best_h": round(best_h),
                                         "secs_per_iter": round((time.time() - t0) / it, 2)}})
        return None

    @torch.no_grad()
    def learned_goals(self, pars, goal):
        from sm64bb.model import unsymlog_t
        z = torch.stack([self.z[i] for i in pars])
        hidx = torch.full((len(pars),), 2, dtype=torch.long, device=self.rt.dev)
        logit, mu, ls, _ = self.rt.m.subgoal(z, torch.full((len(pars),), goal, dtype=torch.long, device=self.rt.dev), hidx)
        k = torch.distributions.Categorical(logits=logit).sample()
        idx = torch.arange(len(pars), device=self.rt.dev)
        y = mu[idx, k] + torch.randn_like(mu[idx, k]) * ls[idx, k].exp() * 0.5
        dp = (unsymlog_t(y) * 1000).cpu().numpy()
        pos = np.array([self.st[i]["pos"] for i in pars], np.float32)
        return (pos + dp).astype(np.float32)

    def new_node(self, par, pads, g, handle, st, via, z):
        j = len(self.parent)
        self.parent.append(par); self.pads.append(np.asarray(pads, np.uint32)); self.g.append(g)
        self.handle.append(handle); self.st.append(st); self.wid.append(None); self.z.append(z); self.via.append(via)
        return j

    def search_basic(self, start, zg, reached, live, k, results, budget):
        """Weighted A* from node `start` toward the state with embedding zg.
        Returns the first node that collects a new star or satisfies
        `reached(status row)`, or None when `budget` seconds run out."""
        a = self.a
        heap = [(0.0, 0, start)]
        best_g = {self.key(self.st[start]): self.g[start]}
        visits = {}
        cnt, it = 0, 0
        stars0 = self.st[start]["stars"]
        t_end = time.time() + budget
        best_h, t0 = 1e18, time.time()
        while heap and time.time() < t_end:
            it += 1
            parents = []
            while heap and len(parents) < self.P:
                _, _, i = heapq.heappop(heap)
                if self.handle[i] is None and self.handle[self.parent[i]] is None:
                    continue
                parents.append(i)
            if not parents:
                break
            # Build the children's inputs.
            ids, src, rows, meta = [], [], [], []
            rebuild = []
            for p, i in enumerate(parents):
                st = self.st[i]
                pre = None if self.handle[i] is not None else self.pads[i]
                h = self.handle[i] if pre is None else self.handle[self.parent[i]]
                if st["controllable"]:
                    ch = moves(st, self.rng)
                    hh = float(self.h(self.z[i][None], zg)[0])
                    if hh < 64 and not isinstance(zg, int):
                        inv = self.rt.inverse(self.z[i][None], zg[None], int(max(CHUNK, min(64, hh))))[0, :CHUNK]
                        ch.append(("inverse", inv))
                else:
                    ch = [("press_a", pack(np.where(np.arange(CHUNK) % 4 == 0, A, 0), np.zeros(CHUNK), np.zeros(CHUNK))),
                          ("wait", np.zeros(CHUNK, np.uint32))]
                for lab, pads in ch:
                    g = len(ids)
                    ids.append(g)
                    src.append(h)
                    rows.append(pads if pre is None else np.concatenate([pre, pads]))
                    meta.append((i, lab, 0 if pre is None else len(pre), pads))
                if pre is not None:
                    g = len(ids)
                    ids.append(g)
                    src.append(h)
                    rows.append(pre)
                    meta.append((i, None, len(pre), None))
                    rebuild.append((g, i))
            ids = ids[: self.N]
            self.pool.load(ids, src[: len(ids)])
            lens = np.array([len(r) for r in rows[: len(ids)]])
            ev = np.zeros((len(ids), lens.max()), np.uint32)
            taken = np.zeros(len(ids), np.int64)
            for L in sorted(set(lens.tolist())):
                grp = [g for g in ids if lens[g] == L]
                e, tk = self.pool.step(grp, np.stack([rows[g] for g in grp]), stop_mask=(1 << STAR_BIT) | DEAD)
                ev[grp, :L], taken[grp] = e, tk
            self.frames += int(taken.sum())
            for g, i in rebuild:
                self.handle[i] = self.pool.save([g])[0]
            st = self.pool.status(ids)
            alive, found = [], None
            for g in ids:
                i, lab, off, pads = meta[g]
                if lab is None:
                    continue
                e = int(np.bitwise_or.reduce(ev[g, : taken[g]]))
                if e & DEAD:
                    continue
                row = status_row(st, g)
                if (e & (1 << STAR_BIT) and row["stars"] > stars0) or reached(row):
                    hnd = self.pool.save([g])[0]
                    j = len(self.parent)
                    self.parent.append(i); self.pads.append(pads[: int(taken[g]) - off]); self.g.append(self.g[i] + int(taken[g]) - off)
                    self.handle.append(hnd); self.st.append(row); self.wid.append(None); self.z.append(None); self.via.append(lab)
                    self.materialize(j, live)
                    if found is None or self.g[j] < self.g[found]:
                        found = j
                    continue
                key = self.key(row)
                gg = self.g[i] + CHUNK
                if best_g.get(key, 1 << 30) <= gg:
                    continue
                best_g[key] = gg
                alive.append(g)
            if found is not None:
                return found
            if alive:
                z = self.rt.encode(self.pool.read(alive))
                hh = self.h(z, zg)
                for q, g in enumerate(alive):
                    i, lab, off, pads = meta[g]
                    row = status_row(st, g)
                    j = len(self.parent)
                    self.parent.append(i); self.pads.append(pads); self.g.append(self.g[i] + CHUNK)
                    self.handle.append(None); self.st.append(row); self.wid.append(None); self.z.append(z[q]); self.via.append(lab)
                    cnt += 1
                    p = row["pos"]
                    area = (round(p[0] / 128), round(p[1] / 128), round(p[2] / 128), row["level"], row["area"])
                    v = visits.get(area, 0)
                    visits[area] = v + 1
                    heapq.heappush(heap, (self.g[j] + a.w * float(hh[q]) + a.decay * v, cnt, j))
                    if hh[q] < best_h:
                        best_h = float(hh[q])
                        self.best_node = j
                        self.materialize(j, live)
            for i in parents:
                self.materialize(i, live)
            if it % 5 == 0:
                live.flush(frames=self.frames, frontier={"keys": 0, "stars": self.st[start]["stars"]},
                           best=self.wid[self.best_node] if getattr(self, "best_node", None) is not None and self.wid[self.best_node] is not None else None,
                           race={"ref": "astar", "milestones": results,
                                 "current": {"milestone": k + 1, "best_frames": None, "ref_frames": None, "iters": it,
                                             "frontier": len(heap), "best_h": round(best_h),
                                             "secs_per_iter": round((time.time() - t0) / it, 3)}})
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "v3_last.pt"))
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--stars", type=int, default=3)
    ap.add_argument("--sub-secs", type=float, default=120, help="search time per sub-goal")
    ap.add_argument("--route", default="tree14", help="reference whose route gives the sub-goals")
    ap.add_argument("--guide", choices=["route", "star", "learned"], default="route",
                    help="route: sub-goals from --route; star: only the star (model frames to collecting it); "
                         "learned: star mode with the attempts aimed at the model's own subgoals (v6+)")
    ap.add_argument("--subgoals", choices=["all", "zone"], default="all",
                    help="all: zone entries, waypoints and work; zone: zone entries and the star only")
    ap.add_argument("--parents", type=int, default=16)
    ap.add_argument("--targets", default=None, help="course:bit,... in order (default: the 14-star route's); auto: the model's nearest uncollected star")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--moves", choices=["ga", "basic"], default="ga",
                    help="ga: objective-rrt attempts (its GA, movement policies, techniques); basic: 16-frame macros")
    ap.add_argument("--attempts", type=int, default=4, help="objective-rrt attempts per expanded state (--moves ga)")
    ap.add_argument("--decay", type=float, default=0.0, help="priority frames added per earlier state in the same 128-unit area")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    a.name = a.name or f"astar-{time.strftime('%m%d-%H%M')}"
    AStar(a).run()


if __name__ == "__main__":
    main()
