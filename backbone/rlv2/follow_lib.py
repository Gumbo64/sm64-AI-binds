"""Trace following: a pool of traces (movies replayed through sm64env), waypoint
sampling, observation features and the network. A trace is any movie from
power-on: a route, a tree node's path, a TAS."""

import glob
import math
import os
import struct

import numpy as np
import torch
import torch.nn as nn

import policy as P
import routes as R
from policy import TAU, symlog

MAX_WP = 140
N_AHEAD = 3
FIELDS = 12  # x y z level area frame vx vy vz speed action yaw


def movies(pattern_list):
    out = []
    for p in pattern_list:
        hits = sorted(glob.glob(os.path.join(R.V2, p)))
        if not hits:
            print("no movie:", p)
        out += hits
    return out


TRAIN = ["runs/q8-g995-s1/first-*.m64", "runs/q8-g995-s1/frontier-*.m64", "runs/q8-g995-s2/first-*.m64", "runs/q8-g995-s2/frontier-*.m64",
         "runs/q8-g995-s3/first-*.m64", "runs/q8-g995-s3/frontier-*.m64", "runs/q15-pw1-s1/first-*.m64", "runs/q15-pw1-s1/frontier-*.m64",
         "runs/q15-pw1-s2/first-*.m64", "runs/q15-pw1-s2/frontier-*.m64", "logs/kingloop/best-*.m64", "oss2/runs/legs-speed-blj-fresh-s1/route.m64",
         "oss2/runs/legs-fast-king-s1/route.m64", "oss2/runs/legs-bob-king-s1/route.m64", "oss2/runs/legs-bob-chomp-s1/route.m64"]
HELD_OUT = ["runs/q15-pw1-s3/frontier-*.m64", "runs/q15-pw1-s3/first-*.m64", "oss2/runs/legs-fast-king-walk-s3/route.m64"]
TAS = "ref/tas/full-120star-2026-U.m64"
# The fast round. Whole routes held out: King 3557 and the route it was mended
# from, a stretch of the TAS with the BoB speed chain + BLJ, whole TAS levels.
FAST_TRAIN = ["logs/kingloop/best-fast-king-hits.m64", "logs/kingloop/best-fast-king-tree-d.m64", "logs/kingloop/best-fast-king-superjump-c.m64",
              "logs/kingloop/best-fast-king-two-trees.m64", "logs/mend/q15-s2-star4-8562.m64", "oss2/runs/legs-speed-blj-fresh-s1/route.m64"]
FAST_HELD = ["logs/mend/king-3557.m64", "logs/kingloop/best-fast-king-blj-tas.ramp0.m64"]
TAS_HELD_WINDOW = (22397, 22800)
TAS_HELD_LEVELS = (11, 22)
TRAIN_TREES = ["runs/q8-g995-s1", "runs/q8-g995-s2", "runs/q8-g995-s3", "runs/q15-pw1-s1", "runs/q15-pw1-s2"]
HELD_TREES = ["runs/q15-pw1-s3"]


def read_tree(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 0)[0]
    pos, out = 4, []
    for _ in range(n):
        par, k = struct.unpack_from("<iI", data, pos)
        pos += 8
        a = np.frombuffer(data, np.uint8, 4 * k, pos).reshape(-1, 4)
        pads = (a[:, 0].astype(np.uint32) << 24) | (a[:, 1].astype(np.uint32) << 16) | (a[:, 2].astype(np.uint32) << 8) | a[:, 3]
        out.append((par, pads))
        pos += 4 * k
    return out


def tree_movies(run, n, rng):
    """Paths from power-on to `n` random nodes of a tree (uniform over nodes,
    deep ones then come with their ancestors' frames)."""
    rec = read_tree(os.path.join(R.V2, run, "tree.bin"))
    picks = rng.choice(len(rec) - 1, min(n, len(rec) - 1), replace=False)
    out = []
    for node in picks:
        chain, v = [], int(node)
        while v >= 0:
            chain.append(rec[v + 1][1])
            v = rec[v + 1][0]
        out.append((f"{run}#{node}", np.concatenate([rec[0][1]] + chain[::-1])))
    return out


class TracePool:
    """Frames of all traces in flat arrays; a suffix a trace shares with no
    earlier one is what's recorded (the shared prefix replays unrecorded).
    t_first/t_last: each trace's range."""

    def __init__(self, pool, gid, paths, every=64, skip_common=True, max_frames=None, tas_window=None, name="train"):
        self.f, self.action, self.floor, self.level, self.area, self.control, self.pads, self.saves = [], [], [], [], [], [], [], []
        self.ranges, self.names = [], []
        total, seen = 0, []
        self.frame = []
        for path in paths:
            force = None
            if isinstance(path, tuple):
                path, pads, *rest = path
                force = rest[0] if rest else None
            else:
                pads = R.read_movie(path)
            if max_frames:
                pads = pads[:max_frames]
            common, src = 0, None
            if skip_common:
                for q in seen:
                    m = min(len(q[0]), len(pads))
                    neq = np.flatnonzero(q[0][:m] != pads[:m])
                    c = int(neq[0]) if len(neq) else m
                    if c > common:
                        common, src = c, q
            start = max(common, 1646)  # before: boot, the same for every run
            if tas_window and path.endswith(TAS.split("/")[-1]):
                start = tas_window[0]
                pads = pads[: tas_window[1]]
            if force is not None:
                start = force
                common, src = 0, None
            if len(pads) - start < 32:
                continue
            # From the nearest earlier savestate on the shared prefix, if any.
            frm, handle = 0, None
            if src is not None:
                for fr, h in src[1]:
                    if fr <= start and fr > frm:
                        frm, handle = fr, h
            if handle is not None:
                pool.load([gid], [handle])
                if start > frm:
                    pool.step([gid], pads[frm:start][None, :])
            else:
                pool.reset([gid])
                pool.step([gid], pads[:start][None, :])
            r = pool.tr_record(gid, pads[start:], every if every else 0)
            n = len(pads) - start
            self.f.append(r["f"]), self.action.append(r["action"]), self.floor.append(r["floor_type"])
            self.level.append(r["level"]), self.area.append(r["area"]), self.control.append(r["control"]), self.pads.append(pads[start:])
            self.frame.append(np.arange(start, start + n))
            k = len(self.ranges)
            self.saves += [(total + t, h) for t, h in r["saves"]]
            if skip_common:
                seen.append((pads, [(start + t, h) for t, h in r["saves"]]))
            self.ranges.append((total, total + n))
            self.names.append(path)
            total += n
        cat = lambda v: np.concatenate(v)
        self.f, self.action, self.floor, self.level, self.area = cat(self.f), cat(self.action), cat(self.floor), cat(self.level), cat(self.area)
        self.control, self.pads, self.frame = cat(self.control), cat(self.pads), cat(self.frame)
        self.tid = np.zeros(total, np.int32)
        self.end = np.zeros(total, np.int64)
        for i, (a, b) in enumerate(self.ranges):
            self.tid[a:b] = i
            self.end[a:b] = b - 1
        # The zone code; a trace position's zone run's last frame isn't needed.
        self.pos = self.f[:, :3]
        self.vel = self.f[:, 3:6]
        self.speed = self.f[:, 6]
        print(f"{name}: {len(self.ranges)} traces, {total} frames, {len(self.saves)} saves, {pool.save_bytes() / 1e6:.0f} MB saves", flush=True)

    def save_arrays(self, path):
        np.savez(path, f=self.f, action=self.action, floor=self.floor, level=self.level, area=self.area, control=self.control, pads=self.pads, frame=self.frame, tid=self.tid, end=self.end)

    @classmethod
    def from_arrays(cls, path):
        self = cls.__new__(cls)
        z = np.load(path)
        for k in z.files:
            setattr(self, k, z[k])
        self.pos, self.vel, self.speed = self.f[:, :3], self.f[:, 3:6], self.f[:, 6]
        self.saves = []
        return self

    def waypoints(self, t0, spacing, horizon):
        """Waypoints (flat index t0 + spacing, ...) to at most t0 + horizon,
        clipped at the trace's end: float32 [N, FIELDS]."""
        e = min(int(self.end[t0]), t0 + horizon)
        idx = list(range(t0 + spacing, e + 1, spacing))
        if not idx or idx[-1] != e:
            idx.append(e)
        # A warp or door on the way: the last control frame before it is a
        # waypoint too (sparse waypoints would otherwise skip the door, and
        # one in another zone can't be steered at).
        zc = self.level[t0 : e + 1] * 1000 + self.area[t0 : e + 1]
        for c in (np.flatnonzero(zc[1:] != zc[:-1]) + t0).tolist():
            b = c
            while b > t0 and not self.control[b]:
                b -= 1
            if self.control[b] and self.level[b] * 1000 + self.area[b] == self.level[c] * 1000 + self.area[c]:
                idx.append(b)
        idx = sorted(set(idx))
        # In a cutscene a position means nothing to steer at: the next control frame.
        idx = [t for t in idx if self.control[t]] or [e]
        idx = idx[-MAX_WP:]
        idx = np.array(idx)
        return np.stack([self.pos[idx, 0], self.pos[idx, 1], self.pos[idx, 2], self.level[idx], self.area[idx], idx,
                         self.vel[idx, 0], self.vel[idx, 1], self.vel[idx, 2], self.speed[idx], self.action[idx] & 0x1FF, self.f[idx, 7]], 1).astype(np.float32)


def wrap_pi(a):
    return torch.remainder(a + math.pi, TAU) - math.pi


def tsymlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def fast_sets(pool, gid, every_train, nodes, seed=7):
    """The fast round's train and held-out pools. The TAS's held-out stretches
    are masked out of training (no starts, no BC)."""
    tas = R.read_movie(os.path.join(R.V2, TAS))
    rng = np.random.default_rng(seed)
    trees = [m for r in ("runs/q8-g995-s1", "runs/q15-pw1-s1") for m in tree_movies(r, nodes, rng)]
    tr = TracePool(pool, gid, movies(FAST_TRAIN) + [("TAS-all", tas, 1646)] + trees, every=every_train, name="fast-train")
    ti = tr.names.index("TAS-all")
    a, b = tr.ranges[ti]
    mask = np.zeros(len(tr.tid), bool)
    mask[a + TAS_HELD_WINDOW[0] - 1646 : a + TAS_HELD_WINDOW[1] - 1646] = True
    lv = tr.level[a:b]
    mask[a:b] |= np.isin(lv, TAS_HELD_LEVELS)
    tr.control = tr.control & ~mask
    tr.saves = [s for s in tr.saves if not mask[s[0]]]
    # The held-out TAS stretches.
    wins = [("TAS-bob", tas[: TAS_HELD_WINDOW[1]], TAS_HELD_WINDOW[0])]
    idx = np.flatnonzero(np.isin(lv, TAS_HELD_LEVELS))
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
    for r in runs:
        if len(r) > 300:
            s = int(r[0]) + 1646
            wins.append((f"TAS-lv{lv[r[0]]}-{s}", tas[: s + len(r)], s))
    held = tree_movies("runs/q15-pw1-s3", 40, np.random.default_rng(11))
    va = TracePool(pool, gid, movies(FAST_HELD) + wins + movies(["oss2/runs/legs-fast-king-walk-s3/route.m64"]) + held, every=every_train // 3 if every_train else 0, name="fast-held")
    return tr, va


def _rot(dx, dz, yaw):
    s, c = torch.sin(yaw), torch.cos(yaw)
    return dx * s + dz * c, dx * c - dz * s


def goal_feat(dp, valid, yaw):
    fw, ri = _rot(dp[:, 0], dp[:, 2], yaw)
    d = torch.hypot(dp[:, 0], dp[:, 2])
    br = torch.atan2(ri, fw)
    near = (d < 800).float()
    out = torch.stack([tsymlog(fw / 200), tsymlog(ri / 200), tsymlog(dp[:, 1] / 200), tsymlog(d / 200), torch.sin(br), torch.cos(br),
                       fw / 400 * near, ri / 400 * near, valid.float()], 1)
    return out * valid[:, None].float()


N_FEAT = 18 + 12 + N_AHEAD * (9 + 4) + 4


def features(f, W, valid, frames_left, radius, spacing):
    """f: rl_obs columns [n, 32] on the device; W: the next N_AHEAD waypoints
    [n, N_AHEAD, FIELDS]; valid [n, N_AHEAD] bool; frames_left to the trace's
    end from the current waypoint. Torch only, so training and inference run
    the one implementation. -> float32 [n, N_FEAT] (clipped: RAM holds odd
    values, e.g. a stale wall pointer's normal), reference action ids."""
    yaw = f[:, 7] * TAU / 65536
    cam = f[:, 9] * TAU / 65536
    vf, vr = _rot(f[:, 3], f[:, 5], yaw)
    wf, wr = _rot(f[:, 17], f[:, 19], yaw)
    nf, nr = _rot(f[:, 14], f[:, 16], yaw)
    base = torch.stack([vf / 64, vr / 64, f[:, 4] / 64, f[:, 6] / 64, tsymlog(f[:, 6] / 32), (f[:, 6].abs() > 48).float(),
                        torch.sin(yaw), torch.cos(yaw), torch.sin(yaw - cam), torch.cos(yaw - cam), f[:, 10] / 4096, f[:, 8] / 16384,
                        tsymlog(f[:, 11] / 32), tsymlog(f[:, 12] / 32), tsymlog(f[:, 13].clamp(min=-500) / 32), nf, nr, f[:, 15]], 1)
    more = torch.stack([wf, wr, f[:, 18], f[:, 20], f[:, 21] / 32, f[:, 22] / 8, f[:, 23], f[:, 24], f[:, 25], f[:, 26], f[:, 27] / 64, f[:, 28] / 64], 1)
    parts = [base, more]
    for k in range(N_AHEAD):
        w, v = W[:, k], valid[:, k]
        parts.append(goal_feat(w[:, :3] - f[:, :3], v, yaw))
        rvf, rvr = _rot(w[:, 6], w[:, 8], yaw)
        parts.append(torch.stack([rvf / 64, rvr / 64, w[:, 7] / 64, w[:, 9] / 64], 1) * v[:, None].float())
    parts.append(torch.stack([tsymlog(frames_left / 32), radius / 200, tsymlog(spacing / 16), (~valid[:, 1]).float()], 1))
    x = torch.nan_to_num(torch.cat(parts, 1), 0.0, 0.0, 0.0).clamp(-8, 8)
    return x, W[:, :, 10].long()


REF_OFF = [0, 1, 2, 4, 8]
N_REF = 6 * len(REF_OFF) + 9


def ref_feat(f, tt, a):
    """The reference near Mario: its inputs at the aligned frame a and just
    after (buttons; stick as a world direction relative to Mario's facing, and
    strength), and its state there relative to him. Any trace has its inputs,
    so a follower may use them: this is what lets it jump where the reference
    jumped. -> float32 [n, N_REF], reference action id [n, 1]."""
    yaw = f[:, 7] * TAU / 65536
    parts = []
    for k in REF_OFF:
        ix = torch.minimum(a + k, tt.end[a])
        p = tt.pads[ix]
        b = (p >> 16) & 0xFFFF
        x, y = (p >> 8) & 0xFF, p & 0xFF
        x = (x - ((x & 0x80) << 1)).float()
        y = (y - ((y & 0x80) << 1)).float()
        ax = torch.where(x.abs() < 8, torch.zeros_like(x), x - 6 * x.sign())
        ay = torch.where(y.abs() < 8, torch.zeros_like(y), y - 6 * y.sign())
        m = torch.hypot(ax, ay)
        world = torch.atan2(ax, -ay) + tt.f[ix, 9] * TAU / 65536
        on = (m > 0).float()
        parts.append(torch.stack([((b >> 15) & 1).float(), ((b >> 14) & 1).float(), ((b >> 13) & 1).float(),
                                  torch.sin(world - yaw) * on, torch.cos(world - yaw) * on, (m / 64).clamp(max=1)], 1))
    rf = tt.f[a]
    dp = rf[:, :3] - f[:, :3]
    fw, ri = _rot(dp[:, 0], dp[:, 2], yaw)
    vf, vr = _rot(rf[:, 3], rf[:, 5], yaw)
    dyaw = (rf[:, 7] - f[:, 7]) * TAU / 65536
    parts.append(torch.stack([tsymlog(fw / 64), tsymlog(ri / 64), tsymlog(dp[:, 1] / 64), vf / 64, vr / 64, rf[:, 4] / 64, torch.sin(dyaw), torch.cos(dyaw), rf[:, 6] / 64], 1))
    return torch.nan_to_num(torch.cat(parts, 1), 0.0, 0.0, 0.0).clamp(-8, 8), (tt.action[a] & 0x1FF)[:, None]


N_FULL = 3 * N_AHEAD


def full_feat(f, W, valid):
    """What a full target adds, per waypoint: facing against Mario's, forward
    speed short of it. Reference speed and velocity are already in `features`."""
    parts = []
    for k in range(N_AHEAD):
        dy = (W[:, k, 11] - f[:, 7]) * TAU / 65536
        v = valid[:, k].float()
        parts.append(torch.stack([torch.sin(dy) * v, torch.cos(dy) * v, (W[:, k, 9] - f[:, 6]) / 64 * v], 1))
    return torch.nan_to_num(torch.cat(parts, 1), 0.0, 0.0, 0.0).clamp(-8, 8)


class Res(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.n, self.a, self.b = nn.LayerNorm(h), nn.Linear(h, h), nn.Linear(h, h)

    def forward(self, x):
        return x + self.b(torch.relu(self.a(self.n(x))))


class FollowNet(nn.Module):
    def __init__(self, d_z=0, h=1024, layers=4, hv=512, ref=False, copy=False):
        super().__init__()
        self.act = nn.Embedding(512, 32)
        self.floor = nn.Embedding(256, 8)
        self.d_z = d_z
        if d_z:
            self.z = nn.Sequential(nn.Linear(d_z, 256), nn.LayerNorm(256), nn.GELU())
        inp = N_FEAT + 32 * (1 + N_AHEAD) + 23 + 8 + (256 if d_z else 0) + ((N_REF + 32) if ref else 0)
        self.pi_body = nn.Sequential(nn.Linear(inp, h), *[Res(h) for _ in range(layers)], nn.LayerNorm(h), nn.ReLU())
        self.v_body = nn.Sequential(nn.Linear(inp, hv), *[Res(hv) for _ in range(2)], nn.LayerNorm(hv), nn.ReLU())
        self.dir, self.mag, self.btn, self.hold = nn.Linear(h, P.N_DIR + (1 if copy else 0)), nn.Linear(h, len(P.MAGS)), nn.Linear(h, len(P.BTNS)), nn.Linear(h, len(P.HOLDS))
        self.v = nn.Linear(hv, 1)
        for head in (self.dir, self.mag, self.btn, self.hold):
            nn.init.orthogonal_(head.weight, 0.01)
            nn.init.zeros_(head.bias)

    def forward(self, x, ract, act, floor, z=None):
        aid, bits = act & 0x1FF, (act >> 9) & 0x7FFFFF
        fl = ((bits[:, None] >> torch.arange(23, device=bits.device)) & 1).float()
        p = [x, self.act(aid), self.act(ract).flatten(1), fl, self.floor(floor)]
        if self.d_z:
            p.append(self.z(z))
        i = torch.cat(p, 1)
        q = self.pi_body(i)
        return (self.dir(q), self.mag(q), self.btn(q), self.hold(q)), self.v(self.v_body(i)).squeeze(-1)


class TraceT:
    """A TracePool's arrays on the device, for hindsight BC at GPU speed.
    Frames are drawn with extra weight on fast play."""

    def __init__(self, tp, device, fast_weight=3.0, copy=False):
        tt = lambda v, ty=None: torch.as_tensor(np.asarray(v), device=device, dtype=ty)
        self.dev = device
        self.copy = copy
        self.f = tt(tp.f, torch.float32)
        self.action = tt(tp.action, torch.int64)
        self.floor = tt(tp.floor, torch.int64)
        self.level = tt(tp.level, torch.int64)
        self.area = tt(tp.area, torch.int64)
        self.control = tt(tp.control, torch.bool)
        self.pads = tt(tp.pads.astype(np.int64), torch.int64)
        self.end = tt(tp.end, torch.int64)
        w = np.where(np.asarray(tp.control), 1.0, 0.0) * (1 + (fast_weight - 1) * (np.abs(tp.speed) > 40))
        # Standing and waiting teach nothing about following.
        w = w * (0.25 + 0.75 * (np.abs(tp.speed) > 2))
        self.cdf = tt(np.cumsum(w) / w.sum(), torch.float32)
        # Each frame's last frame in the same zone (and trace): hindsight goals stop there.
        zc = np.asarray(tp.level) * 1000 + np.asarray(tp.area)
        ze = np.arange(len(zc))
        for i in range(len(zc) - 2, -1, -1):
            if zc[i] == zc[i + 1] and tp.end[i] == tp.end[i + 1]:
                ze[i] = ze[i + 1]
        self.zend = tt(ze, torch.int64)
        self.rel = tt(np.array(P.ANGLES) * math.pi / 180, torch.float32)
        self.mags = tt(P.MAGS, torch.float32)
        lut = np.zeros(8, np.int64)
        for i, b in enumerate(P.BTNS):
            lut[((b >> 15) & 1) | ((b >> 13) & 2) | ((b >> 11) & 4)] = i
        self.blut = tt(lut, torch.int64)

    def row(self, ix):
        p, v = self.f[ix, :3], self.f[ix, 3:6]
        return torch.cat([p, self.level[ix, None].float(), self.area[ix, None].float(), ix[:, None].float(), v, self.f[ix, 6, None], (self.action[ix] & 0x1FF)[:, None].float(), self.f[ix, 7, None]], 1)

    def batch(self, n, spacings, radii, ref=False):
        d = self.dev
        t = torch.searchsorted(self.cdf, torch.rand(n, device=d)).clamp(max=len(self.cdf) - 1)
        S = spacings[torch.randint(len(spacings), (n,), device=d)]
        Rd = radii[torch.randint(len(radii), (n,), device=d)].float()
        phase = torch.randint(0, 1 << 20, (n,), device=d)
        nxt = t + S - ((t + phase) % S)
        end = self.end[t]
        hz = S * torch.randint(1, MAX_WP // 2, (n,), device=d)
        last = torch.minimum(self.zend[t], t + hz)
        nxt = torch.minimum(nxt, last)
        W = torch.stack([self.row(torch.minimum(nxt + k * S, last)) for k in range(N_AHEAD)], 1)
        valid = torch.stack([(nxt + k * S <= last) | (k == 0) for k in range(N_AHEAD)], 1)
        x, ract = features(self.f[t], W, valid, (last - nxt).clamp(min=0).float(), Rd, S.float())
        if ref:
            # The aligned frame online is never exact: jitter it, and sometimes
            # far off, so that a reference that doesn't match the state is seen
            # (a copy is only the right answer where it does).
            far = torch.rand(n, device=d) < (0.4 if self.copy else 0.0)
            j = torch.where(far, torch.randint(-60, 61, (n,), device=d), torch.randint(-4, 6, (n,), device=d) if not self.copy else torch.randint(0, 2, (n,), device=d) * 0)
            a = (t + j).clamp(min=0, max=len(self.end) - 1)
            a = torch.where(self.end[a] == end, a, t)
            rx, ra = ref_feat(self.f[t], self, a)
            x, ract = torch.cat([x, rx], 1), torch.cat([ract, ra], 1)
        y = self.labels(t, W[:, 0, :3], end)
        if ref and self.copy:
            same = (self.pads[a] == self.pads[t]) & ((a - t).abs() <= 1)
            y[:, 0] = torch.where(same, torch.full_like(y[:, 0], P.N_DIR), y[:, 0])
        return x, ract, self.action[t], (self.floor[t] + 1).clamp(0, 255), y

    def label_of(self, p, f, wp):
        """A pad in state f toward waypoint wp, as hold-1 labels."""
        return self._labels(p, f, wp, None, None)

    def labels(self, t, wp, end):
        return self._labels(self.pads[t], self.f[t], wp, t, end)

    def _labels(self, p, f, wp, t, end):
        b = (p >> 16) & 0xFFFF
        x = (p >> 8) & 0xFF
        y = p & 0xFF
        x = (x - ((x & 0x80) << 1)).float()
        y = (y - ((y & 0x80) << 1)).float()
        ax = torch.where(x.abs() < 8, torch.zeros_like(x), x - 6 * x.sign())
        ay = torch.where(y.abs() < 8, torch.zeros_like(y), y - 6 * y.sign())
        m = torch.hypot(ax, ay)
        btn = self.blut[((b >> 15) & 1) | ((b >> 13) & 2) | ((b >> 11) & 4)]
        mag = (torch.minimum(m / 64, torch.ones_like(m))[:, None] - self.mags).abs().argmin(1)
        cam, yaw = f[:, 9] * TAU / 65536, f[:, 7] * TAU / 65536
        world = torch.atan2(ax, -ay) + cam
        e1 = wrap_pi(world[:, None] - yaw[:, None] - self.rel).abs()
        by = torch.atan2(wp[:, 0] - f[:, 0], wp[:, 2] - f[:, 2])
        e2 = wrap_pi(world[:, None] - by[:, None] - self.rel).abs()
        m1, b1 = e1.min(1)
        m2, b2 = e2.min(1)
        d = 1 + torch.where(m2 < m1, P.N_ANG + b2, b1)
        d = torch.where(m == 0, torch.zeros_like(d), d)
        run = torch.ones_like(p)
        if t is not None:
            same = torch.ones_like(t, dtype=torch.bool)
            for k in range(1, 8):
                same = same & (self.pads[torch.minimum(t + k, end)] == p)
                if k in (1, 3, 7):
                    run[same] = k + 1
        hold = torch.searchsorted(torch.tensor(P.HOLDS, device=self.dev), run)
        return torch.stack([d, mag, btn, hold], 1)
