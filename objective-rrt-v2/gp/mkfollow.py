#!/usr/bin/env python3
"""Exp. 3's case file: the RL follower's own held-out starts (backbone/rlv2/eval_follow.py,
grid at H=256, seed 5, 128 starts) rebuilt from its data without a game pool, plus training and
validation starts from its training movies (other runs than the held-out one).

  cd objective-rrt-v2; ../backbone/.venv/bin/python gp/mkfollow.py

Writes gp/cases/follow.txt and the traces as raw movies in runs/gp/follow_movies/."""
import os, random, subprocess, sys, csv, tempfile
import numpy as np
BB = os.path.abspath("../backbone")
sys.path[:0] = [os.path.join(BB, "rlv2")]
import follow_lib as L
import routes as R

OUT = "runs/gp/follow_movies"
os.makedirs(OUT, exist_ok=True)
SPACINGS = [8, 16, 32, 64, 128, 256]
H = 256

def write_raw(path, pads):
    p = np.asarray(pads, np.uint32)
    np.stack([(p >> 24) & 0xFF, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF], -1).astype(np.uint8).tofile(path)

# --- As TracePool's constructor with skip_common, only the bookkeeping.
held = [m for r in L.HELD_TREES for m in L.tree_movies(r, 60, np.random.default_rng(11))]
paths = L.movies(L.HELD_OUT) + held
z = np.load(os.path.join(BB, "data/rlv2/held.npz"))
ctl, pos, speed, end, frame = z["control"], z["f"][:, :3], z["f"][:, 6], z["end"], z["frame"]
seen, ranges, tracks, total = [], [], [], 0
for path in paths:
    if isinstance(path, tuple):
        path, pads = path
    else:
        pads = R.read_movie(path)
    common = 0
    for q in seen:
        m = min(len(q), len(pads))
        neq = np.flatnonzero(q[:m] != pads[:m])
        c = int(neq[0]) if len(neq) else m
        common = max(common, c)
    start = max(common, 1646)
    if len(pads) - start < 32:
        continue
    seen.append(pads)
    n = len(pads) - start
    ranges.append((total, total + n, start))
    tracks.append(pads)
    total += n
assert total == len(ctl), (total, len(ctl))
for (a, b, st), pads in zip(ranges, tracks):
    assert frame[a] == st and frame[b - 1] == st + b - a - 1 and end[a] == b - 1
    assert np.array_equal(z["pads"][a:b], pads[st:]), "held.npz isn't these traces"
print("held.npz matches the traces:", len(ranges), total, flush=True)

# Saves as tr_record(every=32): k % 32 == 0 and in control.
saves = []
tid = np.zeros(total, np.int32)
for i, (a, b, st) in enumerate(ranges):
    tid[a:b] = i
    saves += [a + k for k in range(0, b - a, 32) if ctl[a + k]]

_u = {}
def usable(t, horizon):
    k = (t, horizon)
    if k not in _u:
        _u[k] = usable_(t, horizon)
    return _u[k]

def usable_(t, horizon):
    e = min(int(end[t]), t + horizon)
    if e - t < horizon // 2 or not ctl[t: e + 1].mean() > 0.7:
        return False
    return np.linalg.norm(np.diff(pos[t: e + 1][:, [0, 2]], axis=0), axis=1).sum() > 200

rng = np.random.default_rng(5)
cand = [t for t in saves if usable(t, H)]
STARTS = int(os.environ.get("STARTS", "128"))
idx = rng.choice(len(cand), min(STARTS, len(cand)), replace=False)
lines = []
used = {}
for i in idx:
    t = cand[i]
    e = min(int(end[t]), t + H)
    fast = int(np.percentile(np.abs(speed[t: e + 1]), 75) > 45)
    k = int(tid[t])
    f = f"{OUT}/h{k}.raw"
    if k not in used:
        write_raw(f, tracks[k])
        used[k] = 1
    for S in SPACINGS:
        lines.append(f"test {f} {int(frame[t])} h={H} s={S} fast={fast}")
print("held-out picks", len(idx), "of", len(cand), "traces", len(used), flush=True)

# --- Rewire (eval_follow.py): the same draws in the same order.
def raw_of(k):
    f = f"{OUT}/h{k}.raw"
    if k not in used:
        write_raw(f, tracks[k])
        used[k] = 1
    return f

level, area = z["level"], z["area"]
c1024 = [t for t in saves if usable(t, 1024)]
rng.choice(len(c1024), min(STARTS, len(c1024)), replace=False)
HR = 512
cand = [t for t in saves if usable(t, HR + 64)]
base = [cand[i] for i in rng.choice(len(cand), min(STARTS, len(cand)), replace=False)]
rw = []
E = 64
for K in (15, 30, 60):
    for S in (16, 32, 64, 128, 256):
        pr = None
        for n, t in enumerate(base):
            if n % E == 0:
                pr = np.random.default_rng(K * 1000 + S).uniform(-180, 180, min(E, len(base) - n))
            yaw = pr[n % E]
            x, y = int(64 * np.sin(np.radians(yaw))), int(64 * np.cos(np.radians(yaw)))
            rw.append(f"test {raw_of(int(tid[t]))} {int(frame[t])} noise={K} raw={x}:{y} h={HR} s={S} fast=0 kind=K{K}")
by = {}
for t in saves:
    by.setdefault((int(level[t]), int(area[t])), []).append(t)
pairs = []
for k in rng.permutation(len(saves)):
    tb = saves[k]
    zz = (int(level[tb]), int(area[tb]))
    cs = [c for c in by[zz] if tid[c] != tid[tb] and usable(c, HR)]
    if not cs:
        continue
    ts = np.array(cs)
    d = np.linalg.norm(pos[ts] - pos[tb], axis=1)
    ok_ = np.flatnonzero((d < 300) & (d > 20) & (np.abs(frame[ts] - frame[tb]) > 8))
    if len(ok_):
        j = ok_[np.argmin(d[ok_])]
        pairs.append((int(ts[j]), tb, float(d[j])))
    if len(pairs) >= STARTS:
        break
print("grafts", len(pairs), "median gap", float(np.median([p[2] for p in pairs])), flush=True)
for S in (16, 32, 64, 128, 256):
    for ta, tb, _ in pairs:
        rw.append(f"test {raw_of(int(tid[tb]))} {int(frame[tb])} rmovie={raw_of(int(tid[ta]))} rframe={int(frame[ta])} h={HR} s={S} fast=0 kind=graft")
open("gp/cases/follow_rw.txt", "w").write("\n".join(rw) + "\n")

if STARTS != 128:
    # Only the held-out grid, to check against an eval run at other --starts.
    open(f"gp/cases/follow{STARTS}.txt", "w").write("\n".join(lines) + "\n")
    sys.exit()

# --- Training and validation: H-frame segments of training movies, by tasplay.
rand = random.Random(1)
tr = L.movies(L.TRAIN)
rand.shuffle(tr)
def segs(m, want):
    pads = R.read_movie(m)
    with tempfile.NamedTemporaryFile(suffix=".csv") as t:
        subprocess.run(["builds/tools/tasplay", "../baserom.us.z64", m, t.name], check=True, stderr=subprocess.DEVNULL)
        rows = list(csv.DictReader(open(t.name)))
    cs = np.array([r["controllable"] == "1" for r in rows])
    xz = np.array([[float(r["x"]), float(r["z"])] for r in rows])
    fv = np.array([abs(float(r["fvel"])) for r in rows])
    lv = np.array([int(r["level"]) for r in rows])
    out = []
    # frames: row i is the state after i + 1 pads.
    fr = list(range(1646, len(rows) - H - 2, 32))
    rand.shuffle(fr)
    for f in fr:
        s = cs[f - 1: f + H]
        if s.mean() > 0.7 and cs[f - 1] and np.linalg.norm(np.diff(xz[f - 1: f + H], axis=0), axis=1).sum() > 200:
            out.append((f, int(np.percentile(fv[f - 1: f + H], 75) > 45), int(lv[f - 1])))
            if len(out) >= want:
                break
    return out
val_runs = ("q8-g995-s3",)
ntrain, nval = 0, 0
spt = [16, 64, 256]
mov = 0
for m in tr:
    isval = any(v in m for v in val_runs)
    if (isval and nval >= 30) or (not isval and ntrain >= 45):
        continue
    if not os.path.exists(m):
        continue
    f = None
    ss = segs(m, 2)
    for fr_, fast, lv in ss:
        if f is None:
            f = f"{OUT}/t{mov}.raw"
            write_raw(f, R.read_movie(m))
            mov += 1
        if isval:
            nval += 1
            lines.append(f"val {f} {fr_} h={H} s={spt[nval % 3]} fast={fast}")
        else:
            ntrain += 1
            lines.append(f"train {f} {fr_} h={H} s={spt[ntrain % 3]} fast={fast}")
print("train", ntrain, "val", nval, flush=True)
open("gp/cases/follow.txt", "w").write("\n".join(lines) + "\n")
# Training with perturbed starts: half of the own-state cases, plus K frames of noise on each.
own = [l for l in lines if not l.startswith("test")]
pr = random.Random(2)
tr2 = []
for i, l in enumerate(own):
    f = l.split()
    if i % 2 == 0:
        tr2.append(l)
    for K in (15, 45):
        yaw = pr.uniform(-180, 180)
        import math
        x, y = int(64 * math.sin(math.radians(yaw))), int(64 * math.cos(math.radians(yaw)))
        tr2.append(f"{f[0]} {f[1]} {f[2]} noise={K} raw={x}:{y} {' '.join(f[3:])}")
open("gp/cases/follow_train2.txt", "w").write("\n".join(tr2) + "\n")
