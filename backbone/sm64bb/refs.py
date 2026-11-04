"""Reference runs to race or follow: a movie's inputs, its per-frame
progress, and its milestones (each star and key, with which star it was).

A reference is replayed once through sm64env and cached as
`refs/<name>.npz`. Its encodings (the model's z every 4th frame) are added by
`encode_reference` and cached next to it, per checkpoint.

Movies are untrusted input files: only the sample count and the 4-byte
samples are read.
"""

import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))  # sm64-AI-binds
REFS_DIR = os.path.join(HERE, "..", "refs")
ROM = os.path.join(ROOT, "baserom.us.z64")
OBRRT = os.path.join(ROOT, "objective-rrt")
TAS_DIR = os.path.join(OBRRT, "ref", "tas")
BINDS_TAS = os.path.normpath(os.path.join(ROOT, "..", "sm64-binds", "tas"))

# name -> (source, description). Sources: an .m64 path, or ("tree", run, node).
SOURCES = {
    "tree14": (("tree", "indef-mgel-lg-pf-s2", 833074), "objective-rrt's 14-star indefinite route (29:29.8)"),
    "tas120": (os.path.join(TAS_DIR, "full-120star-2026-U.m64"), "120-star TAS, 2026 (1:14:04)"),
    "tas120_2012": (os.path.join(TAS_DIR, "full-120star-2012-U.m64"), "120-star TAS, 2012"),
    "tas120_rikku": (os.path.join(TAS_DIR, "full-120star-rikku-U.m64"), "120-star TAS, 2007 (Rikku)"),
    "tas70": (os.path.join(TAS_DIR, "full-70star-U.m64"), "70-star TAS (no BLJ)"),
    "tas16": (os.path.join(BINDS_TAS, "tasvideos-6943M-16star.m64"), "16-star TAS"),
    "tas0": (os.path.join(BINDS_TAS, "tasvideos-2016M-0star.m64"), "0-star TAS"),
}

STAR_BIT, KEY_BIT = 8, 9
DATA_DIRS = [os.path.join(HERE, "..", "data", d) for d in ("tree14", "tas120")]


def make_pool(n, threads=None):
    """An sm64env pool with every zone table the model was trained on
    preloaded (the largest seen per zone), so states read before Mario first
    has control in a zone carry the same triangles as the training data."""
    import glob

    import sm64env

    pool = sm64env.Pool(ROM, n, threads)
    best = {}
    for d in DATA_DIRS:
        for z in glob.glob(os.path.join(d, "zones", "*.f32")):
            lv, ar = map(int, os.path.basename(z)[:-4].split("_"))
            t = np.fromfile(z, np.float32).reshape(-1, 18)
            if len(t) > len(best.get((lv, ar), ())):
                best[(lv, ar)] = t
    for (lv, ar), t in best.items():
        pool.set_zone(lv, ar, t)
    return pool


def pack(button, x, y):
    return (np.asarray(button, np.uint32) << 16) | ((np.asarray(x).astype(np.int64) & 0xFF).astype(np.uint32) << 8) \
        | (np.asarray(y).astype(np.int64) & 0xFF).astype(np.uint32)


def read_m64(path):
    """Inputs of a Mupen64 .m64 (power-on, the header's sample count) or of
    a raw 4-byte-per-frame replay, as packed uint32 pads."""
    b = open(path, "rb").read()
    if len(b) >= 0x400 and b[:4] == b"M64\x1a":
        n = int.from_bytes(b[0x18:0x1C], "little")
        if int.from_bytes(b[0x1C:0x1E], "little") != 2:
            raise ValueError(f"{path}: movie does not start from power-on")
        body = b[0x400:]
        body = body[: min(n * 4, len(body) // 4 * 4)]
    else:
        body = b[: len(b) // 4 * 4]
    a = np.frombuffer(body, np.uint8).reshape(-1, 4)
    btn = (a[:, 0].astype(np.uint32) << 8) | a[:, 1]
    return (btn << 16) | (a[:, 2].astype(np.uint32) << 8) | a[:, 3]


def write_raw(path, pads):
    """Raw replay (4 bytes per frame: big-endian buttons, stick x, stick y),
    the format viz.py's /api/m64 and the viewer read."""
    p = np.asarray(pads, np.uint32)
    out = np.stack([(p >> 24) & 0xFF, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF], -1).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(out.tobytes())


def source_pads(name):
    src = SOURCES[name][0]
    if isinstance(src, tuple):
        _, run, node = src
        path = os.path.join(REFS_DIR, f"{name}.m64")
        if not os.path.exists(path):
            os.makedirs(REFS_DIR, exist_ok=True)
            subprocess.run([os.path.join(OBRRT, "target", "release", "nodepath"), os.path.join(OBRRT, "runs", run),
                            str(node), path], check=True, capture_output=True)
        return read_m64(path)
    return read_m64(src)


class Reference:
    """pads [T]; per frame t (state after t frames, t = 1..T): stars, keys,
    level, area, pos [T, 3], events; milestones: list of dicts
    {frame, kind ('star'|'key'), course, bit, stars, keys, level, area, pos}."""

    def __init__(self, name, d):
        self.name = name
        for k in d.files:
            setattr(self, k, d[k])
        m = self.milestones_arr
        self.milestones = [dict(frame=int(r[0]), kind="star" if r[1] == 0 else "key", course=int(r[2]), bit=int(r[3]),
                                stars=int(r[4]), keys=int(r[5]), level=int(r[6]), area=int(r[7]),
                                pos=self.pos[int(r[0]) - 1].tolist()) for r in m]
        self.T = len(self.pads)

    def state_frame_index(self, frame):
        """Row for the state after `frame` frames."""
        return frame - 1


def build_reference(name, pool=None, game=0):
    """Replays the reference once, frame by frame, and caches its progress
    after every frame."""
    path = os.path.join(REFS_DIR, f"{name}.npz")
    if os.path.exists(path):
        return Reference(name, np.load(path))
    pads = source_pads(name)
    pool = pool or make_pool(1)
    pool.reset([game])
    T = len(pads)
    stars = np.zeros(T, np.int16)
    keys = np.zeros(T, np.int8)
    level = np.zeros(T, np.int16)
    area = np.zeros(T, np.int8)
    pos = np.zeros((T, 3), np.float32)
    events = np.zeros(T, np.uint32)
    ms = []
    prev = pool.status([game])
    for f in range(T):
        ev, _ = pool.step([game], pads[f : f + 1][None, :])
        e = int(ev[0, 0])
        events[f] = e
        s = pool.status([game])
        stars[f], keys[f] = s["stars"][0], s["keys"][0]
        level[f], area[f] = s["level"][0], s["area"][0]
        pos[f] = s["pos"][0]
        if e & ((1 << STAR_BIT) | (1 << KEY_BIT)):
            cs, cs0 = s["course_stars"][0], prev["course_stars"][0]
            kind = 0 if e & (1 << STAR_BIT) else 1
            diff = np.flatnonzero(cs != cs0)
            course, bit = -1, -1
            if kind == 0 and len(diff):
                course = int(diff[0]) + 1
                bit = int(np.log2((int(cs[diff[0]]) ^ int(cs0[diff[0]])) & 0xFF or 1))
            elif kind == 0:
                # Castle stars live in the save flags (bits 24-30).
                x = (int(s["save_flags"][0]) ^ int(prev["save_flags"][0])) >> 24 & 0x7F
                course, bit = 0, int(np.log2(x)) if x else -1
            ms.append((f + 1, kind, course, bit, int(s["stars"][0]), int(s["keys"][0]), int(s["level"][0]), int(s["area"][0])))
        prev = s
    os.makedirs(REFS_DIR, exist_ok=True)
    np.savez_compressed(path, pads=pads, stars=stars, keys=keys, level=level, area=area, pos=pos, events=events,
                        milestones_arr=np.array(ms, np.int64).reshape(-1, 8))
    return Reference(name, np.load(path))
