"""Routes for RL: a legs route file (objectives) plus a movie (a demonstration
of it). Replays the movie through the leg objectives, finding each leg's
start and Done frame, and keeps savestates along it to reset from.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))  # v2's sm64env, not the venv's v1 build
sys.path.insert(0, BB)
V2 = os.path.normpath(os.path.join(BB, "..", "objective-rrt-v2"))
ROM = os.path.normpath(os.path.join(BB, "..", "baserom.us.z64"))
ENV = {"OBRRT_MOVES": "1", "OBRRT_MOVE_GA": "1", "OBRRT_MOVE_GA_EXTRA": "1", "OBRRT_MOVES_MODES": "1", "OBRRT_MOVES_EXTRA": "1",
       "OBRRT_MOVES_SEEDED": "1", "OBRRT_CALM": "1", "OBRRT_LAKITU_ARC": "1", "OBRRT_PAINTING_FAST": "1", "OBRRT_GP_CATCH": "1",
       "OBRRT_SWIM_HOLD": "1", "OBRRT_TECH": "all"}  # legs.rs LEGS_DEFAULT_ENV


def setup():
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    import sm64env

    sm64env.Pool.configure()
    return sm64env


def read_movie(path):
    b = open(path, "rb").read()
    if len(b) >= 0x400 and b[:4] == b"M64\x1a":
        n = int.from_bytes(b[0x18:0x1C], "little")
        body = b[0x400:]
        body = body[: min(n * 4, len(body) // 4 * 4)]
    else:
        body = b[: len(b) // 4 * 4]
    a = np.frombuffer(body, np.uint8).reshape(-1, 4)
    btn = (a[:, 0].astype(np.uint32) << 8) | a[:, 1]
    return (btn << 16) | (a[:, 2].astype(np.uint32) << 8) | a[:, 3]


def write_m64(path, pads):
    p = np.asarray(pads, np.uint32)
    out = np.stack([(p >> 24) & 0xFF, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF], -1).astype(np.uint8)
    open(path, "wb").write(out.tobytes())


def make_pool(sm64env, n, threads):
    """As sm64bb.refs.make_pool: zone tables preloaded, so states read before
    Mario first has control in a zone carry the training data's triangles."""
    import glob

    pool = sm64env.Pool(ROM, n, threads)
    best = {}
    for d in ("tree14", "tas120"):
        for z in glob.glob(os.path.join(BB, "data", d, "zones", "*.f32")):
            lv, ar = map(int, os.path.basename(z)[:-4].split("_"))
            t = np.fromfile(z, np.float32).reshape(-1, 18)
            if len(t) > len(best.get((lv, ar), ())):
                best[(lv, ar)] = t
    for (lv, ar), t in best.items():
        pool.set_zone(lv, ar, t)
    return pool


class Demo:
    """A movie folded through a route's legs [first..last] on game `gid`:
    legs[k] = dict(leg, start (frame of the state the leg was bound at),
    control (first controllable frame), done (Done frame or None), saves: list
    of (frame, handle), pads (played, from `start`), ...)."""

    def __init__(self, pool, route, names, pads, first, last, start_frame, gid=0, every=2, overlay=False, stop_frame=None, name="demo"):
        self.name, self.legs = name, []
        pool.reset([gid])
        pool.step([gid], pads[:start_frame][None, :])
        e = pool.rl_bind([gid], route, first, last, overlay)
        if not np.isfinite(e[0]):
            raise RuntimeError(f"{name}: leg {first} ({names[first]}) has no objective at frame {start_frame}")
        i, leg, f = start_frame, first, start_frame
        cur = dict(leg=first, start=f, control=None, done=None, saves=[], name=names[first], eta0=float(e[0]))
        self.term = 0
        while i < len(pads) and (stop_frame is None or f < stop_frame):
            o = pool.rl_obs([gid])
            if bool(o["control"][0]):
                if cur["control"] is None:
                    cur["control"] = f
                if (f - cur["control"]) % every == 0:
                    cur["saves"].append((f, pool.save([gid])[0]))
            r = pool.rl_raw([gid], pads[i : i + 1][None, :])
            n = int(r["frames"][0])
            i_next = i + max(n, 1)
            f += n
            term, done = int(r["term"][0]), int(r["legs_done"][0])
            if done:
                # legs' press-through ran inside; `dones` holds the Done frame.
                _, dones = pool.rl_played(gid)
                cur["done"] = int(dones[-1][1])
                self.legs.append(cur)
                if term:
                    self.term = term
                    break
                leg += 1
                cur = dict(leg=leg, start=f, control=None, done=None, saves=[], name=names[leg], eta0=float(pool.rl_obs([gid])["eta"][0]))
            elif term:
                self.term = term
                cur["term"] = term
                self.legs.append(cur)
                break
            i = i_next
        else:
            self.legs.append(cur)
        self.pads = pads
        self.end_frame = f


def load_route(pool, path):
    text = open(path).read()
    rid, names = pool.rl_route(text)
    return rid, names, json.loads(text)


def first_frame(pool, pads, pred, gid=0, frm=0):
    """Replays from power-on; the first frame >= frm whose status satisfies
    pred (a function of the status dict, row 0)."""
    pool.reset([gid])
    for f in range(len(pads)):
        if f >= frm:
            s = pool.status([gid])
            if pred(s):
                return f
        pool.step([gid], pads[f : f + 1][None, :])
    return None


def in_castle(s):
    return int(s["level"][0]) == 6 and bool(s["controllable"][0])
