#!/usr/bin/env python3
"""The backbone as a GPU service for objective-rrt (search/src/model.rs),
over a Unix socket. One request at a time, batch 1, so the same input always
gets the same answer (objective-rrt's runs stay reproducible).

    .venv/bin/python serve_model.py [--socket run/model.sock] [--ckpt checkpoints/v3_last.pt]

Frames: u32 op, u32 payload length, payload (little endian). Ops:
- 1 NODE: a model input (see `read_input`) + course_stars[25] u8 + save_flags
  u32 + keys u8 -> z (d f32) + score f32 (model frames to the nearest goal
  not collected yet; 0 if none) + goal index i32.
- 2 POLICY: a model input + goal pos 3 f32 + budget f32 + n u32 + seed u64
  -> n x 6 f32: sampled stick/button summaries for the next 16 frames
  (sin, cos of the stick's yaw relative to Mario's facing, magnitude 0..1,
  fractions of frames with A, B, Z held).
- 3 INVERSE: a model input + target z (d f32) + k u32 -> k u32 packed pads.
- 4 INFO -> json (d, goal count, schema columns, checkpoint step).
- 5 PROPOSE (v6 checkpoints): a model input + course_stars[25] u8 +
  save_flags u32 + keys u8 + n u32 + seed u64 -> n learned objectives toward
  the nearest uncollected goal (distance_star), PROPOSAL bytes each: style u8
  (data.STYLES), verb u8 (data.VERBS), object slot i16 (-1 none), end
  position 3 f32 (world), P(zone change) f32, event mask u16, 2 pad, end
  forward speed f32, budget frames f32, goal id i32.

Goals ("goal bank"): the 2026 120-star TAS's states at each star, and the
70-star TAS's at each key, encoded once at start-up (cached in refs/).
"""

import argparse
import json
import os
import socket
import struct
import threading

import numpy as np
import torch

from follow import encode_reference
from sm64bb.refs import build_reference, make_pool
from sm64bb.runtime import Runtime
from sm64bb.schema import Schema

HERE = os.path.dirname(os.path.abspath(__file__))


class Server:
    def __init__(self, ckpt, star_goals=False):
        self.rt = Runtime(ckpt)
        # --star-goals (v4+): a node's score is the model's frames to
        # collecting each uncollected star directly (distance_star), not its
        # distance to the TAS's state there.
        self.star_goals = star_goals
        self.sch = Schema()
        s = self.sch
        self.shapes = [("gi", np.int32, (s.n_gi,)), ("gf", np.float32, (s.n_gf,)), ("pad_prev", np.uint32, ()),
                       ("oi", np.int32, (s.max_obj, s.n_oi)), ("of", np.float32, (s.max_obj, s.n_of)),
                       ("om", np.uint8, (s.max_obj,)), ("tri", np.float32, (128, s.n_tri)), ("tm", np.uint8, (128,))]
        self.input_bytes = sum(np.dtype(t).itemsize * int(np.prod(sh)) for _, t, sh in self.shapes)
        self.lock = threading.Lock()
        self.build_goals()

    def build_goals(self):
        zs, meta = [], []
        pool = make_pool(1)
        for name, kinds in (("tas120", ("star",)), ("tas70", ("key",))):
            ref = build_reference(name)
            zr, rframes = encode_reference(ref, self.rt, pool, 0)
            for m in ref.milestones:
                if m["kind"] not in kinds:
                    continue
                i = int(np.clip(m["frame"] // 4 - 1, 0, len(rframes) - 1))
                zs.append(zr[i].float())
                meta.append((0 if m["kind"] == "star" else 1, m["course"], m["bit"], m["keys"]))
        self.gz = torch.stack(zs)
        self.gmeta = np.array(meta, np.int64)
        from sm64bb.data import KEY_GOAL
        self.gids = torch.as_tensor([KEY_GOAL + k if kind == 1 else c * 8 + b for kind, c, b, k in meta],
                                    device=self.rt.dev, dtype=torch.long)
        print(f"goal bank: {len(meta)} goals ({int((self.gmeta[:, 0] == 0).sum())} stars, "
              f"{int((self.gmeta[:, 0] == 1).sum())} keys)", flush=True)

    def read_input(self, buf, off):
        x = {}
        for k, t, sh in self.shapes:
            n = np.dtype(t).itemsize * int(np.prod(sh))
            a = np.frombuffer(buf, t, count=int(np.prod(sh)), offset=off).reshape((1,) + sh)
            x[k] = a.astype(np.int64) if k == "pad_prev" else (a.astype(bool) if t == np.uint8 else a)
            off += n
        return x, off

    def uncollected(self, cs, flags, keys):
        g = self.gmeta
        out = np.ones(len(g), bool)
        for i, (kind, course, bit, k) in enumerate(g):
            if kind == 1:
                out[i] = keys < k
            elif course > 0 and bit >= 0:
                out[i] = not (cs[course - 1] >> bit) & 1
            elif course == 0 and bit >= 0:
                out[i] = not (flags >> (24 + bit)) & 1
        return out

    @torch.no_grad()
    def handle(self, op, buf):
        if op == 4:
            return json.dumps({"d": self.rt.m.d, "goals": len(self.gmeta), "step": self.rt.step,
                               "input_bytes": self.input_bytes}).encode()
        x, off = self.read_input(buf, 0)
        if op == 1:
            cs = np.frombuffer(buf, np.uint8, 25, off)
            flags, keys = struct.unpack_from("<IB", buf, off + 25)
            with self.lock:
                z = self.rt.encode(x)
                if self.star_goals:
                    d = self.rt.m.distance_star(z.expand(len(self.gids), -1), self.gids).cpu().numpy()
                else:
                    d = self.rt.distance(z.expand(len(self.gz), -1), self.gz).cpu().numpy()
            mask = self.uncollected(cs, flags, keys)
            if mask.any():
                i = int(np.flatnonzero(mask)[np.argmin(d[mask])])
                score = float(d[i])
            else:
                i, score = -1, 0.0
            return z[0].cpu().numpy().astype(np.float32).tobytes() + struct.pack("<fi", score, i)
        if op == 2:
            gx, gy, gz_, budget, n, seed = struct.unpack_from("<4fIQ", buf, off)
            with self.lock:
                z, h, pm = self.rt.encode(x, need_h=True)
                sg = {"gi": x["gi"].copy(), "gf": x["gf"].copy()}
                f = self.sch.gf
                sg["gf"][0, f["pos_x"]], sg["gf"][0, f["pos_y"]], sg["gf"][0, f["pos_z"]] = gx, gy, gz_
                sg["gf"][0, f["vel_x"]] = sg["gf"][0, f["vel_z"]] = 0.0
                # GOAL_REGION only (GOAL_STATE needs a state; GOAL_OBJECTIVE is off).
                m = self.rt.m
                B = 1
                keep = torch.zeros(B, 3, dtype=torch.bool, device=self.rt.dev)
                keep[:, 0] = True
                ta = {k: torch.from_numpy(np.ascontiguousarray(x[k])).to(self.rt.dev) for k in ("gi", "gf")}
                tg = {k: torch.from_numpy(np.ascontiguousarray(sg[k])).to(self.rt.dev) for k in ("gi", "gf")}
                gt, gm = m.goal_tokens(ta, tg, z, torch.zeros(B, dtype=torch.long, device=self.rt.dev), keep)
                _, pol = m.goal_tail(h, pm, gt, gm, torch.tensor([budget], device=self.rt.dev))
                pol = pol.float()[0]
            nk = m.n_kinds
            mu, lv = pol[nk : nk + 6].cpu().numpy(), pol[nk + 6 :].clamp(-6, 4).cpu().numpy()
            rng = np.random.default_rng(seed)
            th = mu[None] + rng.standard_normal((n, 6)) * np.exp(0.5 * lv)[None]
            th[:, 2:] = np.clip(th[:, 2:], 0, 1)
            return th.astype(np.float32).tobytes()
        if op == 3:
            d = self.rt.m.d
            zt = np.frombuffer(buf, np.float32, d, off)
            (k,) = struct.unpack_from("<I", buf, off + 4 * d)
            with self.lock:
                z = self.rt.encode(x)
                pads = self.rt.inverse(z, torch.from_numpy(zt.copy()).to(self.rt.dev)[None], int(min(max(k, 1), 64)))
            return pads[0].astype(np.uint32).tobytes()
        if op == 5:
            cs = np.frombuffer(buf, np.uint8, 25, off)
            flags, keys, n, seed = struct.unpack_from("<IBIQ", buf, off + 25)
            return self.propose(x, cs, flags, keys, n, seed)
        raise ValueError(f"unknown op {op}")

    def propose(self, x, cs, flags, keys, n, seed):
        from sm64bb.data import SALIENT_EVENTS, STYLES, VERBS
        from sm64bb.model import unsymlog_t
        m, dev = self.rt.m, self.rt.dev
        mask = self.uncollected(cs, flags, keys)
        if not mask.any() or n == 0:
            return b""
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed) & ((1 << 63) - 1))
        with self.lock:
            z, h, pm = self.rt.encode(x, need_h=True)
            d = m.distance_star(z.expand(len(self.gids), -1), self.gids).cpu().numpy()
            gi = int(np.flatnonzero(mask)[np.argmin(d[mask])])
            goal = self.gids[gi : gi + 1].expand(n)
            zn = z.expand(n, -1)
            hidx = torch.full((n,), 2, dtype=torch.long, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit, mu, ls, zl = m.subgoal(zn, goal, hidx)
                eo = m.subgoal_edge(zn, goal)
            logit, mu, ls, zl = logit.float(), mu.float(), ls.float(), zl.float()
            k = torch.multinomial(torch.softmax(logit, -1), 1, generator=g)[:, 0]
            i = torch.arange(n, device=dev)
            y = mu[i, k] + torch.randn(mu[i, k].shape, device=dev, generator=g) * ls[i, k].exp() * 0.5
            dp = (unsymlog_t(y) * 1000).cpu().numpy()
            style = torch.multinomial(torch.softmax(eo["style"], -1), 1, generator=g)[:, 0]
            verb = torch.multinomial(torch.softmax(eo["verb"][:, 1:], -1), 1, generator=g)[:, 0] + 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                plg = m.object_pointer(zn, h.expand(n, -1, -1), pm.expand(n, -1), goal, verb).float()
            ok = torch.isfinite(plg).any(-1)
            j = torch.multinomial(torch.softmax(plg.masked_fill(~ok[:, None], 0), -1), 1, generator=g)[:, 0]
            evp = torch.sigmoid(eo["events"]).cpu().numpy()
            speed = (eo["speed"] * 32).cpu().numpy()
            budget = eo["budget"].expm1().clamp(8, 900).cpu().numpy()
            zp = torch.sigmoid(zl).cpu().numpy()
            style, verb, j, ok = style.cpu().numpy(), verb.cpu().numpy(), j.cpu().numpy(), ok.cpu().numpy()
        f = self.sch.gf
        pos = np.array([x["gf"][0, f["pos_x"]], x["gf"][0, f["pos_y"]], x["gf"][0, f["pos_z"]]], np.float32)
        slot_col = self.sch.oi["slot"]
        out = bytearray()
        rng = np.random.default_rng(seed)
        for q in range(n):
            st = int(style[q])
            vb, slot = 0, -1
            if STYLES[st] == "object":
                if ok[q]:
                    vb, slot = int(verb[q]), int(x["oi"][0, j[q], slot_col])
                else:
                    st = 0
            ev = 0
            if STYLES[st] == "event":
                bits = [b for b in range(16) if (SALIENT_EVENTS >> b) & 1]
                p = evp[q, bits]
                ev = 1 << bits[int(rng.choice(len(bits), p=p / p.sum()))] if p.sum() > 0 else 0
                if not ev:
                    st = 0
            e = pos + dp[q]
            out += struct.pack("<BBh3ffHHffi", st, vb, slot, *e.tolist(), float(zp[q]), ev, 0,
                               float(speed[q]), float(budget[q]), int(self.gids[gi]))
        return bytes(out)


def serve(srv, conn):
    def recv(n):
        b = bytearray()
        while len(b) < n:
            c = conn.recv(n - len(b))
            if not c:
                raise ConnectionError
            b.extend(c)
        return bytes(b)

    try:
        while True:
            op, n = struct.unpack("<II", recv(8))
            out = srv.handle(op, recv(n))
            conn.sendall(struct.pack("<I", len(out)) + out)
    except ConnectionError:
        pass
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=os.path.join(HERE, "run", "model.sock"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "v3_last.pt"))
    ap.add_argument("--star-goals", action="store_true", help="score by distance_star (v4+ checkpoints)")
    a = ap.parse_args()
    srv = Server(a.ckpt, a.star_goals)
    if os.path.exists(a.socket):
        os.remove(a.socket)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(a.socket)
    s.listen(64)
    print(f"serving {a.ckpt} (step {srv.rt.step}) on {a.socket}; input {srv.input_bytes} bytes", flush=True)
    while True:
        conn, _ = s.accept()
        threading.Thread(target=serve, args=(srv, conn), daemon=True).start()


if __name__ == "__main__":
    main()
