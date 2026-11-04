"""The trained backbone at run time, on the GPU: batched encoding of states
read from sm64env, distances, reachability, inverse dynamics (inputs that
lead from one state to another) and the policy's stick/button summary.

    rt = Runtime("checkpoints/v3_last.pt")
    z, h, pm = rt.encode(pool.read(ids))
    frames = rt.distance(z, z_goal)              # quasimetric estimate a -> g
    pads = rt.inverse(z_now, z_target, k=16)      # uint32 [B, 16]
"""

import math

import numpy as np
import torch

from .model import Backbone
from .schema import BUTTON_BITS, Schema

STATE_KEYS = ["gi", "gf", "pad_prev", "oi", "of", "om", "tri", "tm"]


class Runtime:
    def __init__(self, ckpt, device="cuda", chunk=256):
        ck = torch.load(ckpt, map_location="cpu")
        cfg = ck["cfg"]
        self.sch = Schema()
        self.kinds = ck["kinds"]
        self.m = Backbone(self.sch, len(self.kinds), cfg["d"], cfg["layers"], cfg["heads"])
        # Checkpoints before v4 have no star goals (star_phi): left untrained.
        missing, unexpected = self.m.load_state_dict(ck["model"], strict=False)
        bad = [k for k in missing if not k.startswith(("star_phi", "sub_", "verb_emb"))] + list(unexpected)
        if bad:
            raise RuntimeError(f"{ckpt}: state dict mismatch: {bad[:5]}")
        self.dev = torch.device(device)
        self.m.to(self.dev).eval()
        self.chunk = chunk
        self.step = ck["step"]

    def _t(self, s):
        return {k: torch.from_numpy(np.ascontiguousarray(s[k])).to(self.dev, non_blocking=True) for k in STATE_KEYS}

    @torch.no_grad()
    def encode(self, s, need_h=False):
        """States (dict of numpy arrays, batch first) -> z [B, d] (float32,
        on the device); with need_h also the token outputs and padding mask."""
        B = len(s["gi"])
        zs, hs, pms = [], [], []
        for a in range(0, B, self.chunk):
            part = {k: s[k][a : a + self.chunk] for k in STATE_KEYS}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                z, h, pm = self.m.enc(self._t(part))
            zs.append(z.float())
            if need_h:
                hs.append(h)
                pms.append(pm)
        z = torch.cat(zs) if zs else torch.zeros(0, self.m.d, device=self.dev)
        return (z, torch.cat(hs), torch.cat(pms)) if need_h else z

    @torch.no_grad()
    def distance(self, za, zg):
        """Estimated frames a -> g (broadcasts over the batch)."""
        return self.m.distance(za, zg)

    @torch.no_grad()
    def distance_matrix(self, za, zg):
        """Frames from each of za [A, d] to each of zg [G, d] -> [A, G]."""
        A, G = len(za), len(zg)
        out = torch.empty(A, G, device=self.dev)
        step = max(1, 65536 // max(G, 1))
        for i in range(0, A, step):
            a = za[i : i + step]
            out[i : i + step] = self.m.distance(a[:, None].expand(-1, G, -1).reshape(-1, za.shape[1]),
                                                zg[None].expand(len(a), -1, -1).reshape(-1, zg.shape[1])).view(len(a), G)
        return out

    @torch.no_grad()
    def inverse(self, za, zb, k, temperature=0.0, rng=None):
        """Inputs leading from a to b in k frames (k <= 64): packed uint32
        [B, k]. temperature 0 takes the most likely stick bin and buttons;
        above 0 samples them (stick bins from the softmax at that
        temperature, buttons as Bernoulli)."""
        B = len(za)
        kt = torch.full((B,), k, device=self.dev, dtype=torch.long)
        pmask = torch.zeros(B, 64, dtype=torch.bool, device=self.dev)
        pmask[:, :k] = True
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            io = self.m.inverse(za, zb, kt, pmask).float()[:, :k]
        lx, ly = io[..., :32], io[..., 32:64]
        mu = io[..., 64:66]
        if temperature > 0:
            g = torch.Generator(device=self.dev)
            g.manual_seed(int(rng.integers(1 << 62)) if rng is not None else 0)
            bx = torch.multinomial(torch.softmax(lx.reshape(-1, 32) / temperature, -1), 1, generator=g).view(B, k)
            by = torch.multinomial(torch.softmax(ly.reshape(-1, 32) / temperature, -1), 1, generator=g).view(B, k)
            btn = (torch.rand(io[..., 68:82].shape, device=self.dev, generator=g) < torch.sigmoid(io[..., 68:82])).long()
        else:
            bx, by = lx.argmax(-1), ly.argmax(-1)
            btn = (io[..., 68:82] > 0).long()
        sx = (bx * 8 + 3.5 + mu[..., 0].clamp(-1, 1) * 4 - 128).round().clamp(-128, 127).long()
        sy = (by * 8 + 3.5 + mu[..., 1].clamp(-1, 1) * 4 - 128).round().clamp(-128, 127).long()
        # Small sticks are inside the game's deadzone anyway; snap them to 0.
        dead = (sx.abs() < 8) & (sy.abs() < 8)
        sx, sy = sx.masked_fill(dead, 0), sy.masked_fill(dead, 0)
        bits = torch.tensor(BUTTON_BITS, device=self.dev)
        button = (btn << bits).sum(-1)
        packed = (button << 16) | ((sx & 0xFF) << 8) | (sy & 0xFF)
        return packed.to(torch.int64).cpu().numpy().astype(np.uint32)

    @torch.no_grad()
    def reach(self, sa, za, h, pm, sg, zg, budget):
        """P(goal g reachable from a within `budget` frames) [B]; GOAL_REGION
        and GOAL_STATE tokens as trained."""
        B = len(za)
        keep = torch.ones(B, 3, dtype=torch.bool, device=self.dev)
        keep[:, 1] = False
        ta = {k: torch.from_numpy(np.ascontiguousarray(sa[k])).to(self.dev) for k in ("gi", "gf")}
        tg = {k: torch.from_numpy(np.ascontiguousarray(sg[k])).to(self.dev) for k in ("gi", "gf")}
        kind = torch.zeros(B, dtype=torch.long, device=self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            gt, gm = self.m.goal_tokens(ta, tg, zg, kind, keep)
            rl, pol = self.m.goal_tail(h, pm, gt, gm, torch.as_tensor(budget, device=self.dev, dtype=torch.float32))
        return torch.sigmoid(rl.float()), pol.float()


def unpack(p):
    p = np.asarray(p, np.int64)
    return (p >> 16) & 0xFFFF, ((p >> 8) & 0xFF).astype(np.int8), (p & 0xFF).astype(np.int8)


def deg(v):
    return v * 180.0 / math.pi
