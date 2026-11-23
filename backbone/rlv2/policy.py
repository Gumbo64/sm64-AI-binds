"""Observation features, the factored high-frequency action space and the
policy network for route RL."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

TAU = 2 * math.pi
# Stick direction: neutral, or an angle from a base (Mario's facing, or
# straight at the objective's steer target). Bins are fine near 0 and 180,
# because frame-exact tricks (BLJ, turn-arounds, line-ups) live there.
ANGLES = [0, 3, -3, 6, -6, 12, -12, 22.5, -22.5, 45, -45, 90, -90, 135, 180]
N_ANG = len(ANGLES)
N_DIR = 1 + 2 * N_ANG
MAGS = [1.0, 0.6, 0.38]
A, B, Z = 0x8000, 0x4000, 0x2000
BTNS = [0, A, B, Z, A | B, A | Z, B | Z, A | B | Z]
HOLDS = [1, 2, 4, 8]
_REL = np.round(np.array(ANGLES) * 65536 / 360).astype(np.int32)
_RELRAD = np.array(ANGLES) * math.pi / 180


def decode(dirs, mags, btns, holds):
    """Action indices -> rl_act's base, rel, mag, btn, hold."""
    d = np.asarray(dirs)
    ang = (d - 1) % N_ANG
    base = np.where(d - 1 >= N_ANG, 1, 0).astype(np.int32)
    rel = _REL[ang]
    mag = np.where(d == 0, 0.0, np.array(MAGS)[np.asarray(mags)]).astype(np.float32)
    return base, rel, mag, np.array(BTNS, np.uint32)[np.asarray(btns)], np.array(HOLDS, np.int32)[np.asarray(holds)]


def encode_pad(pad, f, target):
    """A played pad (packed) in state f (rl_obs row) -> the nearest action
    (dir, mag, btn); the hold is set by the caller from run lengths."""
    b = (pad >> 16) & 0xFFFF
    x = ((pad >> 8) & 0xFF) - (((pad >> 8) & 0x80) << 1)
    y = (pad & 0xFF) - ((pad & 0x80) << 1)
    ax = 0 if abs(x) < 8 else x - (6 if x > 0 else -6)
    ay = 0 if abs(y) < 8 else y - (6 if y > 0 else -6)
    m = math.hypot(ax, ay)
    btn = BTNS.index(b & (A | B | Z))
    if m == 0:
        return 0, 0, btn
    mag = int(np.argmin(np.abs(np.array(MAGS) - min(m / 64, 1.0))))
    cam, yaw = f[9] * TAU / 65536, f[7] * TAU / 65536
    world = math.atan2(ax, -ay) + cam
    wrap = lambda a: (a + math.pi) % TAU - math.pi
    errs = np.abs(wrap(world - yaw - _RELRAD))
    best, base = int(np.argmin(errs)), 0
    if np.isfinite(target).all():
        by = math.atan2(target[0] - f[0], target[2] - f[2])
        e2 = np.abs(wrap(world - by - _RELRAD))
        if e2.min() < errs.min():
            best, base = int(np.argmin(e2)), 1
    return 1 + base * N_ANG + best, mag, btn


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _rot(dx, dz, yaw):
    """World (x, z) -> Mario's frame (forward, right)."""
    s, c = np.sin(yaw), np.cos(yaw)
    return dx * s + dz * c, dx * c - dz * s


def _goal(dp, has, yaw):
    """A goal offset [n, 3] in Mario's frame: symlog forward/right/up,
    distance, bearing relative to his facing; zeros without one."""
    dp = np.nan_to_num(dp)
    fw, ri = _rot(dp[:, 0], dp[:, 2], yaw)
    d = np.hypot(dp[:, 0], dp[:, 2])
    br = np.arctan2(ri, fw)
    out = np.stack([symlog(fw / 200), symlog(ri / 200), symlog(dp[:, 1] / 200), symlog(d / 200), np.sin(br), np.cos(br), has.astype(np.float32)], 1)
    return out * has[:, None]


FREQS = [8192 / 2**k for k in range(0, 7)]
N_FEAT = 18 + 12 + 7 * 3 + 7 * 2 + 1 + 3 * 2 * len(FREQS)


def featurize(o, landmarks, eta_scale=100.0):
    """rl_obs -> (float32 [n, N_FEAT], action id, action group bits, floor
    type, leg). landmarks [n_legs + 2, 3] holds each leg's end position (NaN:
    none); the current and next legs' go in, so the policy can set up for
    the next one."""
    f = o["f"].astype(np.float64)
    n = len(f)
    yaw = f[:, 7] * TAU / 65536
    cam = f[:, 9] * TAU / 65536
    vf, vr = _rot(f[:, 3], f[:, 5], yaw)
    wf, wr = _rot(f[:, 17], f[:, 19], yaw)
    nf, nr = _rot(f[:, 14], f[:, 16], yaw)
    fx = f[:, 27] / 64
    fy = f[:, 28] / 64
    base = np.stack([
        vf / 64, vr / 64, f[:, 4] / 64, f[:, 6] / 64, symlog(f[:, 6] / 32), np.abs(f[:, 6]) > 48,
        np.sin(yaw), np.cos(yaw), np.sin(yaw - cam), np.cos(yaw - cam), f[:, 10] / 4096, f[:, 8] / 16384,
        symlog(f[:, 11] / 32), symlog(f[:, 12] / 32), symlog(np.maximum(f[:, 13], -500) / 32),
        nf, nr, f[:, 15],
    ], 1)
    more = np.stack([
        wf, wr, f[:, 18], f[:, 20], f[:, 21] / 32, f[:, 22] / 8, f[:, 23], f[:, 24], f[:, 25], f[:, 26], fx, fy,
    ], 1)
    pos = f[:, :3]
    has_t = np.isfinite(o["target"]).all(1)
    g_t = _goal(o["target"] - pos, has_t, yaw)
    leg = o["leg"]
    lm = landmarks[leg]
    g_e = _goal(lm - pos, np.isfinite(lm).all(1), yaw)
    lm2 = landmarks[leg + 1]
    g_n = _goal(lm2 - pos, np.isfinite(lm2).all(1), yaw)
    # The next leg relative to this leg's end: how to arrive.
    dl = np.nan_to_num(lm2 - lm)
    hl = np.isfinite(lm2).all(1) & np.isfinite(lm).all(1)
    g_l = _goal(dl, hl, yaw)[:, :7]
    # The speed along the way to the target: what the objective's ETA pays.
    gt2 = np.stack([symlog(o["eta"] / eta_scale), o["eta"] / 300, f[:, 29] / 64, symlog(f[:, 29] / 16),
                    (f[:, 30]), f[:, 31], vf * 0], 1).astype(np.float64)
    four = [g for w in FREQS for g in (np.sin(TAU * pos / w), np.cos(TAU * pos / w))]
    x = np.concatenate([base, more, g_t, g_e, g_n, g_l, gt2, np.ones((n, 1)), *four], 1).astype(np.float32)
    act = o["action"]
    return x, (act & 0x1FF).astype(np.int64), ((act >> 9) & 0x7FFFFF).astype(np.int64), np.clip(o["floor_type"] + 1, 0, 255).astype(np.int64), leg.astype(np.int64)


class Policy(nn.Module):
    def __init__(self, n_legs, d_z=0, h=512, layers=3):
        super().__init__()
        self.act = nn.Embedding(512, 32)
        self.floor = nn.Embedding(256, 8)
        self.leg = nn.Embedding(n_legs + 2, 16)
        self.d_z = d_z
        if d_z:
            self.z = nn.Sequential(nn.Linear(d_z, 128), nn.LayerNorm(128), nn.GELU())
        inp = N_FEAT + 32 + 23 + 8 + 16 + (128 if d_z else 0)

        def body():
            m, w = [], inp
            for _ in range(layers):
                m += [nn.Linear(w, h), nn.GELU()]
                w = h
            return nn.Sequential(*m)

        self.pi_body, self.v_body = body(), body()
        self.dir, self.mag, self.btn, self.hold = nn.Linear(h, N_DIR), nn.Linear(h, len(MAGS)), nn.Linear(h, len(BTNS)), nn.Linear(h, len(HOLDS))
        self.v = nn.Linear(h, 1)
        for head in (self.dir, self.mag, self.btn, self.hold):
            nn.init.orthogonal_(head.weight, 0.01)
            nn.init.zeros_(head.bias)

    def inp(self, x, act, bits, floor, leg, z=None):
        fl = ((bits[:, None] >> torch.arange(23, device=bits.device)) & 1).float()
        p = [x, self.act(act), fl, self.floor(floor), self.leg(leg)]
        if self.d_z:
            p.append(self.z(z))
        return torch.cat(p, 1)

    def forward(self, x, act, bits, floor, leg, z=None):
        i = self.inp(x, act, bits, floor, leg, z)
        p = self.pi_body(i)
        return (self.dir(p), self.mag(p), self.btn(p), self.hold(p)), self.v(self.v_body(i)).squeeze(-1)


def dists(logits):
    return [Categorical(logits=l) for l in logits]


def log_prob(ds, a):
    return sum(d.log_prob(a[:, i]) for i, d in enumerate(ds))


def entropy(ds):
    return sum(d.entropy() for d in ds)
