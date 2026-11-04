"""The state backbone and heads of sm64_AI_backbone.md.

Input is one state as raw extractor arrays (gi, gf, pad_prev, oi/of/om,
tri/tm); every normalisation (relative coordinates, symlog, Fourier
features, sin/cos of s16 angles, multi-hot bits, lookup embeddings) happens
here, so the ONNX export takes exactly what Rust reads.

Tokens: CLS, MARIO, MARIO_SURFACE, CAMERA, PROGRESS, CONTEXT, UI, PAD_PREV,
GEO_WATER x4, OBJ x64 (one MLP per category; the held object gets a HELD
type embedding), GEO x128 (one MLP per kind). No positional encoding: the
encoder is permutation-invariant over tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import (CATEGORIES, GEO_KINDS, N_ACTION_ID, bhv_category_table, camera_surface_table,
                     rng_index_table, special_table)

TAU = 2 * math.pi
SUBGOAL_H = (64, 256, -1)  # -1: "the end of the objective-rrt edge" (v6)
SUB_M = 4
N_EVENTS = 16
KS = (4, 16, 64)


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def clean(x):
    """Raw RAM floats can be garbage (a stale `m->wall` gives "normals" of
    1e35): non-finite -> 0, then clamp."""
    return torch.nan_to_num(x.float(), 0.0, 0.0, 0.0).clamp(-1e7, 1e7)


# Bit fields by AND with constant masks (exporters lack integer shifts, and
# lower integer div / mod through float32, which is inexact above 2^24).
def bitvec(x, lo, n):
    """Bits lo .. lo+n-1 of int64 x (two's complement) as 0/1 floats."""
    m = torch.tensor([1 << k for k in range(lo, lo + n)], dtype=torch.long, device=x.device)
    return (torch.bitwise_and(x.long().unsqueeze(-1), m) != 0).float()


def field(x, lo, n):
    """(x >> lo) & (2^n - 1), n <= 24, as int64."""
    w = torch.tensor([float(1 << k) for k in range(n)], device=x.device)
    return (bitvec(x, lo, n) * w).sum(-1).long()


def shr(x, n):
    return field(x, n, 32 - n)


def lowb(x, n):
    return field(x, 0, n)


def pad_fields(p):
    """Packed u32 pads (buttons << 16 | stick x << 8 | stick y) -> 14 button
    bits (A B Z Start, D-pad, L R, C), signed stick x, y."""
    bits_ = torch.cat([bitvec(p, 16 + b, 1) for b in [15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1, 0]], -1)
    sx = field(p, 8, 8)
    sy = field(p, 0, 8)
    return bits_, sx - 256 * (sx >= 128).long(), sy - 256 * (sy >= 128).long()


def unsymlog_t(y):
    return torch.sign(y) * torch.expm1(y.abs())


def u32(x):
    return torch.bitwise_and(x.long(), 0xFFFFFFFF)


def bits(x, n, lo=0):
    return bitvec(u32(x), lo, n)


def ang(a):
    t = a.float() * (TAU / 65536)
    return torch.stack([torch.sin(t), torch.cos(t)], -1)


def dir_ang(x, y, phi):
    """sin, cos of (atan2(x, y) + phi) without atan2 (ONNX lowers it with
    quadrant errors); angle 0 for a zero vector."""
    r = torch.sqrt(x * x + y * y)
    nz = r > 0
    s0 = torch.where(nz, x / r.clamp(min=1e-12), 0.0)
    c0 = torch.where(nz, y / r.clamp(min=1e-12), 1.0)
    sp, cp = torch.sin(phi), torch.cos(phi)
    return torch.stack([s0 * cp + c0 * sp, c0 * cp - s0 * sp], -1)


def ang_rad(t):
    return torch.stack([torch.sin(t), torch.cos(t)], -1)


class Fourier(nn.Module):
    def __init__(self, n=8, lo=-3):
        super().__init__()
        self.register_buffer("f", (2.0 ** torch.arange(lo, lo + n).float()) * math.pi, persistent=False)

    def forward(self, x):  # [..., c] -> [..., c * 2n]
        xf = x.unsqueeze(-1) * self.f
        return torch.cat([torch.sin(xf), torch.cos(xf)], -1).flatten(-2)


def mlp(i, d, o=None):
    o = o or d
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o), nn.LayerNorm(o))


class Cols:
    def __init__(self, m):
        self.m = m

    def __getitem__(self, k):
        return self.m[k]


class Tokenizer(nn.Module):
    def __init__(self, sch, d=256):
        super().__init__()
        self.sch, self.d = sch, d
        self.G, self.Fc, self.O, self.OF, self.T = Cols(sch.gi), Cols(sch.gf), Cols(sch.oi), Cols(sch.of), Cols(sch.tri)
        self.four = Fourier(8, -3)
        self.register_buffer("cat_of", torch.tensor(bhv_category_table(sch)), persistent=False)
        self.register_buffer("special", torch.tensor(special_table()), persistent=False)
        self.register_buffer("camsurf", torch.tensor(camera_surface_table()), persistent=False)
        idx, _ = rng_index_table()
        self.register_buffer("rng_idx", torch.tensor(idx), persistent=False)
        self.register_buffer("rng_f", (2.0 ** torch.arange(16).float()) * TAU, persistent=False)
        self.register_buffer("timer_p", 2.0 ** torch.arange(1, 12).float(), persistent=False)
        E = nn.Embedding
        self.e_action = E(N_ACTION_ID, 64)
        self.e_group = E(8, 16)
        self.e_level = E(64, 16)
        self.e_area = E(16, 8)
        self.e_act = E(8, 4)
        self.e_course = E(32, 8)
        self.e_play = E(8, 4)
        self.e_surf = E(257, 16)
        self.e_bhv = E(sch.n_bhv, 64)
        self.e_oact = E(64, 16)
        self.e_osub = E(32, 8)
        self.e_held = E(4, 4)
        self.e_bp1 = E(256, 8)
        self.e_bp2 = E(256, 8)
        self.e_cam = E(32, 8)
        self.e_cut = E(256, 8)
        self.e_dialog = E(256, 16)
        self.e_small = E(16, 4)  # dialog state / menu line / menu mode / act cursor (offset)
        self.types = nn.ParameterDict({k: nn.Parameter(torch.randn(d) * 0.02) for k in
                                       ["CLS", "MARIO", "SURFACE", "CAMERA", "PROGRESS", "CONTEXT", "UI", "PAD_PREV",
                                        "WATER", "HELD", "BUDGET", "GOAL_REGION", "GOAL_OBJECTIVE", "GOAL_STATE"]})
        self.obj_type = nn.Embedding(len(CATEGORIES), d)
        self.geo_type = nn.Embedding(len(GEO_KINDS), d)
        self.mask_obj = nn.Parameter(torch.randn(d) * 0.02)
        self.m_mario = mlp(self._n_mario(), d)
        self.m_surface = mlp(3 + 16 + 3 + 3 + 2 + 2, d)
        self.m_camera = mlp(2 + 2 + 8 + 8 + 16 + 16 + 2 + 16, d)
        self.m_progress = mlp(1 + 175 + 32 + 3 + 1, d)
        self.m_context = mlp(16 + 8 + 4 + 8 + 4 + 1 + 33 + 22 + 3, d)
        self.m_ui = mlp(16 + 4 + 2 + 4 + 4 + 1 + 4 + 4 + 1, d)
        self.m_pad = mlp(14 + 3 + 1 + 2 + 2 + 2, d)
        self.m_water = mlp(4 + 1 + 2, d)
        n_obj = self._n_obj()
        self.m_obj = nn.ModuleList([mlp(n_obj, d) for _ in CATEGORIES])
        n_geo = 3 + 48 + 9 + 3 + 1 + 16 + 3 + 1 + 3
        self.m_geo = nn.ModuleList([mlp(n_geo, d) for _ in GEO_KINDS])

    def _n_mario(self):
        return 3 + 48 + 3 + 1 + 2 + 10 + 2 + 3 + 64 + 16 + 23 + 64 + 3 + 32 + 9 + 2 + 1 + 3 + 16

    def _n_obj(self):
        return 3 + 48 + 1 + 3 + 1 + 1 + 2 + 2 + 2 + 2 + 3 + 1 + 16 + 8 + 1 + 10 + 16 + 32 + 32 + 4 + 1 + 4 + 3 + 8 + 8 + 4 + 2 + 64

    # -- helpers --
    def mario_frame(self, gi, gf):
        G, Fc = self.G, self.Fc
        pos = torch.stack([gf[:, Fc["pos_x"]], gf[:, Fc["pos_y"]], gf[:, Fc["pos_z"]]], -1)
        return pos, gi[:, G["yaw"]]

    def rel(self, v):
        return torch.cat([symlog(v / 1000), self.four(v / 1000)], -1)

    def forward(self, s, mask_obj=None):
        """s: dict of raw arrays (see data.Store.state). Returns tokens [B, T, d]
        and the padding mask [B, T] (True = padding)."""
        G, Fc = self.G, self.Fc
        gi, gf = s["gi"].long(), clean(s["gf"])
        B = gi.shape[0]
        pos, yaw = self.mario_frame(gi, gf)
        g = lambda k: gi[:, G[k]]
        f = lambda k: gf[:, Fc[k]]
        sl = lambda x, sc=1.0: symlog(x.float() / sc).unsqueeze(-1)
        dev = gi.device

        action = u32(g("action"))
        aid = lowb(action, 9)
        mario = torch.cat([
            symlog(pos / 8192), self.four(pos / 1000),
            symlog(torch.stack([f("vel_x"), f("vel_y"), f("vel_z")], -1) / 10), sl(f("fwd_vel"), 10),
            symlog(torch.stack([f("slide_vel_x"), f("slide_vel_z")], -1) / 10),
            torch.cat([ang(g(k)) for k in ["yaw", "pitch", "twirl_yaw", "slide_yaw", "intended_yaw"]], -1),
            ang(g("intended_yaw") - yaw),
            symlog(torch.stack([g("angvel_x"), g("angvel_y"), g("angvel_z")], -1).float() / 100),
            self.e_action(aid), self.e_group(lowb(shr(aid, 6), 3)), bits(action, 23, 9),
            self.e_action(lowb(u32(g("prev_action")), 9)),
            sl(g("action_state")), sl(g("action_arg")), sl(g("action_timer")),
            bits(g("mario_flags"), 32),
            torch.cat([sl(g(k)) for k in ["cap_timer", "invinc_timer", "squish_timer", "hurt_counter", "heal_counter",
                                          "wall_kick_timer", "double_jump_timer", "frames_since_a", "frames_since_b"]], -1),
            shr(g("health"), 8).float().unsqueeze(-1) / 8, g("health").float().unsqueeze(-1) / 0x880,
            g("holding").float().unsqueeze(-1),
            sl(f("peak_height") - pos[:, 1], 100), sl(f("quicksand_depth")), (f("intended_mag") / 32).unsqueeze(-1),
            self.e_level(g("level").clamp(0, 63)),
        ], -1)
        wall_n = torch.stack([f("wall_nx"), f("wall_ny"), f("wall_nz")], -1)
        has_wall = (g("has_wall") > 0) & (wall_n.abs().amax(-1) <= 1.01)
        wall_n = wall_n * has_wall.unsqueeze(-1)
        has_wall = has_wall.float()
        ftype = lowb(g("floor_type"), 8).where(g("floor_type") >= 0, torch.full_like(g("floor_type"), 256))
        surface = torch.cat([
            symlog(torch.stack([f("floor_y") - pos[:, 1], f("ceil_y") - pos[:, 1], g("water_level").float() - pos[:, 1]], -1) / 100),
            self.e_surf(ftype),
            torch.stack([f("floor_nx"), f("floor_ny"), f("floor_nz")], -1),
            wall_n, has_wall.unsqueeze(-1), g("floor_is_obj").float().unsqueeze(-1),
            sl(f("floor_obj_vy"), 10), sl((f("floor_obj_y") - pos[:, 1]) * g("floor_is_obj").float(), 100),
        ], -1)
        camera = torch.cat([
            ang(g("cam_yaw")), ang(g("cam_yaw") - yaw),
            self.e_cam(g("cam_mode").clamp(0, 31)), self.e_cut(g("cam_cutscene").clamp(0, 255)),
            bits(g("cam_selection"), 16), bits(g("cam_move_flags"), 16), ang(g("cam_c_side_yaw")),
            self.e_surf(ftype) * self.camsurf[ftype].float().unsqueeze(-1),
        ], -1)
        cs = torch.stack([gi[:, G[f"cs{k}"]] for k in range(25)], -1)
        progress = torch.cat([
            sl(g("stars")), bits(cs, 7).flatten(-2), bits(g("save_flags"), 32),
            (g("keys").float() / 2).unsqueeze(-1), sl(g("coins")), (g("red_coins").float() / 8).unsqueeze(-1), sl(g("lives")),
        ], -1)
        ridx = self.rng_idx[lowb(g("rng_seed"), 16)]
        rf = ridx.unsqueeze(-1) * self.rng_f
        gt = u32(g("global_timer")).float()
        # The global timer only as phases: the absolute frame would leak time-to-goal.
        tph = (gt.unsqueeze(-1) % self.timer_p) / self.timer_p * TAU
        context = torch.cat([
            self.e_level(g("level").clamp(0, 63)), self.e_area(g("area").clamp(0, 15)), self.e_act(g("act").clamp(0, 7)),
            self.e_course(g("course").clamp(0, 31)), self.e_play(g("play_mode").clamp(0, 7)), sl(g("transition")),
            torch.sin(rf), torch.cos(rf), (ridx >= 0).float().unsqueeze(-1),
            torch.sin(tph), torch.cos(tph),
            sl(g("level_changes")), sl(g("wip")), g("controllable").float().unsqueeze(-1),
        ], -1)
        es = lambda k: self.e_small((g(k) + 1).clamp(0, 15))
        ui = torch.cat([
            self.e_dialog((g("dialog") + 1).clamp(0, 255)), es("dialog_state"),
            g("dialog_last_page").float().unsqueeze(-1), g("dialog_choice").float().unsqueeze(-1),
            es("menu_line"), es("menu_mode"), g("act_select").float().unsqueeze(-1), es("act_init"), es("act_cursor"),
            sl(g("menu_timer")),
        ], -1)
        pp = s["pad_prev"].long()
        pbits, sx, sy = pad_fields(pp)
        pad = torch.cat([
            pbits, (sx.float() / 64).unsqueeze(-1), (sy.float() / 64).unsqueeze(-1),
            (torch.sqrt(sx.float() ** 2 + sy.float() ** 2) / 64).unsqueeze(-1), (f("intended_mag") / 32).unsqueeze(-1),
            # World stick direction: atan2s(-sy, sx) + camera yaw (mario.c).
            dir_ang(sx.float(), -sy.float(), g("cam_yaw").float() * TAU / 65536),
            ang(g("intended_yaw")), ang(g("intended_yaw") - yaw),
        ], -1)
        wtoks, wmask = [], []
        for k in range(4):
            wid = f(f"w{k}_id")
            lo = torch.stack([f(f"w{k}_lox") - pos[:, 0], f(f"w{k}_loz") - pos[:, 2]], -1)
            hi = torch.stack([f(f"w{k}_hix") - pos[:, 0], f(f"w{k}_hiz") - pos[:, 2]], -1)
            inside = ((lo < 0) & (hi > 0)).all(-1).float().unsqueeze(-1)
            wtoks.append(torch.cat([symlog(torch.cat([lo, hi], -1) / 1000), sl(f(f"w{k}_y") - pos[:, 1], 100),
                                    inside, (wid < 50).float().unsqueeze(-1)], -1))
            wmask.append(wid < 0)
        water = self.m_water(torch.stack(wtoks, 1)) + self.types["WATER"]

        T = self.types
        singles = torch.stack([
            T["CLS"].expand(B, -1),
            self.m_mario(mario) + T["MARIO"], self.m_surface(surface) + T["SURFACE"],
            self.m_camera(camera) + T["CAMERA"], self.m_progress(progress) + T["PROGRESS"],
            self.m_context(context) + T["CONTEXT"], self.m_ui(ui) + T["UI"], self.m_pad(pad) + T["PAD_PREV"],
        ], 1)
        obj, ocat = self.objects(s, pos, yaw)
        if mask_obj is not None:
            obj = torch.where(mask_obj.unsqueeze(-1), self.mask_obj.to(obj.dtype), obj)
        geo = self.geometry(s, pos)
        toks = torch.cat([singles, water, obj, geo], 1)
        pad_mask = torch.cat([torch.zeros(B, 8, dtype=torch.bool, device=dev), torch.stack(wmask, 1),
                              ~s["om"].bool(), ~s["tm"].bool()], 1)
        return toks, pad_mask

    def objects(self, s, pos, yaw):
        O, OF = self.O, self.OF
        oi, of = s["oi"].long(), clean(s["of"])
        c = lambda k: oi[..., O[k]]
        cf = lambda k: of[..., OF[k]]
        sl = lambda x, sc=1.0: symlog(x.float() / sc).unsqueeze(-1)
        p = torch.stack([cf("pos_x"), cf("pos_y"), cf("pos_z")], -1) - pos[:, None]
        h = torch.stack([cf("home_x"), cf("home_y"), cf("home_z")], -1) - pos[:, None]
        dist = p.norm(dim=-1, keepdim=True)
        yawr = yaw[:, None].float() * TAU / 65536
        timer = c("timer").float()
        tp = torch.tensor([16.0, 32, 64, 128, 256], device=oi.device)
        tph = (timer.unsqueeze(-1) % tp) / tp * TAU
        bhv = c("bhv").clamp(0, self.sch.n_bhv - 1)
        bp = u32(c("beh_params"))
        x = torch.cat([
            symlog(p / 1000), self.four(p / 1000), symlog(dist / 1000),
            symlog(torch.stack([cf("vel_x"), cf("vel_y"), cf("vel_z")], -1) / 10), sl(cf("fwd_vel"), 10), sl(c("angvel_yaw"), 100),
            ang(c("face_yaw")), ang(c("face_yaw") - yaw[:, None]), dir_ang(p[..., 0], p[..., 2], -yawr.expand_as(p[..., 0])), ang(c("move_yaw")),
            symlog(h / 1000), symlog(h.norm(dim=-1, keepdim=True) / 1000),
            self.e_oact(c("action").clamp(0, 63)), self.e_osub(c("sub_action").clamp(0, 31)),
            symlog(timer / 30).unsqueeze(-1), torch.sin(tph), torch.cos(tph),
            bits(c("active_flags"), 16), bits(c("interact_type"), 32), bits(c("interact_status"), 32),
            self.e_held(c("held_state").clamp(0, 3)), sl(c("intangible_timer")),
            symlog(torch.stack([cf("hitbox_r"), cf("hitbox_h"), cf("hurtbox_r"), cf("hurtbox_h")], -1) / 100),
            sl(c("health")), sl(c("value")), sl(c("loot_coins")),
            self.e_bp1(lowb(shr(bp, 24), 8)), self.e_bp2(lowb(shr(bp, 16), 8)),
            symlog(torch.stack([c(f"raw{k}") for k in range(4)], -1).float()),
            c("is_floor").float().unsqueeze(-1), c("is_held").float().unsqueeze(-1),
            self.e_bhv(bhv),
        ], -1)
        cat = self.cat_of[bhv]
        out = 0
        for k, m in enumerate(self.m_obj):
            out = out + m(x) * (cat == k).unsqueeze(-1).to(x.dtype)
        out = out + self.obj_type(cat) + self.types["HELD"] * c("is_held").float().unsqueeze(-1)
        return out, cat

    def geometry(self, s, pos):
        T = self.T
        t = clean(s["tri"])
        v = t[..., 0:9].reshape(*t.shape[:-1], 3, 3)
        cen = v.mean(-2)
        n = t[..., T["nx"] : T["nz"] + 1]
        kind_s = t[..., T["kind"]].long()
        kidx = lowb(kind_s, 8).clamp(0, 256)
        area = 0.5 * torch.linalg.cross(v[..., 1, :] - v[..., 0, :], v[..., 2, :] - v[..., 0, :]).norm(dim=-1, keepdim=True)
        lst = t[..., T["list"]].long().clamp(0, 2)
        owner = t[..., T["owner"]]
        x = torch.cat([
            self.rel(cen - pos[:, None]), symlog((v - cen.unsqueeze(-2)) / 100).flatten(-2), n, symlog(area / 1000),
            self.e_surf(kidx), F.one_hot(lst, 3).float(), (owner >= 0).float().unsqueeze(-1),
            symlog(t[..., T["ovx"] : T["ovz"] + 1] / 10),
        ], -1)
        ny = n[..., 1]
        kind = torch.where(self.special[kidx] > 0, 3, torch.where(ny > 0.01, 0, torch.where(ny < -0.01, 2, 1)))
        out = 0
        for k, m in enumerate(self.m_geo):
            out = out + m(x) * (kind == k).unsqueeze(-1).to(x.dtype)
        return out + self.geo_type(kind)


class Encoder(nn.Module):
    def __init__(self, sch, d=256, layers=6, heads=8):
        super().__init__()
        self.tok = Tokenizer(sch, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.z = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, s, mask_obj=None):
        toks, pm = self.tok(s, mask_obj)
        h = self.norm(self.tf(toks, src_key_padding_mask=pm))
        z = self.z(torch.cat([h[:, 0], h[:, 1]], -1))
        return z, h, pm


OBJ_START = 12  # 8 single tokens + 4 water
N_STYLES = 5  # data.STYLES
N_VERBS = 6  # data.VERBS


def small_tf(d, layers=2, heads=8):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def iqe(x, y):
    """Interval quasimetric embedding (Wang & Isola 2022; torchqmet's IQE):
    per component, the total length of the union of [x_i, max(x_i, y_i)].
    x, y: [..., k, l] -> [..., k]."""
    l = x.shape[-1]
    valid = x < y
    xy = torch.cat(torch.broadcast_tensors(x, y), -1)
    ixy = xy.argsort(-1)
    sxy = xy.gather(-1, ixy)
    neg_inc = torch.gather(valid, -1, ixy % l) * torch.where(ixy < l, -1, 1)
    neg_f = (torch.cumsum(neg_inc, -1) < 0).float() * -1.0
    neg_incf = torch.cat([neg_f[..., :1], torch.diff(neg_f, dim=-1)], -1)
    return (sxy * neg_incf).sum(-1)


class Backbone(nn.Module):
    def __init__(self, sch, n_kinds, d=256, layers=6, heads=8, iqe_k=16, iqe_l=32):
        super().__init__()
        self.d, self.n_kinds = d, n_kinds
        self.enc = Encoder(sch, d, layers, heads)
        four = self.enc.tok.four
        # Pads as tokens (dynamics input), with time-step embeddings.
        self.pad_in = mlp(14 + 2, d)
        self.t_emb = nn.Embedding(64, d)
        self.k_emb = nn.Embedding(len(KS), d)
        self.dyn_tf = small_tf(d)
        self.dyn_mario = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 12 + N_ACTION_ID))
        self.dyn_obj = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 4))
        self.inv_tf = small_tf(d)
        self.inv_out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 32 + 32 + 4 + 14))
        self.events = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, len(KS) * N_EVENTS))
        self.mask_out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, sch.n_bhv + 6 + 64))
        self.iqe_k, self.iqe_l = iqe_k, iqe_l
        self.phi = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, iqe_k * iqe_l))
        # "Collect star X" goals (data.N_GOALS ids): a learned point per goal
        # on the goal side of the IQE, so d(state, star X) is one lookup.
        from .data import N_GOALS
        self.star_phi = nn.Embedding(N_GOALS, iqe_k * iqe_l)
        # Subgoal head: where Mario should be ~H frames later on a good path
        # to goal X (H in SUBGOAL_H): a mixture of SUB_M Gaussians over the
        # position change (symlog(dp / 1000), x y z) and whether it is in
        # another zone. Its proposals become objective-rrt PosDist targets.
        self.sub_goal = nn.Embedding(N_GOALS, 64)
        self.sub_h = nn.Embedding(len(SUBGOAL_H), 16)
        self.sub_out = nn.Sequential(nn.Linear(d + 64 + 16, d), nn.GELU(), nn.Linear(d, d), nn.GELU(),
                                     nn.Linear(d, SUB_M * (1 + 3 + 3) + 1))
        # Edge mode extras (v6): the edge's budget (log1p frames) and which
        # objective kind made it.
        # Learned objective (v6 edge mode): budget, objective-rrt kind, style,
        # event bits, end forward speed, verb; and a pointer over the state's
        # object tokens choosing the object the verb is done to.
        self.n_kinds = n_kinds
        self.sub_edge = nn.Sequential(nn.Linear(d + 64, d), nn.GELU(), nn.Linear(d, 1 + n_kinds + N_STYLES + 16 + 1 + N_VERBS))
        self.verb_emb = nn.Embedding(N_VERBS, 32)
        self.sub_ptr = nn.Sequential(nn.Linear(2 * d + 64 + 32, d), nn.GELU(), nn.Linear(d, 1))
        self.iqe_alpha = nn.Parameter(torch.zeros(()))
        self.iqe_scale = nn.Parameter(torch.tensor(math.log(30.0)))
        # Goal tokens and the goal-conditioned tail (reachability, policy).
        self.g_region = mlp(3 + 48 + 1 + 1 + 8 + 2 + 1, d)
        self.e_ggroup = nn.Embedding(8, 8)
        self.e_kind = nn.Embedding(n_kinds, 32)
        self.g_objective = mlp(32 + 51, d)
        self.g_state = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.budget_in = mlp(1 + 16, d)
        self.q = nn.Parameter(torch.randn(2, d) * 0.02)
        self.tail = small_tf(d)
        self.reach_out = nn.Linear(d, 1)
        self.pol_out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_kinds + 12))
        self.four = four

    # -- pieces --
    def pad_tokens(self, pads):
        pb, sx, sy = pad_fields(pads)
        sx, sy = sx.float() / 64, sy.float() / 64
        return self.pad_in(torch.cat([pb, sx.unsqueeze(-1), sy.unsqueeze(-1)], -1))

    def k_index(self, k):
        return (k >= 16).long() + (k >= 64).long()

    def dynamics(self, z, h, k, pads, pmask):
        B = z.shape[0]
        pt = self.pad_tokens(pads) + self.t_emb.weight[None]
        x = torch.cat([z.unsqueeze(1), self.k_emb(self.k_index(k)).unsqueeze(1), pt], 1)
        m = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=z.device), ~pmask], 1)
        ctx = self.dyn_tf(x, src_key_padding_mask=m)[:, 0]
        mario = self.dyn_mario(ctx)
        ho = h[:, OBJ_START : OBJ_START + 64]
        obj = self.dyn_obj(torch.cat([ho, ctx.unsqueeze(1).expand(-1, ho.shape[1], -1)], -1))
        return mario, obj

    def inverse(self, za, zb, k, pmask):
        B = za.shape[0]
        q = self.t_emb.weight[None].expand(B, -1, -1)
        x = torch.cat([za.unsqueeze(1), zb.unsqueeze(1), self.k_emb(self.k_index(k)).unsqueeze(1), q], 1)
        m = torch.cat([torch.zeros(B, 3, dtype=torch.bool, device=za.device), ~pmask], 1)
        return self.inv_out(self.inv_tf(x, src_key_padding_mask=m)[:, 3:])

    def distance(self, za, zg):
        """Quasimetric frames estimate d(a -> g) >= 0."""
        with torch.autocast("cuda", enabled=False):
            pa = self.phi(za.float()).view(*za.shape[:-1], self.iqe_k, self.iqe_l)
            pg = self.phi(zg.float()).view(*zg.shape[:-1], self.iqe_k, self.iqe_l)
            comp = iqe(pa, pg)
            a = torch.sigmoid(self.iqe_alpha)
            d = a * comp.max(-1).values + (1 - a) * comp.mean(-1)
            return d * torch.exp(self.iqe_scale)

    def distance_star(self, za, goal):
        """Estimated frames from states za to collecting goal ids `goal`."""
        with torch.autocast("cuda", enabled=False):
            pa = self.phi(za.float()).view(*za.shape[:-1], self.iqe_k, self.iqe_l)
            pg = self.star_phi(goal).view(*goal.shape, self.iqe_k, self.iqe_l)
            comp = iqe(pa, pg)
            a = torch.sigmoid(self.iqe_alpha)
            d = a * comp.max(-1).values + (1 - a) * comp.mean(-1)
            return d * torch.exp(self.iqe_scale)

    def subgoal(self, za, goal, h_idx):
        """Mixture over where Mario is ~SUBGOAL_H[h_idx] frames later on a
        good path to `goal`: (logits [B, M], mean [B, M, 3], log std [B, M, 3]
        in symlog(dp / 1000)), and a zone-change logit [B]."""
        o = self.sub_out(torch.cat([za, self.sub_goal(goal), self.sub_h(h_idx)], -1)).float()
        M = SUB_M
        logit = o[:, :M]
        mu = o[:, M : M + 3 * M].view(-1, M, 3)
        ls = o[:, M + 3 * M : M + 6 * M].view(-1, M, 3).clamp(-5, 2)
        return logit, mu, ls, o[:, -1]

    def subgoal_edge(self, za, goal):
        """Edge mode extras [B]: dict of budget (log1p frames), kind, style,
        events (16 bit logits), speed (forward velocity / 32), verb logits."""
        o = self.sub_edge(torch.cat([za, self.sub_goal(goal)], -1)).float()
        n = [1, self.n_kinds, N_STYLES, 16, 1, N_VERBS]
        parts = torch.split(o, n, -1)
        return dict(budget=parts[0][:, 0], kind=parts[1], style=parts[2], events=parts[3],
                    speed=parts[4][:, 0], verb=parts[5])

    def object_pointer(self, za, h, pm, goal, verb):
        """Logits over the state's object tokens [B, 64] (padding -inf) for the
        object a learned objective's verb is done to."""
        ho = h[:, OBJ_START : OBJ_START + 64]
        q = torch.cat([za, self.sub_goal(goal), self.verb_emb(verb)], -1).to(ho.dtype)
        x = torch.cat([ho, q.unsqueeze(1).expand(-1, ho.shape[1], -1)], -1)
        lg = self.sub_ptr(x)[..., 0].float()
        return lg.masked_fill(pm[:, OBJ_START : OBJ_START + 64], float("-inf"))

    def goal_tokens(self, sa, sg, zg, g_kind, keep):
        tk = self.enc.tok
        gi_a, gf_a = sa["gi"].long(), clean(sa["gf"])
        gi_g, gf_g = sg["gi"].long(), clean(sg["gf"])
        pa, _ = tk.mario_frame(gi_a, gf_a)
        pg, _ = tk.mario_frame(gi_g, gf_g)
        G, Fc = tk.G, tk.Fc
        rel = pg - pa
        hs = torch.sqrt(gf_g[:, Fc["vel_x"]] ** 2 + gf_g[:, Fc["vel_z"]] ** 2)
        act = u32(gi_g[:, G["action"]])
        region = torch.cat([
            symlog(rel / 1000), self.four(rel / 1000), torch.full_like(hs, math.log(100.0)).unsqueeze(-1),
            symlog(hs / 10).unsqueeze(-1), self.e_ggroup(lowb(shr(lowb(act, 9), 6), 3)),
            bits(act, 2, 11),  # ACT_FLAG_AIR (bit 11), ACT_FLAG_INTANGIBLE (12): action class
            symlog((gi_g[:, G["stars"]] - gi_a[:, G["stars"]]).float()).unsqueeze(-1),
        ], -1)
        T = tk.types
        toks = torch.stack([
            self.g_region(region) + T["GOAL_REGION"],
            self.g_objective(torch.cat([self.e_kind(g_kind), symlog(rel / 1000), self.four(rel / 1000)], -1)) + T["GOAL_OBJECTIVE"],
            self.g_state(zg) + T["GOAL_STATE"],
        ], 1)
        return toks, ~keep

    def goal_tail(self, h, pm, gtoks, gmask, budget):
        B = h.shape[0]
        lb = torch.log1p(budget.float().clamp(min=0)).unsqueeze(-1)
        bt = self.budget_in(torch.cat([lb / 10, self.four(lb / 4)[..., :16]], -1)) + self.enc.tok.types["BUDGET"]
        x = torch.cat([h, gtoks, bt.unsqueeze(1), self.q[None].expand(B, -1, -1)], 1)
        m = torch.cat([pm, gmask, torch.zeros(B, 3, dtype=torch.bool, device=h.device)], 1)
        o = self.tail(x, src_key_padding_mask=m)
        reach = self.reach_out(o[:, -2]).squeeze(-1)
        pol = self.pol_out(o[:, -1])
        return reach, pol
