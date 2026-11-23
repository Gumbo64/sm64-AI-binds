#!/usr/bin/env python3
"""Trains the SM64 state backbone and heads (sm64_AI_backbone.md) on the
indefinite tree run's states and the 120-star TAS, with MLflow tracking.

    .venv/bin/python train.py [--steps N] [--phase1 FRAC] [--batch B] ...

Phase 1 (sm64_AI_backbone.md 6.1): forward dynamics, inverse dynamics,
events, masked-object prediction. Phase 2 (6.2) adds time distance (IQE
quasimetric), reachability and policy on hindsight pairs. Phase 3 (expert
iteration) needs the search to use the model and is not part of this run.

Offline metrics (7): k-step dynamics error, inverse-dynamics accuracy,
distance rank correlation on held-out TAS levels, reachability AUC.
"""

import argparse
import json
import math
import os
import time

import mlflow

from sm64bb import mlog
import numpy as np
import torch
import torch.nn.functional as F

from sm64bb.data import VERBS, Batcher, Store
from sm64bb.model import KS, N_EVENTS, Backbone, symlog, u32
from sm64bb.schema import BUTTON_BITS, Schema

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "objective-rrt", "runs", "indef-mgel-lg-pf-s2")
SOURCES = [os.path.join(HERE, "data", "tree14"), os.path.join(HERE, "data", "tas120")]
RUNS_DIR = os.path.join(HERE, "..", "objective-rrt", "runs")
# v4: every extracted tree (data/<run> with its nodes.csv) but the held-out one,
# the indefinite tree, and the TAS.
HELD_OUT_TREE = "pf-mgel-lg-pf-s1"


def v4_sources():
    import glob as _g
    out = [(os.path.join(HERE, "data", "tree14"), os.path.join(RUN, "nodes.csv"))]
    for d in sorted(_g.glob(os.path.join(HERE, "data", "*"))):
        run = os.path.basename(d)
        if run in ("tree14", "tas120", HELD_OUT_TREE) or not os.path.isdir(os.path.join(d, "s0")):
            continue
        csv = os.path.join(RUNS_DIR, run, "nodes.csv")
        if os.path.exists(csv):
            out.append((d, csv))
    out.append((os.path.join(HERE, "data", "tas120"), None))
    return out


def v7_sources():
    """v2-era data biased to fast play (rlv2/extract_v7.sh): TAS, best routes
    (no nodes, like the TAS), a little of two v2 trees."""
    import glob as _g
    v2 = os.path.join(HERE, "..", "objective-rrt-v2", "runs")
    out = []
    for d in sorted(_g.glob(os.path.join(HERE, "data", "v7", "*"))):
        run = os.path.basename(d)
        if not os.path.isdir(os.path.join(d, "s0")):
            continue
        out.append((d, None if run.startswith("r-") else os.path.join(v2, run, "nodes.csv")))
    out.append((os.path.join(HERE, "data", "tas120"), None))
    return out


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, store, batch, seed, tas_frac):
        self.store, self.batch, self.seed, self.tas_frac = store, batch, seed, tas_frac

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        b = Batcher(self.store, self.batch, self.tas_frac, seed=self.seed * 1000 + (wi.id if wi else 0))
        while True:
            yield b.batch()


def to_dev(x, dev):
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    return torch.from_numpy(np.asarray(x)).to(dev, non_blocking=True)


def keep_numpy(x):
    # Workers hand back numpy arrays, pickled through pipes: torch tensors
    # would go through /dev/shm, which is 64 MB in this container.
    return x


def cat_states(*ss):
    return {k: torch.cat([s[k] for s in ss], 0) for k in ss[0]}


def mario_targets(sch, sa, sb):
    G, Fc = sch.gi, sch.gf
    pa = torch.stack([sa["gf"][:, Fc[k]] for k in ("pos_x", "pos_y", "pos_z")], -1)
    pb = torch.stack([sb["gf"][:, Fc[k]] for k in ("pos_x", "pos_y", "pos_z")], -1)
    va = torch.stack([sa["gf"][:, Fc[k]] for k in ("vel_x", "vel_y", "vel_z")], -1)
    vb = torch.stack([sb["gf"][:, Fc[k]] for k in ("vel_x", "vel_y", "vel_z")], -1)
    dyaw = (sb["gi"][:, G["yaw"]] - sa["gi"][:, G["yaw"]]).float() * (2 * math.pi / 65536)
    t = torch.cat([
        symlog((pb - pa) / 50), symlog((vb - va) / 10),
        symlog((sb["gf"][:, Fc["fwd_vel"]] - sa["gf"][:, Fc["fwd_vel"]]) / 10).unsqueeze(-1),
        torch.sin(dyaw).unsqueeze(-1), torch.cos(dyaw).unsqueeze(-1),
        ((sb["gi"][:, G["health"]] - sa["gi"][:, G["health"]]).float() / 256).unsqueeze(-1),
        symlog((sb["gi"][:, G["coins"]] - sa["gi"][:, G["coins"]]).float()).unsqueeze(-1),
        (sb["gi"][:, G["stars"]] - sa["gi"][:, G["stars"]]).float().unsqueeze(-1),
    ], -1)
    return t, u32(sb["gi"][:, G["action"]]) & 0x1FF, pa, pb, va


def masked_mean(x, m):
    m = m.float()
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(x)
    return (x * m).sum() / m.sum().clamp(min=1)


def unsymlog(y):
    return torch.sign(y) * torch.expm1(y.abs())


def auc(score, label):
    score, label = score.float().flatten(), label.float().flatten()
    npos, nneg = label.sum(), (1 - label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = torch.empty_like(score)
    r[score.argsort()] = torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype)
    return float((r[label > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-9))


def compute(model, sch, bt, cfg, phase2, train=True):
    """Forward pass and every loss; returns (total, logs, extras for eval)."""
    B = bt["k"].shape[0]
    dev = bt["k"].device
    sa, sb, sg = bt["a"], bt["b"], bt["g"]
    # Masked-object prediction on the k-step futures (their z only feeds the
    # inverse head; dynamics targets come from the raw arrays).
    om_b = sb["om"]
    mask_obj = (torch.rand(om_b.shape, device=dev) < cfg.mask_rate) & om_b if train else torch.zeros_like(om_b)
    no_mask = torch.zeros_like(om_b)
    allm = torch.cat([no_mask, mask_obj, no_mask], 0)
    z, h, pm = model.enc(cat_states(sa, sb, sg), allm)
    za, zb, zg = z[:B], z[B : 2 * B], z[2 * B :]
    ha, pma, hb = h[:B], pm[:B], h[B : 2 * B]
    L, logs, ex = {}, {}, {}
    k, has_k = bt["k"], bt["has_k"]

    # Forward dynamics.
    mt, aid, pa, pb, va = mario_targets(sch, sa, sb)
    dm, do = model.dynamics(za, ha, k, bt["pads"], bt["pmask"])
    L["dyn_mario"] = masked_mean(F.huber_loss(dm[:, :12].float(), mt, reduction="none", delta=1.0), has_k)
    L["dyn_action"] = masked_mean(F.cross_entropy(dm[:, 12:].float(), aid, reduction="none"), has_k) * 0.5
    om = bt["obj_dm"] & has_k[:, None]
    L["dyn_obj"] = masked_mean(F.huber_loss(do[..., :3].float(), symlog(bt["obj_d"] / 50), reduction="none"), om)
    L["dyn_alive"] = masked_mean(F.binary_cross_entropy_with_logits(do[..., 3].float(), bt["obj_alive"], reduction="none"), om) * 0.5
    ex["dpos_pred"] = unsymlog(dm[:, :3].float()) * 50
    ex["dpos_true"], ex["k"], ex["has_k"], ex["cv"] = pb - pa, k, has_k, va * k[:, None].float()
    ex["act_acc"] = (dm[:, 12:].argmax(-1) == aid).float()

    # Inverse dynamics: stick as 32 bins + Gaussian residual, buttons Bernoulli.
    io = model.inverse(za, zb, k, bt["pmask"]).float()
    pads = bt["pads"]
    sx = ((((pads >> 8) & 0xFF) ^ 0x80) - 0x80) + 128
    sy = (((pads & 0xFF) ^ 0x80) - 0x80) + 128
    bx, by = (sx // 8).clamp(0, 31), (sy // 8).clamp(0, 31)
    rx, ry = (sx.float() - (bx * 8 + 3.5)) / 4, (sy.float() - (by * 8 + 3.5)) / 4
    btn = (pads >> 16) & 0xFFFF
    bbits = torch.stack([(btn >> b) & 1 for b in BUTTON_BITS], -1).float()
    imask = bt["pmask"] & has_k[:, None]
    ce = F.cross_entropy(io[..., :32].transpose(1, 2), bx, reduction="none") + F.cross_entropy(io[..., 32:64].transpose(1, 2), by, reduction="none")
    mu, lv = io[..., 64:66], io[..., 66:68].clamp(-6, 4)
    nll = 0.5 * (lv + (torch.stack([rx, ry], -1) - mu) ** 2 / lv.exp()).sum(-1)
    L["inv_stick"] = masked_mean(ce, imask) * 0.5
    L["inv_resid"] = masked_mean(nll, imask) * 0.1
    L["inv_btn"] = masked_mean(F.binary_cross_entropy_with_logits(io[..., 68:82], bbits, reduction="none").mean(-1), imask) * 2
    ex["inv_x"] = ((io[..., :32].argmax(-1) == bx).float(), imask)
    ex["inv_y"] = ((io[..., 32:64].argmax(-1) == by).float(), imask)
    ex["inv_btn"] = (((io[..., 68:82] > 0).float() == bbits).float().mean(-1), imask)

    # Events within 4 / 16 / 64 frames.
    el = model.events(za).float().view(B, len(KS), N_EVENTS)
    L["events"] = masked_mean(F.binary_cross_entropy_with_logits(el, bt["ev"], reduction="none").mean(-1), bt["evm"])
    ex["ev"] = (el, bt["ev"], bt["evm"])

    # Masked objects: behaviour, position, velocity, action.
    if mask_obj.any():
        mo = model.mask_out(hb[:, 12 : 12 + 64]).float()
        nb = sch.n_bhv
        Oi, Of = sch.oi, sch.of
        bhv = sb["oi"][..., Oi["bhv"]].long().clamp(0, nb - 1)
        pos_b = torch.stack([sb["gf"][:, sch.gf[k]] for k in ("pos_x", "pos_y", "pos_z")], -1)
        rel = sb["of"][..., [Of["pos_x"], Of["pos_y"], Of["pos_z"]]] - pos_b[:, None]
        vel = sb["of"][..., [Of["vel_x"], Of["vel_y"], Of["vel_z"]]]
        oact = sb["oi"][..., Oi["action"]].long().clamp(0, 63)
        lm = F.cross_entropy(mo[..., :nb].transpose(1, 2), bhv, reduction="none") \
            + F.huber_loss(mo[..., nb : nb + 3], symlog(rel / 1000), reduction="none").sum(-1) \
            + F.huber_loss(mo[..., nb + 3 : nb + 6], symlog(vel / 10), reduction="none").sum(-1) \
            + 0.5 * F.cross_entropy(mo[..., nb + 6 :].transpose(1, 2), oact, reduction="none")
        L["masked"] = masked_mean(lm, mask_obj) * 0.5
        ex["mask_acc"] = ((mo[..., :nb].argmax(-1) == bhv).float(), mask_obj)

    if phase2:
        # Time distance: expectile regression on log frames toward the
        # smallest bound (tree labels are upper bounds: tau 0.2; TAS tight: 0.5),
        # plus a QRL-style push apart of random pairs.
        d = model.distance(za, zg)
        pred = torch.log1p(d)
        tgt = torch.log1p(bt["dist"])
        u = tgt - pred
        tau = torch.where(bt["tight"], 0.5, 0.2)
        L["dist"] = masked_mean(torch.abs(tau - (u < 0).float()) * u**2, bt["dist_mask"])
        d_rand = model.distance(za, zg.roll(1, 0))
        L["qrl"] = 0.1 * F.relu(math.log1p(cfg.qrl_cap) - torch.log1p(d_rand)).mean()
        L["dist_imp"] = 0.1 * masked_mean(F.relu(math.log1p(5000.0) - pred), bt["impossible"])
        ex["dist"] = (d.detach(), bt["dist"], bt["dist_mask"], bt["tight"])
        # Time to star X: expectile toward the smallest bound (tree: upper
        # bounds, tau 0.3; TAS: tight, 0.5).
        if bt["star_m"].any():
            ds = model.distance_star(za, bt["star_x"])
            us = torch.log1p(bt["star_t"]) - torch.log1p(ds)
            tau_s = torch.where(bt["star_tight"], 0.5, 0.3)
            L["star"] = masked_mean(torch.abs(tau_s - (us < 0).float()) * us**2, bt["star_m"])
            # Bootstrapped target (literature.md, rec. 1): d(a, X) <= k + d(b, X)
            # for the state b k frames later, so labels from other branches
            # and the model's own short-range accuracy pull long estimates
            # down to the best known way; no gradient through d(b, X).
            tdm = bt["star_m"] & has_k & (bt["star_t"] > k.float())
            if tdm.any():
                with torch.no_grad():
                    y = k.float() + model.distance_star(zb, bt["star_x"])
                    tgt = torch.minimum(bt["star_t"], y)
                ut = torch.log1p(tgt) - torch.log1p(ds)
                L["star_td"] = 0.5 * masked_mean(torch.abs(0.5 - (ut < 0).float()) * ut**2, tdm)
            ex["star"] = (ds.detach(), bt["star_t"], bt["star_m"], bt["star_tight"])
        # Subgoal head: mixture negative log-likelihood of the position ~H frames on.
        # Edge mode (v6): credit-weighted end position, zone change, budget and
        # objective kind of objective-rrt edges toward star X.
        if "edge_m" in bt and bt["edge_m"].any():
            em = bt["edge_m"]
            w = bt["edge_w"][em]
            hidx = torch.full((int(em.sum()),), 2, dtype=torch.long, device=dev)
            logit, mu, ls, zl = model.subgoal(za[em], bt["edge_x"][em], hidx)
            y = symlog(bt["edge_dp"][em] / 1000)[:, None]
            comp = -0.5 * (((y - mu) / ls.exp()) ** 2 + 2 * ls + math.log(2 * math.pi)).sum(-1)
            nll = -torch.logsumexp(torch.log_softmax(logit, -1) + comp, -1)
            zb_ = F.binary_cross_entropy_with_logits(zl, bt["edge_zone"][em], reduction="none")
            eo = model.subgoal_edge(za[em], bt["edge_x"][em])
            bud, kl = eo["budget"], eo["kind"]
            lb = F.huber_loss(bud, torch.log1p(bt["edge_frames"][em]), reduction="none")
            lk = F.cross_entropy(kl, bt["edge_kind"][em], reduction="none")
            # The rest of the learned objective: style, events, speed, verb and
            # (teacher-forced on the verb) which object.
            ls_ = F.cross_entropy(eo["style"], bt["edge_style"][em], reduction="none")
            le = F.binary_cross_entropy_with_logits(eo["events"], bt["edge_ev"][em], reduction="none").sum(-1) * 0.25
            lsp = F.huber_loss(eo["speed"], bt["edge_speed"][em] / 32, reduction="none")
            lv = F.cross_entropy(eo["verb"], bt["edge_verb"][em], reduction="none")
            tgt = bt["edge_obj"][em]
            has_o = tgt >= 0
            lo = torch.zeros_like(lv)
            if has_o.any():
                plg = model.object_pointer(za[em][has_o], ha[em][has_o], pma[em][has_o], bt["edge_x"][em][has_o], bt["edge_verb"][em][has_o])
                lo[has_o] = F.cross_entropy(plg, tgt[has_o], reduction="none")
                ex["edge_ptr"] = (plg.argmax(-1) == tgt[has_o]).float(), w[has_o]
            L["edge"] = 0.2 * ((nll + zb_ + lb + 0.5 * lk + 0.5 * ls_ + le + lsp + 0.5 * lv + 0.5 * lo) * w).sum() / w.sum().clamp(min=1e-3)
            ex["edge"] = (kl.argmax(-1) == bt["edge_kind"][em]).float(), w, (bud.expm1() / bt["edge_frames"][em].clamp(min=1)).detach()
            ex["edge_style"] = (eo["style"].argmax(-1) == bt["edge_style"][em]).float(), w
            ex["edge_verb"] = (eo["verb"].argmax(-1) == bt["edge_verb"][em]).float(), w
        if "sub_m" in bt and bt["sub_m"].any():
            sm = bt["sub_m"]
            logit, mu, ls, zl = model.subgoal(za[sm], bt["star_x"][sm], bt["sub_h"][sm])
            y = symlog(bt["sub_dp"][sm] / 1000)[:, None]
            comp = -0.5 * (((y - mu) / ls.exp()) ** 2 + 2 * ls + math.log(2 * math.pi)).sum(-1)
            nll = -torch.logsumexp(torch.log_softmax(logit, -1) + comp, -1)
            L["subgoal"] = 0.1 * nll.mean() + 0.1 * F.binary_cross_entropy_with_logits(zl, bt["sub_zone"][sm])
            best = mu.gather(1, logit.argmax(-1)[:, None, None].expand(-1, 1, 3))[:, 0]
            ex["sub_err"] = (unsymlog(best) * 1000 - bt["sub_dp"][sm]).norm(dim=-1).detach()
            ex["sub_move"] = bt["sub_dp"][sm].norm(dim=-1)
        # Goal-conditioned tail: reachability and policy.
        # Goal tokens: REGION, OBJECTIVE, STATE. GOAL_OBJECTIVE is always off:
        # its kind is the goal's own edge's objective, which is the policy
        # label (and a reachability shortcut). It needs real objective goals
        # (the search's attempts) before it can be trained.
        keep = torch.rand(B, 3, device=dev) < (cfg.goal_keep if train else 1.0)
        keep[:, 1] = False
        pick = torch.randint(0, 2, (B,), device=dev) * 2  # REGION or STATE
        keep[torch.arange(B, device=dev), pick] = True
        gt, gm = model.goal_tokens(sa, sg, zg, bt["g_kind"], keep)
        rl, pol = model.goal_tail(ha, pma, gt, gm, bt["budget"])
        rl, pol = rl.float(), pol.float()
        L["reach"] = masked_mean(F.binary_cross_entropy_with_logits(rl, bt["reach"], reduction="none"), bt["rmask"])
        ex["reach"] = (rl, bt["reach"], bt["rmask"], bt["rtype"])
        nk = model.n_kinds
        tree_a = bt["pol"] > 0  # TAS samples (class 0) have no controller label
        pmask = bt["pmask_pol"]
        L["pol_cls"] = masked_mean(F.cross_entropy(pol[:, :nk], bt["pol"], reduction="none"), pmask & tree_a) * 0.5
        tm, tlv = pol[:, nk : nk + 6], pol[:, nk + 6 :].clamp(-6, 4)
        # TAS samples are expert data: their stick / button summary weighs 2x.
        wt = torch.where(tree_a, 1.0, 2.0)
        L["pol_theta"] = masked_mean(0.5 * (tlv + (bt["theta"] - tm) ** 2 / tlv.exp()).sum(-1) * wt, pmask) * 0.1
        ex["pol_acc"] = ((pol[:, :nk].argmax(-1) == bt["pol"]).float(), pmask & tree_a)
        ex["pol"] = (bt["pol"], pmask & tree_a, bt["g_same_node"])
    total = sum(L.values())
    logs = {f"loss/{k}": float(v.detach()) for k, v in L.items()}
    logs["loss/total"] = float(total.detach())
    return total, logs, ex


@torch.no_grad()
def evaluate(model, sch, loader_iter, n, dev, cfg):
    model.eval()
    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(v)

    for _ in range(n):
        bt = to_dev(next(loader_iter), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, logs, ex = compute(model, sch, bt, cfg, True, train=False)
        for k, v in logs.items():
            add("val_" + k, torch.tensor([v]))
        err = (ex["dpos_pred"] - ex["dpos_true"]).norm(dim=-1)
        cv = (ex["cv"] - ex["dpos_true"]).norm(dim=-1)
        for kk in KS:
            m = ex["has_k"] & (ex["k"] == kk)
            add(f"dyn/pos_err_k{kk}", err[m].cpu())
            add(f"dyn/constvel_err_k{kk}", cv[m].cpu())
        add("dyn/action_acc", ex["act_acc"][ex["has_k"]].cpu())
        for key in ("inv_x", "inv_y", "inv_btn", "mask_acc", "pol_acc"):
            if key in ex:
                v, m = ex[key]
                add(f"acc/{key}", v[m].cpu())
        el, ev, evm = ex["ev"]
        add("ev_logit", el[:, 1][evm[:, 1]].cpu())
        add("ev_label", ev[:, 1][evm[:, 1]].cpu())
        rl, r, rm, rt = ex["reach"]
        add("reach_logit", rl[rm].cpu())
        add("reach_label", r[rm].cpu())
        add("reach_type", rt[rm].cpu())
        pl, pm_, same = ex["pol"]
        hit, _ = ex["pol_acc"]
        add("pol_label", pl[pm_].cpu())
        add("pol_hit", hit[pm_].cpu())
        add("pol_same", same[pm_].cpu())
        d, dt, dm, tight = ex["dist"]
        if "edge" in ex:
            hit, w, br = ex["edge"]
            add("edge_hit", hit.cpu()); add("edge_w", w.cpu()); add("edge_br", br.cpu())
            for k in ("edge_style", "edge_verb", "edge_ptr"):
                if k in ex:
                    add(k + "_hit", ex[k][0].cpu()); add(k + "_w", ex[k][1].cpu())
        if "sub_err" in ex:
            add("sub_err", ex["sub_err"].cpu())
            add("sub_move", ex["sub_move"].cpu())
        if "star" in ex:
            sd, stt, sm, stight = ex["star"]
            add("s_pred_tas", sd[sm & stight].cpu()); add("s_true_tas", stt[sm & stight].cpu())
            add("s_pred_tree", sd[sm & ~stight].cpu()); add("s_true_tree", stt[sm & ~stight].cpu())
        add("d_pred_tas", d[dm & tight].cpu())
        add("d_true_tas", dt[dm & tight].cpu())
        add("d_pred_tree", d[dm & ~tight].cpu())
        add("d_true_tree", dt[dm & ~tight].cpu())
    out = {}
    for k, v in acc.items():
        if k.startswith(("val_", "acc/")) or k == "dyn/action_acc":
            out[k] = float(torch.cat(v).float().mean())
        elif k.startswith("dyn/"):
            # Median: PU-speed TAS frames (|v| ~ 1e6) make means meaningless.
            x = torch.cat(v).float()
            if len(x):
                out[k] = float(x.median())
    el, ev = torch.cat(acc["ev_logit"]), torch.cat(acc["ev_label"])
    from sm64bb.schema import Schema as _S  # noqa: F401
    for j, name in enumerate(sch.events):
        a = auc(el[:, j], ev[:, j])
        if not math.isnan(a):
            out[f"events_auc16/{name}"] = a
    rl, rlab, rt = torch.cat(acc["reach_logit"]), torch.cat(acc["reach_label"]), torch.cat(acc["reach_type"])
    out["reach/auc"] = auc(rl, rlab)
    for t, name in [(1, "impossible"), (2, "tas_budget"), (3, "tree_budget")]:
        m = (rt == 0) | (rt == t)
        out[f"reach/auc_{name}"] = auc(rl[m], rlab[m])
    m = rt != 1
    out["reach/auc_budget_only"] = auc(rl[m], rlab[m])
    pl, ph, ps = torch.cat(acc["pol_label"]), torch.cat(acc["pol_hit"]), torch.cat(acc["pol_same"])
    if len(pl):
        out["policy/majority_baseline"] = float(torch.bincount(pl).max()) / len(pl)
        out["policy/acc_goal_same_edge"] = float(ph[ps].mean()) if ps.any() else float("nan")
        out["policy/acc_goal_other_edge"] = float(ph[~ps].mean()) if (~ps).any() else float("nan")
        out["policy/frac_goal_same_edge"] = float(ps.float().mean())
    out["dist/spearman_tas_heldout"] = spearman(torch.cat(acc["d_pred_tas"]), torch.cat(acc["d_true_tas"]))
    out["dist/spearman_tree_val"] = spearman(torch.cat(acc["d_pred_tree"]), torch.cat(acc["d_true_tree"]))
    if "edge_hit" in acc:
        h, w, br = torch.cat(acc["edge_hit"]), torch.cat(acc["edge_w"]), torch.cat(acc["edge_br"])
        out["edge/kind_acc_weighted"] = float((h * w).sum() / w.sum().clamp(min=1e-3))
        out["edge/budget_ratio_median"] = float(br.median())
        for k in ("style", "verb", "ptr"):
            if f"edge_{k}_hit" in acc:
                h, w = torch.cat(acc[f"edge_{k}_hit"]), torch.cat(acc[f"edge_{k}_w"])
                out[f"edge/{k}_acc_weighted"] = float((h * w).sum() / w.sum().clamp(min=1e-3))
    if "sub_err" in acc:
        # Subgoal: median error of the most likely mode against how far the
        # path actually moved (a "stay put" guess scores 1.0).
        e, mv = torch.cat(acc["sub_err"]), torch.cat(acc["sub_move"])
        out["subgoal/median_err"] = float(e.median())
        out["subgoal/err_vs_move"] = float((e / mv.clamp(min=1)).median())
    for src in ("tas", "tree"):
        if f"s_pred_{src}" in acc:
            p_, t_ = torch.cat(acc[f"s_pred_{src}"]), torch.cat(acc[f"s_true_{src}"])
            if len(p_) > 3:
                out[f"star/spearman_{src}"] = spearman(p_, t_)
                out[f"star/log_ratio_{src}"] = float((torch.log1p(p_) - torch.log1p(t_)).abs().mean())
                far = t_ > 1000
                if far.sum() > 3:
                    out[f"star/spearman_{src}_far"] = spearman(p_[far], t_[far])
    tp, tt = torch.cat(acc["d_pred_tas"]), torch.cat(acc["d_true_tas"])
    if len(tp):
        out["dist/log_ratio_tas"] = float((torch.log1p(tp) - torch.log1p(tt)).abs().mean())
    model.train()
    return {k: v for k, v in out.items() if not math.isnan(v)}


@torch.no_grad()
def init_star_goals(model, dev):
    """star_phi rows for goals the 2026 TAS collects: phi of its state there."""
    from follow import encode_reference
    from sm64bb.data import goal_id
    from sm64bb.refs import build_reference, make_pool
    from sm64bb.runtime import Runtime

    class _RT:
        pass
    rt = Runtime.__new__(Runtime)
    rt.m, rt.dev, rt.chunk, rt.step = model, dev, 256, -1
    ref = build_reference("tas120")
    model.eval()
    zr, rframes = encode_reference(ref, rt, make_pool(1), 0)
    model.train()
    n = 0
    for m in ref.milestones:
        g = goal_id(m)
        if g is None:
            continue
        i = int(np.clip(m["frame"] // 4 - 1, 0, len(rframes) - 1))
        model.star_phi.weight[g] = model.phi(zr[i : i + 1].float().to(dev))[0]
        n += 1
    print(f"star goals initialised from {n} TAS collection states", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--phase1", type=float, default=0.25, help="fraction of steps with phase-1 losses only")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tas_frac", type=float, default=0.25)
    ap.add_argument("--mask_rate", type=float, default=0.15)
    ap.add_argument("--goal_keep", type=float, default=0.7)
    ap.add_argument("--qrl_cap", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--v4", action="store_true", help="all extracted trees, time-to-star labels and goals")
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--edges", action="store_true", help="v6: credit-weighted objective-rrt edges (subgoal edge mode)")
    ap.add_argument("--v7", action="store_true", help="v2-era sources (data/v7 + TAS), no star labels")
    ap.add_argument("--class-min", dest="class_min", action="store_true",
                    help="time-to-star labels: fastest of any tree from the same state class (v5)")
    cfg = ap.parse_args()

    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    sch = Schema()
    nodes = os.path.join(RUN, "nodes.csv")
    t0 = time.time()
    srcs = v7_sources() if cfg.v7 else v4_sources() if cfg.v4 else SOURCES
    tr = Store(srcs, nodes, sch).setup_pairs("train")
    va = Store(srcs, nodes, sch).setup_pairs("val")
    if cfg.v4:
        tr.star_labels(class_min=cfg.class_min)
        va.star_labels(class_min=cfg.class_min)
        if cfg.edges:
            tr.edges()
            va.edges()
            print(f"edges: train {len(tr.edge_a)}, val {len(va.edge_a)}; credit weight mean {tr.edge_w.mean():.2f}; verbs "
                  + str(dict(zip(VERBS, np.bincount(tr.edge_verb, minlength=len(VERBS)).tolist()))), flush=True)
        print(f"sources: {[os.path.basename(s if isinstance(s, str) else s[0]) for s in srcs]}; "
              f"star labels train {len(tr.lab_x)}, val {len(va.lab_x)}", flush=True)
    print(f"data: {tr.N} samples, train tree {len(tr.tree_idx)} tas {len(tr.tas_idx)}; "
          f"val tree {len(va.tree_idx)} tas {len(va.tas_idx)} ({time.time() - t0:.0f} s)", flush=True)

    model = Backbone(sch, len(tr.kind_names), cfg.d, cfg.layers, cfg.heads).to(dev)
    if cfg.init:
        # Warm start: every weight whose shape still fits (the policy's kind
        # count changes with the trees), then the star goals from the init
        # model's view of the TAS's states as it collects each star.
        ck = torch.load(cfg.init, map_location="cpu")
        own = model.state_dict()
        keep = {k: v for k, v in ck["model"].items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"init from {cfg.init} (step {ck['step']}): {len(keep)}/{len(own)} tensors", flush=True)
        if "star_phi.weight" not in ck["model"]:
            init_star_goals(model, dev)
    n_par = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.enc.parameters())
    print(f"model: {n_par / 1e6:.2f} M parameters (encoder {n_enc / 1e6:.2f} M)", flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "types" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "types" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup)
                                              * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps))))
    step_fn = torch.compile(compute) if cfg.compile else compute

    mk = lambda st, seed, nw: torch.utils.data.DataLoader(Stream(st, cfg.batch, seed, cfg.tas_frac), batch_size=None,
                                                          num_workers=nw, collate_fn=keep_numpy, prefetch_factor=4,
                                                          persistent_workers=True, timeout=300)
    train_it = iter(mk(tr, cfg.seed + 1, cfg.workers))
    val_it = iter(mk(va, 12345, 2))

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(HERE, 'mlflow.db')}")
    exp = "sm64-backbone"
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=os.path.join(HERE, "mlartifacts"))
    mlflow.set_experiment(exp)
    ckdir = os.path.join(HERE, "checkpoints", "v7") if cfg.v7 else os.path.join(HERE, "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    p1 = int(cfg.steps * cfg.phase1)
    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params({**vars(cfg), "params_M": round(n_par / 1e6, 2), "encoder_params_M": round(n_enc / 1e6, 2),
                           "samples": tr.N, "train_tree": len(tr.tree_idx), "train_tas": len(tr.tas_idx),
                           "val_tree": len(va.tree_idx), "val_tas": len(va.tas_idx), "phase1_steps": p1,
                           "tree_run": "indef-mgel-lg-pf-s2", "tas": "full-120star-2026-U.m64",
                           "val_tas_levels": "11 (WDW), 22 (LLL)", "n_kinds": len(tr.kind_names),
                           "val_tree": "whole subtrees off the route (~5% of recorded nodes)",
                           "goal_objective_token": "off (leaks the policy label)",
                           "reach_negatives": "impossible (fewer stars), TAS budget <50%, tree budget <1/8"})
        mlflow.log_dict({"kinds": tr.kind_names, "events": sch.events}, "labels.json")
        mlflow.log_artifact(os.path.join(HERE, "data", "schema.json"))
        print(f"mlflow run {run.info.run_id}", flush=True)
        model.train()
        t_last, best, best_tas = time.time(), None, None
        for step in range(cfg.steps + 1):
            phase2 = step >= p1
            bt = to_dev(next(train_it), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logs, _ = step_fn(model, sch, bt, cfg, phase2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % cfg.log_every == 0:
                dt = time.time() - t_last
                t_last = time.time()
                logs.update({"lr": sched.get_last_lr()[0], "grad_norm": float(gn), "phase": 2 if phase2 else 1,
                             "steps_per_s": cfg.log_every / dt if step else 0.0})
                mlog.log_metrics(logs, step=step)
                if step % (cfg.log_every * 10) == 0:
                    print(f"step {step} phase {2 if phase2 else 1} loss {logs['loss/total']:.3f} "
                          f"{logs['steps_per_s']:.2f} it/s", flush=True)
            if step and (step % cfg.eval_every == 0 or step == cfg.steps):
                m = evaluate(model, sch, val_it, cfg.eval_batches, dev, cfg)
                mlog.log_metrics(m, step=step)
                print(json.dumps({k: round(v, 4) for k, v in m.items() if not k.startswith("events_auc")}), flush=True)
                ck = {"model": model.state_dict(), "cfg": vars(cfg), "kinds": tr.kind_names, "step": step}
                torch.save(ck, os.path.join(ckdir, "last.pt"))
                tas = m.get("dist/spearman_tas_heldout")
                if phase2 and tas is not None and (best_tas is None or tas > best_tas):
                    best_tas = tas
                    torch.save(ck, os.path.join(ckdir, "best_tas.pt"))
                    mlog.log_metric("best_tas_spearman", best_tas, step=step)
                score = m.get("val_loss/total")
                if phase2 and score is not None and (best is None or score < best):
                    best = score
                    torch.save(ck, os.path.join(ckdir, "best.pt"))
                    mlog.log_metric("best_val_total", best, step=step)
        mlflow.log_artifact(os.path.join(ckdir, "last.pt"), "checkpoints")
        for f in ("best.pt", "best_tas.pt"):
            if os.path.exists(os.path.join(ckdir, f)):
                mlflow.log_artifact(os.path.join(ckdir, f), "checkpoints")


if __name__ == "__main__":
    main()
