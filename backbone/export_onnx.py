#!/usr/bin/env python3
"""Exports a trained checkpoint to ONNX for objective-rrt (sm64_AI_backbone.md 8):

    .venv/bin/python export_onnx.py [checkpoints/best.pt] [out_dir=onnx]

Three graphs, all batched over the first axis:
- encoder.onnx: raw state arrays (gi, gf, pad_prev, oi, of, om, tri, tm; the
  extractor's columns, data/schema.json) -> z [B, 256], h [B, T, 256], pm [B, T].
  Run once per tree node; z is the goal-free embedding.
- distance.onnx: (za, zg) -> frames estimate (IQE quasimetric).
- goal_tail.onnx: (h, pm, gi_a, gf_a, gi_g, gf_g, zg, g_kind, keep, budget)
  -> reach logit, policy output (kind logits, theta mean, theta log-variance).

Geometry tokens are chosen by the caller (as data.Store.geometry does): the
96 static triangles of the zone nearest Mario (24 of them along the
direction to the goal), then up to 32 object triangles. Checks each graph
against PyTorch with onnxruntime on a validation batch.
"""

import json
import os
import sys

import numpy as np
import torch

from sm64bb.data import Batcher, Store
from sm64bb.model import Backbone
from sm64bb.schema import Schema

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "objective-rrt", "runs", "indef-mgel-lg-pf-s2")
SOURCES = [os.path.join(HERE, "data", "tree14"), os.path.join(HERE, "data", "tas120")]
STATE_KEYS = ["gi", "gf", "pad_prev", "oi", "of", "om", "tri", "tm"]


class Enc(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, gi, gf, pad_prev, oi, of, om, tri, tm):
        s = dict(gi=gi, gf=gf, pad_prev=pad_prev, oi=oi, of=of, om=om, tri=tri, tm=tm)
        return self.m.enc(s)


class Dist(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, za, zg):
        return self.m.distance(za, zg)


class Tail(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, h, pm, gi_a, gf_a, gi_g, gf_g, zg, g_kind, keep, budget):
        gt, gm = self.m.goal_tokens({"gi": gi_a, "gf": gf_a}, {"gi": gi_g, "gf": gf_g}, zg, g_kind, keep)
        return self.m.goal_tail(h, pm, gt, gm, budget)


def main():
    ck_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "checkpoints", "best.pt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "onnx")
    os.makedirs(out, exist_ok=True)
    ck = torch.load(ck_path, map_location="cpu")
    cfg = ck["cfg"]
    sch = Schema()
    m = Backbone(sch, len(ck["kinds"]), cfg["d"], cfg["layers"], cfg["heads"])
    m.load_state_dict(ck["model"])
    m.eval()
    va = Store(SOURCES, os.path.join(RUN, "nodes.csv"), sch).setup_pairs("val")
    bt = Batcher(va, 8, seed=7).batch()
    s = {k: torch.from_numpy(bt["a"][k]) for k in STATE_KEYS}
    sg = {k: torch.from_numpy(bt["g"][k]) for k in STATE_KEYS}
    import onnxruntime as ort

    def export(mod, args, names, outs, path, dyn):
        torch.onnx.export(mod, args, path, input_names=names, output_names=outs, dynamic_axes=dyn,
                          opset_version=18, dynamo=True)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(None, {n: a.numpy() for n, a in zip(names, args)})
        with torch.no_grad():
            ref = mod(*args)
        ref = ref if isinstance(ref, tuple) else (ref,)
        errs = [float(np.abs(g.astype(np.float64) - r.numpy().astype(np.float64)).max()) for g, r in zip(got, ref)]
        print(f"{os.path.basename(path)}: max |onnx - torch| per output {errs}")
        return got

    enc_args = tuple(s[k] for k in STATE_KEYS)
    dyn = {k: {0: "B"} for k in STATE_KEYS + ["z", "h", "pm"]}
    export(Enc(m), enc_args, STATE_KEYS, ["z", "h", "pm"], f"{out}/encoder.onnx", dyn)
    with torch.no_grad():
        za, h, pm = m.enc(s)
        zg, _, _ = m.enc(sg)
    export(Dist(m), (za, zg), ["za", "zg"], ["frames"], f"{out}/distance.onnx",
           {"za": {0: "B"}, "zg": {0: "B"}, "frames": {0: "B"}})
    B = za.shape[0]
    targs = (h, pm, s["gi"], s["gf"], sg["gi"], sg["gf"], zg, torch.from_numpy(bt["g_kind"]),
             torch.ones(B, 3, dtype=torch.bool), torch.from_numpy(bt["budget"]))
    tnames = ["h", "pm", "gi_a", "gf_a", "gi_g", "gf_g", "zg", "g_kind", "keep", "budget"]
    export(Tail(m), targs, tnames, ["reach_logit", "policy"], f"{out}/goal_tail.onnx",
           {**{k: {0: "B"} for k in tnames}, "reach_logit": {0: "B"}, "policy": {0: "B"}})
    json.dump({"kinds": ck["kinds"], "step": ck["step"], "schema": sch.raw, "n_kinds": len(ck["kinds"]),
               "policy_layout": "kind logits [n_kinds], theta mean [6], theta logvar [6]; theta = "
                                "(sin, cos of stick yaw rel. to facing, stick mag, A, B, Z fractions) over 16 frames"},
              open(f"{out}/meta.json", "w"))
    print(f"wrote {out}/ (step {ck['step']})")


if __name__ == "__main__":
    main()
