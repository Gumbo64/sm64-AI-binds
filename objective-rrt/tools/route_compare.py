#!/usr/bin/env python3
"""Tabulates route bench runs: tools/route_compare.py <glob of run dirs>
(default runs/route-cmp-*): per method, frames saved per seed, mean, and
improvements / trials / repairs."""
import glob, json, os, sys
from collections import defaultdict

dirs = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/route-cmp-*"))
by = defaultdict(list)
for d in dirs:
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    t = [json.loads(l) for l in open(os.path.join(d, "trials.jsonl"))]
    rep = sum(1 for x in t if x.get("repair") and x.get("improved"))
    by[s["method"]].append((s["saved"], s["improvements"], s["trials"], rep, s["original_frame"], s["best_frame"]))
print(f"{'method':14} {'saved per seed':22} {'mean':>6} {'impr':>5} {'trials':>7} {'repairs ok':>10}")
for m, rows in sorted(by.items(), key=lambda kv: -sum(r[0] for r in kv[1]) / len(kv[1])):
    print(f"{m:14} {' / '.join(str(r[0]) for r in rows):22} {sum(r[0] for r in rows)/len(rows):6.1f} {sum(r[1] for r in rows):5} {sum(r[2] for r in rows):7} {sum(r[3] for r in rows):10}")
