#!/usr/bin/env python3
"""Sidecar for a running train.py: after each eval, copies checkpoints/last.pt
to checkpoints/best_tas.pt when that eval's distance rank correlation on the
held-out TAS levels is the best so far. Exits when the log says the run ended.

    .venv/bin/python keep_best_tas.py logs/train_v2.log
"""
import json, os, shutil, sys, time

log, ck = sys.argv[1], "checkpoints"
# Only evals from now on have checkpoints: start from the current last.pt.
evals = [json.loads(l) for l in open(log) if l.startswith("{")]
best, seen = evals[-1].get("dist/spearman_tas_heldout", -1), len(evals)
shutil.copyfile(f"{ck}/last.pt", f"{ck}/best_tas.pt")
print(f"start: eval {seen - 1} tas spearman {best:.4f} -> best_tas.pt", flush=True)
while True:
    evals = [json.loads(l) for l in open(log) if l.startswith("{")]
    for i in range(seen, len(evals)):
        v = evals[i].get("dist/spearman_tas_heldout", -1)
        if i == len(evals) - 1 and v > best:
            # last.pt is written right after the eval line is printed.
            time.sleep(20)
            shutil.copyfile(f"{ck}/last.pt", f"{ck}/best_tas.tmp")
            os.replace(f"{ck}/best_tas.tmp", f"{ck}/best_tas.pt")
            print(f"eval {i}: tas spearman {v:.4f} -> best_tas.pt", flush=True)
        best = max(best, v)
    seen = len(evals)
    if os.system("pgrep -f 'train.py --steps 80000' > /dev/null") != 0:
        break
    time.sleep(30)
