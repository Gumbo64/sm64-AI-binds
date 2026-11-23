#!/usr/bin/env python3
"""Does an RL route help the GA, beyond the demo it learned from? Per leg, from
one fixed start state, same budgets:
  1. GA unseeded, seed 1: its route is the demo.
  2. RL trained on that demo alone (no TAS).
  3. GA, seeds 2..N: seeded with nothing / the demo / the RL route.
Writes logs/rlv2/seed/results.jsonl. Resumable: finished steps are kept.

    .venv/bin/python rlv2/seed_legs.py --threads 2 --rl-frames 2e6
"""

import argparse
import json
import os
import subprocess
import sys

import routes as R

OUT = os.path.join(R.BB, "logs", "rlv2", "seed")
LEGS = os.path.join(R.V2, "builds", "legs13", "legs")
PY = os.path.join(R.BB, ".venv", "bin", "python")
S = "oss2/runs/legs-speed-blj-fresh-s1/route.m64"
W = "oss2/runs/legs-fast-king-walk-s1/route.m64"
K = "logs/kingloop/best-fast-king-blj-tas.ramp0.m64"
CASES = [("fast-king-walk.json", W, "ChompSide", 2746), ("fast-king-walk.json", W, "PoleBase", 2819), ("fast-king-speed-blj.json", S, "SpeedChain", 2360),
         ("fast-king-speed-blj.json", S, "Blj", 2521), ("fast-king-speed-blj.json", S, "Climb", 2682), ("fast-king-walk.json", W, "Hill1", 2437),
         ("fast-king-speed-blj.json", S, "BobPainting", 1951), ("fast-king-blj-tas.ramp0.json", K, "KingHit1", 2931)]


def ga(route_path, out, seed, threads):
    """-> leg frames (None: unsolved), GA frames, seconds."""
    if not os.path.exists(os.path.join(out, "route.json")):
        env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
        with open(out + ".log", "w") as f:
            subprocess.run(["nice", "-n", "5", LEGS, route_path, "--out", out, "--seed", str(seed)], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
    j = json.load(open(os.path.join(out, "route.json")))
    l = j["legs"][0]
    return l.get("frames"), l.get("ga_frames"), j["totals"].get("seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rl-frames", type=float, default=2e6)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = open(os.path.join(OUT, "results.jsonl"), "a")
    for rf, movie, leg, start in CASES:
        if a.only and leg not in a.only:
            continue
        doc = json.load(open(os.path.join(R.V2, "tools", "spec", "routes", rf)))
        spec = [l for l in doc["legs"] if l["name"] == leg][0]
        spec = {k: v for k, v in spec.items() if k != "reference"}
        spec["stable_frames"] = 0
        d = os.path.join(OUT, leg)
        os.makedirs(d, exist_ok=True)

        def route(name, ref=None):
            l = dict(spec)
            if ref:
                l["reference"] = {"m64": ref}
            p = os.path.join(d, f"{name}.json")
            json.dump(dict(name=leg, start={"m64": movie, "frames": start}, budget=doc.get("budget", {"population": 128, "generations": 60}), legs=[l], seed_ref=bool(ref)), open(p, "w"))
            return p

        base = route("none")
        fr0 = ga(base, os.path.join(d, "demo"), 1, a.threads)
        demo = os.path.join(d, "demo", "route.m64")
        print(leg, "demo", fr0, flush=True)
        if fr0[0] is None:
            res.write(json.dumps(dict(leg=leg, failed="the GA found no demo", demo=fr0)) + "\n"), res.flush()
            continue
        name = f"seed-{leg}"
        ck = os.path.join(R.BB, "checkpoints", "rlv2", f"{name}.best.pt")
        log = os.path.join(R.BB, "logs", "rlv2", f"{name}.log")
        if not os.path.exists(ck) or not os.path.exists(log + ".done"):
            subprocess.run(["nice", "-n", "5", PY, os.path.join(R.HERE, "train_rl.py"), "--route", base, "--demo", f"{demo}@{start}", "--legs", leg, "--frames", str(a.rl_frames),
                            "--name", name, "--envs", "64", "--threads", str(a.threads), "--rollout", "32", "--eval-every", "20", "--bc-steps", "1000"], stdout=open(log, "w"), stderr=subprocess.STDOUT)
            open(log + ".done", "w").close()
        rec = dict(leg=leg, start=start, demo_frames=fr0[0], demo_ga=fr0[1])
        rlm = None
        if os.path.exists(ck):
            ev = subprocess.run(["nice", "-n", "5", PY, os.path.join(R.HERE, "eval_legs.py"), ck, "--samples", "64", "--threads", str(a.threads)], capture_output=True, text=True)
            lines = [json.loads(x) for x in ev.stdout.splitlines() if x.startswith("{")]
            if lines:
                rec["rl_policy"] = {k: lines[0].get(k) for k in ("greedy", "sampled", "greedy_replay", "frames_trained")}
                rlm = lines[0].get("greedy_m64") or lines[0].get("best_m64")
            else:
                rec["rl_eval_error"] = ev.stderr[-400:]
        arms = {"none": base, "demo": route("demo-seed", demo)}
        if rlm:
            arms["rl_seed"] = route("rl-seed", rlm)
        for arm, rp in arms.items():
            rec[arm] = []
            for s in range(2, a.seeds + 2):
                r = ga(rp, os.path.join(d, f"{arm}-{s}"), s, a.threads)
                rec[arm].append(r[0])
                rec[arm + "_ga"] = r[1]
        res.write(json.dumps(rec) + "\n"), res.flush()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
