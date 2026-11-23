#!/usr/bin/env python3
"""RL as a GA seed, round 2: controls, long stretches in other levels, and the
King record's own stretch. From one fixed start state, same budgets. Arms:
  none       unseeded
  demo       seeded with the demo the RL trained on
  route      seeded with the stretch of the source route
  rl         seeded with the RL route (RL trained on the demo)
  rl2        a second RL training (another seed) on the same demo
  ctrl-*     seeded with other unseeded GA routes: unrelated but valid
Results in logs/rlv2/seed2/results.jsonl. Resumable.

    .venv/bin/python rlv2/seed_legs2.py --threads 4 --cases controls
"""
import argparse, json, os, subprocess
import numpy as np
import routes as R
from seed_legs import PY, ga

OUT = os.path.join(R.BB, "logs", "rlv2", "seed2")
Q2 = "logs/mend/q15-s2-star4-8492.m64"
Q3 = "logs/mend/q15-s3-star5-11672.m64"
K = "logs/mend/king-3541.m64"
S = "oss2/runs/legs-speed-blj-fresh-s1/route.m64"
W = "oss2/runs/legs-fast-king-walk-s1/route.m64"
RT = os.path.join(R.V2, "tools", "spec", "routes")
# name, source movie, start frame, leg: (route file, leg name) or end frame (a stretch: reach the route's position there)
GROUPS = {
    "controls": [("Hill1", W, 2437, ("fast-king-walk.json", "Hill1")), ("BobPainting", S, 1951, ("fast-king-speed-blj.json", "BobPainting"))],
    "king": [("King-BobPainting", K, 1945, ("fast-king-blj-tas.ramp0.json", "BobPainting"))],
    "levels": [("WF3-to-dialog", Q2, 6277, 7639), ("WF1", Q2, 4154, 4692), ("BoB-q15", Q2, 2356, 3019), ("CCM-q15", Q3, 8572, 9686), ("Castle-to-WF", Q3, 3700, 4521), ("WF2-q3", Q3, 6256, 7253)],
}


def stretch_spec(name, pos, level, area):
    return dict(name=name, zone=[level, area], stable_frames=0,
                phases=[dict(name="p", target=dict(sphere=dict(center=[round(float(x)) for x in pos], r=150)), done=dict(dist=dict(le=0)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["controls", "king", "levels"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rl-frames", type=float, default=2e6)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--budget", nargs=2, type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = open(os.path.join(OUT, "results.jsonl"), "a")
    sm = R.setup()
    pool = R.make_pool(sm, 1, 1)
    for grp in a.cases:
        for name, movie, start, what in GROUPS[grp]:
            if a.only and name not in a.only:
                continue
            d = os.path.join(OUT, name)
            os.makedirs(d, exist_ok=True)
            pads = R.read_movie(os.path.join(R.V2, movie))
            budget = {"population": 128, "generations": 60}
            if isinstance(what, tuple):
                doc = json.load(open(os.path.join(RT, what[0])))
                spec = {k: v for k, v in [l for l in doc["legs"] if l["name"] == what[1]][0].items() if k != "reference"}
                spec["stable_frames"] = 0
                budget = doc.get("budget", budget)
            else:
                pool.reset([0])
                pool.step([0], pads[:what][None, :])
                st = pool.status([0])
                spec = stretch_spec(name, st["pos"][0], int(st["level"][0]), int(st["area"][0]))

            def route(tag, ref=None):
                l = dict(spec)
                if ref:
                    l["reference"] = {"m64": ref}
                p = os.path.join(d, f"{tag}.json")
                json.dump(dict(name=name, start={"m64": movie, "frames": start}, budget=budget, legs=[l], seed_ref=bool(ref)), open(p, "w"))
                return p

            base = route("none")
            src = os.path.join(R.V2, movie)
            # How long the source route takes here, by the leg's own fold.
            pool.reset([0])
            pool.step([0], pads[:start][None, :])
            rid, names = pool.rl_route(json.dumps(dict(legs=[spec])))
            e0 = pool.rl_bind([0], rid, 0, 0, False)
            r0 = pool.rl_raw([0], pads[start:][None, :], stop_at_leg=True)
            _, dn = pool.rl_played(0)
            route_frames = (dn[-1][1] - start) if dn else None
            rec = dict(leg=name, group=grp, start=start, movie=movie, route_frames=route_frames, eta0=float(e0[0]))
            print(json.dumps(rec), flush=True)
            fr0 = ga(base, os.path.join(d, "demo"), 1, a.threads)
            rec["demo_frames"], rec["demo_ga"] = fr0[0], fr0[1]
            print(name, "demo", fr0, flush=True)
            demo = os.path.join(d, "demo", "route.m64") if fr0[0] is not None else src

            def train(tag, seed):
                nm = f"seed2-{name}-{tag}"
                ck = os.path.join(R.BB, "checkpoints", "rlv2", f"{nm}.best.pt")
                log = os.path.join(R.BB, "logs", "rlv2", f"{nm}.log")
                if not os.path.exists(log + ".done"):
                    subprocess.run(["nice", "-n", "5", PY, os.path.join(R.HERE, "train_rl.py"), "--route", base, "--demo", f"{demo}@{start}", "--legs", spec["name"], "--frames", str(a.rl_frames),
                                    "--name", nm, "--envs", "64", "--threads", str(a.threads), "--rollout", "32", "--eval-every", "20", "--bc-steps", "1000", "--seed", str(seed)],
                                   stdout=open(log, "w"), stderr=subprocess.STDOUT)
                    open(log + ".done", "w").close()
                if not os.path.exists(ck):
                    return None, None
                ev = subprocess.run(["nice", "-n", "5", PY, os.path.join(R.HERE, "eval_legs.py"), ck, "--samples", "64", "--threads", str(a.threads)], capture_output=True, text=True)
                ls = [json.loads(x) for x in ev.stdout.splitlines() if x.startswith("{")]
                if not ls:
                    return None, ev.stderr[-300:]
                return ls[0].get("greedy_m64") or ls[0].get("best_m64"), {k: ls[0].get(k) for k in ("greedy", "sampled", "frames_trained")}

            arms = {"none": base}
            if fr0[0] is not None:
                arms["demo"] = route("demo-seed", demo)
            if route_frames:
                arms["route"] = route("route-seed", src)
            m1, info = train("a", 0)
            rec["rl_policy"] = info
            if m1:
                arms["rl"] = route("rl-seed", m1)
            if grp == "controls":
                m2, info2 = train("b", 1)
                rec["rl2_policy"] = info2
                if m2:
                    arms["rl2"] = route("rl2-seed", m2)
            for arm, rp in arms.items():
                rec[arm] = []
                for s in range(2, a.seeds + 2):
                    r = ga(rp, os.path.join(d, f"{arm}-{s}"), s, a.threads)
                    rec[arm].append(r[0])
                    rec[arm + "_ga"] = r[1]
            if grp == "controls":
                # Unrelated but valid: the unseeded arm's own routes, one per control.
                ok = sorted([(f, s) for f, s in zip(rec["none"], range(2, a.seeds + 2)) if f is not None])
                for tag, (f, s) in zip(("fast", "mid", "slow"), (ok[0], ok[len(ok) // 2], ok[-1])):
                    src2 = os.path.join(d, f"none-{s}", "route.m64")
                    rp = route(f"ctrl-{tag}", src2)
                    k = f"ctrl_{tag}"
                    rec[k], rec[k + "_from"] = [], f
                    for s2 in range(2, a.seeds + 2):
                        rec[k].append(ga(rp, os.path.join(d, f"{k}-{s2}"), s2, a.threads)[0])
            res.write(json.dumps(rec) + "\n"), res.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
