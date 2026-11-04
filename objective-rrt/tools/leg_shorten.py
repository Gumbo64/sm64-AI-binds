#!/usr/bin/env python3
"""Shortens a finished route leg by leg (star landing -> next star landing).

    tools/leg_shorten.py <work dir> <route run dir> <route end file> [--budget N] [--seeds 1,2]
                         [--threads 8] [--leg0 <run dir>] [--from K]

<route end file>: the route's last node id (legs end at the route's first
controllable castle node after each star).

For each leg k -> k+1, with P = the best inputs so far up to the star-k
landing:
  * replay: P + the route's own inputs for that leg (they drift in courses,
    so this can fail);
  * search: a fresh obrrt tree from P (start.m64, OBRRT_BOOT_NOWAIT=1) per
    seed, `budget` frames each; candidate = its soonest castle state with
    more than k stars.
Every candidate is judged the same way: replayed with tasplay (plus idle
frames), the first frame Mario is controllable in the castle with more than
k stars. The soonest becomes P for the next leg. Results are appended to
<work dir>/legs.txt; each leg's inputs are kept as <work dir>/leg<k>.m64 (raw).
"""
import argparse
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROM = os.path.join(ROOT, "..", "baserom.us.z64")
BIN = os.path.join(ROOT, "builds/legs1/obrrt")
NODEPATH = os.path.join(ROOT, "builds/target-route1/release/nodepath")
TASPLAY = os.path.join(ROOT, "builds/target-route1/release/tasplay")
CASTLE = {"6", "16", "26"}
MODAL = ("Menu::", "Dialog::", "Npc::Talk")
# Star landing / star dance / warp-in actions: not yet in control.
NOT_CONTROL = {"0x1327", "0x1926", "0x1302", "0x1904"}


def in_control(r):
    """A node Mario controls (not a dialog, menu or landing), in any zone:
    after some stars the route goes straight back into a painting."""
    return not r["via"].startswith(MODAL) and r["action"] not in NOT_CONTROL


def soonest_control(run, k):
    """The soonest controllable castle node with more than k stars in run."""
    best = None
    for r in csv.DictReader(open(os.path.join(run, "nodes.csv"))):
        if int(r["stars"]) > k and in_control(r):
            f = int(r["frame"])
            if best is None or f < best[0]:
                best = (f, int(r["id"]))
    return best


def route_controls(run, end):
    """Per star count: the first controllable castle node of the route to
    node `end` after that star."""
    rows = {}
    for r in csv.DictReader(open(os.path.join(run, "nodes.csv"))):
        rows[r["id"]] = r
    path, at = [], str(end)
    while at != "-1":
        path.append(rows[at])
        at = rows[at]["parent"]
    path.reverse()
    out = {}
    for r in path:
        s = int(r["stars"])
        if s > 0 and s not in out and in_control(r):
            out[s] = (int(r["frame"]), int(r["id"]))
    out.setdefault(int(path[-1]["stars"]), (int(path[-1]["frame"]), int(path[-1]["id"])))
    return out
IDLE = b"\x00\x00\x00\x00" * 900

ENV = (
    "OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64 "
    "OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 "
    "OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_CHAIN=route OBRRT_SAMPLER=rate "
    "OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 "
    "OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10 OBRRT_BOOT_NOWAIT=1"
)


def samples(path):
    """Raw 4-byte samples of an .m64 (Mupen header dropped)."""
    b = open(path, "rb").read()
    return b[0x400:] if b[:4] == b"M64\x1a" else b


def nodepath(run, node, out):
    subprocess.run([NODEPATH, run, str(node), out], check=True, stdout=subprocess.DEVNULL)
    return samples(out)


def landing(inputs, k, work, name):
    """The first sample after which Mario is controllable with more than k
    stars when `inputs` (plus idle frames) are replayed (tasplay writes one
    row per sample); None if never."""
    m = os.path.join(work, name + ".raw")
    c = os.path.join(work, name + ".csv")
    open(m, "wb").write(inputs + IDLE)
    subprocess.run([TASPLAY, ROM, m, c], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for r in csv.DictReader(open(c)):
        if int(r["stars"]) > k and r["controllable"] == "1":
            return int(r["frame"])
    return None


def search(prefix, k, args, budget, attempt):
    """Fresh trees from `prefix`, one per seed, in parallel; returns
    (inputs, seed, node) candidates, soonest castle state of more stars."""
    start = os.path.join(args.work, f"start{k}.raw")
    open(start, "wb").write(prefix)
    procs = []
    for seed in args.seeds:
        out = os.path.join(args.work, f"leg{k}-s{seed}" + (f"-r{attempt}" if attempt else ""))
        env = dict(os.environ)
        for kv in ENV.split():
            a, b = kv.split("=", 1)
            env[a] = b
        env["OBRRT_MAX_FRAMES"] = str(budget)
        env["RAYON_NUM_THREADS"] = str(args.threads)
        log = open(out + ".log", "w")
        procs.append((seed, out, subprocess.Popen(["nice", "-n", "5", BIN, ROM, out, str(seed), "600", start], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
    cands = []
    for seed, out, p in procs:
        p.wait()
        try:
            best = soonest_control(out, k)
        except FileNotFoundError:
            best = None
        if best:
            cands.append((nodepath(out, best[1], out + "-best.m64"), seed, best[1]))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("route")
    ap.add_argument("landings")
    ap.add_argument("--budget", type=int, default=120_000_000)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--leg0", default=None, help="a run dir whose soonest 1-star castle state competes for leg 0")
    ap.add_argument("--from", dest="start", type=int, default=0, help="resume at leg K (needs <work>/leg<K-1>.m64)")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.work, exist_ok=True)
    end = int(open(args.landings).read().split()[0])
    ctl = route_controls(args.route, end)
    nodes = {s: n for s, (f, n) in ctl.items()}
    orig = {s: f for s, (f, n) in ctl.items()}
    print("route controls:", sorted(ctl.items()), flush=True)
    route_in = {s: nodepath(args.route, n, os.path.join(args.work, f"route{s}.m64")) for s, n in nodes.items()}
    log = open(os.path.join(args.work, "legs.txt"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    top = max(nodes)
    # The route's own legs, cut where it first has control after each star.
    full = route_in[top]
    cut = {0: 0}
    for s in range(1, top + 1):
        cut[s] = landing(full, s - 1, args.work, f"route-cut{s}") or len(full) // 4
    say("route control samples: " + ", ".join(f"{s}:{cut[s]}" for s in range(1, top + 1)))
    route_leg = {k: (full + IDLE)[4 * cut[k]:4 * cut[k + 1]] for k in range(top)}
    prefix = b"" if args.start == 0 else open(os.path.join(args.work, f"leg{args.start - 1}.m64"), "rb").read()
    for k in range(args.start, top):
        cands = []
        # The route's own leg from this prefix.
        cands.append(("route leg", prefix + route_leg[k]))
        if k == 0 and args.leg0:
            best = soonest_control(args.leg0, 0)
            if best:
                cands.append((f"leg0 {os.path.basename(args.leg0)} node {best[1]}", nodepath(args.leg0, best[1], os.path.join(args.work, "leg0-run.m64"))))
        judged = []

        def judge(cs, tag):
            for i, (name, inp) in enumerate(cs):
                f = landing(inp, k, args.work, f"judge{k}-{tag}{i}")
                judged.append((f if f is not None else 1 << 30, name, inp))
                say(f"leg {k}->{k + 1}: {name}: {'no landing' if f is None else f'control at {f}'}")

        judge(cands, "c")
        budget = args.budget
        for attempt in range(3 if k > 0 else 0):
            found = search(prefix, k, args, budget, attempt)
            judge([(f"search s{seed} node {node} ({budget // 1_000_000} M)", inp) for inp, seed, node in found], f"s{attempt}-")
            if any(j[0] < 1 << 30 for j in judged):
                break
            say(f"leg {k}->{k + 1}: nothing at {budget // 1_000_000} M; retrying with 3x")
            budget *= 3
        judged.sort(key=lambda x: x[0])
        f, name, inp = judged[0]
        if f >= 1 << 30:
            say(f"leg {k}->{k + 1}: no candidate reached {k + 1} stars; stopping")
            sys.exit(1)
        # Cut at control, so the next leg's boot starts right there.
        prefix = (inp + IDLE)[:4 * f]
        open(os.path.join(args.work, f"leg{k}.m64"), "wb").write(prefix)
        say(f"LEG {k}->{k + 1}: {name} at {f} (route {cut[k + 1]}, {cut[k + 1] - f:+d} vs route)")


if __name__ == "__main__":
    main()
