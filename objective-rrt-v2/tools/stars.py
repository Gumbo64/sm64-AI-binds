#!/usr/bin/env python3
"""Every star any run collected: tools/stars.py [run ...] (default: all runs).

A star is a state whose star count is above its parent's; it is named by the
objective that made it (events.jsonl key) and the level it was in. Prints,
per star, the runs that got it and the earliest game time from power-on.
"""
import csv, json, os, sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
LEVELS = {9: "BoB", 24: "WF", 12: "JRB", 5: "CCM", 4: "BBH", 7: "HMC", 22: "LLL", 8: "SSL", 23: "DDD", 10: "SL",
          11: "WDW", 36: "TTM", 13: "THI", 14: "TTC", 15: "RR", 27: "PSS", 6: "Castle", 16: "Grounds", 26: "Courtyard",
          29: "TotWC", 28: "CotMC", 18: "VCutM", 31: "WMotR", 20: "SA", 17: "BitDW", 19: "BitFS", 21: "BitS"}
# Placed star positions (BoB), for naming; others by level.
BOB = {(1550, 1200, 300): "Behind Chain Chomp's Gate", (3304, 4242, -4603): "Big Bob-omb on the Summit"}


def clock(f):
    return f"{int(f / 30 // 60)}:{f / 30 % 60:04.1f}"


def main():
    runs = sys.argv[1:] or sorted(d for d in os.listdir(ROOT) if os.path.exists(os.path.join(ROOT, d, "nodes.csv")))
    stars = defaultdict(list)
    for run in runs:
        d = os.path.join(ROOT, run)
        nodes = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(d, "nodes.csv")))}
        made = {}
        for line in open(os.path.join(d, "events.jsonl")):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("added") is not None:
                made[e["added"]] = e
        best = {}
        for i, n in nodes.items():
            p = nodes.get(int(n["parent"]))
            if not p or int(n["stars"]) <= int(p["stars"]):
                continue
            # The level the star was collected in: the parent's (the state
            # itself may be back in the castle after the star's exit).
            e = made.get(i, {})
            key = e.get("key", n["via"])
            where = LEVELS.get(int(p["level"]), f"level {p['level']}")
            name = key
            if "bhvStar" in key or "Star (" in key:
                name = f"{where}: {key.split(' ', 2)[-1]}"
            if e.get("kind") == "Boss::Hit" or "King" in key:
                name = f"{where}: King Bob-omb"
            label = f"{where} | {e.get('kind', n['via'])} | {name}"
            f = int(n["frame"])
            if label not in best or f < best[label]:
                best[label] = f
        for label, f in best.items():
            stars[label].append((f, run))
    for label, got in sorted(stars.items(), key=lambda kv: min(kv[1])[0]):
        got.sort()
        print(f"{label}\n    first {clock(got[0][0])} ({got[0][1]}); {len(got)} runs: {', '.join(r for _, r in got)}")


if __name__ == "__main__":
    main()
