#!/usr/bin/env python3
"""What a run spends on the castle grounds before it first enters the castle.

    tools/analysis/grounds_opening.py runs/<run> [runs/<run> ...] [--top N]

For each run: the first node in the castle (level 6) by creation order, its
game frame (castle entry time), the soonest 1-star node's frame (first star
time), and for every attempt (events.jsonl) made before that node was added
from a castle-grounds (level 16) node: attempts, frames spent, successes per
objective kind, with summary.json's entry_weights mean_decay (the rate
sampler's per-kind multiplier) where present.
"""
import argparse, collections, csv, json, os

LEVEL_CASTLE, LEVEL_GROUNDS = 6, 16


def load_nodes(run):
    nodes = {}
    with open(os.path.join(run, "nodes.csv")) as f:
        for r in csv.DictReader(f):
            nodes[int(r["id"])] = (int(r["level"]), int(r["frame"]), int(r["stars"]), r["via"])
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()
    for run in a.runs:
        nodes = load_nodes(run)
        castle = [i for i, n in nodes.items() if n[0] == LEVEL_CASTLE]
        first_castle = min(castle) if castle else None
        entry_frame = min((nodes[i][1] for i in castle), default=None)
        star = min((n[1] for n in nodes.values() if n[2] >= 1), default=None)
        try:
            summ = json.load(open(os.path.join(run, "summary.json")))
        except Exception:
            summ = {}
        ew = summ.get("entry_weights", {})
        per = collections.defaultdict(lambda: [0, 0, 0, 0])
        total = 0
        n_ev = 0
        with open(os.path.join(run, "events.jsonl")) as f:
            for line in f:
                e = json.loads(line)
                if first_castle is not None and e.get("added") is not None and e["added"] >= first_castle:
                    break
                n_ev += 1
                fr = nodes.get(e["from"])
                if not fr or fr[0] != LEVEL_GROUNDS:
                    continue
                p = per[e["kind"]]
                p[0] += 1
                p[1] += e["frames"]
                p[2] += bool(e["success"])
                p[3] += e.get("added") is not None
                total += e["frames"]
        fmt = lambda f: "-" if f is None else f"f{f} ({(f - 1646) / 30:.1f}s from control)"
        print(f"== {run}: first castle node #{first_castle} after {n_ev} events; "
              f"castle entry {fmt(entry_frame)}; first star {fmt(star)}; grounds frames before entry {total / 1e6:.2f}M")
        print(f"  {'kind':32} {'att':>6} {'Mframes':>8} {'share':>6} {'succ':>5} {'added':>6} {'decay':>6}")
        for k, p in sorted(per.items(), key=lambda kv: -kv[1][1])[: a.top]:
            d = ew.get(k, {}).get("mean_decay")
            print(f"  {k:32} {p[0]:6} {p[1] / 1e6:8.2f} {p[1] / max(total, 1):6.1%} {p[2]:5} {p[3]:6} "
                  f"{'' if d is None else f'{d:6.2f}'}")


if __name__ == "__main__":
    main()
