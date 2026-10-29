#!/usr/bin/env python3
"""Lakitu-skip summary of obrrt runs: per run, the first node of each
Tech::LakituSkip key (frame, position), the earliest castle-entry node
(level 6) and whether its ancestry went through the intro dialog
(a Lakitu::Intro / Dialog node) or a Tech::LakituSkip node.

  tools/analysis/lakitu_runs.py runs/<a> [runs/<b> ...]
"""
import csv, json, os, sys

def main(dirs):
    for d in dirs:
        rows = {}
        with open(f"{d}/nodes.csv") as f:
            for r in csv.DictReader(f):
                rows[int(r["id"])] = r
        print(f"== {d}: {len(rows)} nodes")
        # Per key, from events.jsonl: attempts, successes, first success.
        keyof, stats = {}, {}
        if os.path.exists(f"{d}/events.jsonl"):
            with open(f"{d}/events.jsonl") as f:
                for line in f:
                    e = json.loads(line)
                    k = e.get("key", "")
                    if not k.startswith("Tech::LakituSkip"):
                        continue
                    st = stats.setdefault(k, [0, 0, None])
                    st[0] += 1
                    if e.get("success"):
                        st[1] += 1
                        a = e.get("added")
                        if st[2] is None and a is not None and a in rows:
                            st[2] = rows[a]
                    if e.get("added") is not None:
                        keyof[e["added"]] = k
        for k, (n, ok, r) in sorted(stats.items()):
            first = f"first success node {r['id']} frame {r['frame']} at ({r['x']},{r['y']},{r['z']})" if r else ""
            print(f"  {k:24} {n} attempts, {ok} successes {first}")
        firsts = {}
        for i in sorted(rows):
            v = rows[i]["via"]
            if v.startswith("Tech::LakituSkip") and v not in firsts:
                firsts[v] = rows[i]
        for k, r in sorted(firsts.items()):
            n = sum(1 for x in rows.values() if x["via"] == k)
            print(f"  {k:24} first node {r['id']} frame {r['frame']} at ({r['x']},{r['y']},{r['z']}); {n} nodes")
        castle = [r for r in rows.values() if r["level"] == "6"]
        if not castle:
            print("  castle: never entered")
            continue
        best = min(castle, key=lambda r: int(r["frame"]))
        chain, cur = [], best
        while cur is not None:
            chain.append(keyof.get(int(cur["id"]), cur["via"]))
            p = int(cur["parent"])
            cur = rows.get(p)
        skip = [v for v in chain if v.startswith("Tech::LakituSkip")]
        intro = [v for v in chain if "Lakitu" in v and not v.startswith("Tech::") or v.startswith("Dialog")]
        f = int(best["frame"])
        print(f"  castle entry: node {best['id']} frame {f} ({f / 30:.1f} s); via skip {skip or '-'}; dialog/intro {intro or '-'}")

if __name__ == "__main__":
    main(sys.argv[1:])
