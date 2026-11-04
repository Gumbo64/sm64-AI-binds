#!/usr/bin/env python3
"""Merge the BLJ spot registries of several runs into one catalogue.

    python3 tools/analysis/blj_spots.py RUN_DIR [RUN_DIR ...] [-o ref/blj_spots_catalog.json]

Each run dir holds blj_spots.jsonl (OBRRT_BLJ_SPOTS=1, objectives/src/extra/
tech/bljany/spots.rs). Spots are deduplicated with the registry's key and
tolerance: zone, owner ("static" or the object's behaviour@home), clause,
the start on a 64-unit grid (object frame for object spots), facing octant;
near-duplicates (same zone, owner and clause, start within 100, facing
within 0x1000) merge into the first. Each catalogue spot lists the runs it
came from ("runs": the run dir names, plus "preset:<name>" for presets) and
sums the counts. Load it in a run with OBRRT_BLJ_SPOTS_CATALOG=<file>
(source "catalog").
"""
import argparse
import json
import math
import os

MERGE_DIST = 100.0
MERGE_YAW = 0x1000


def key(s):
    g = tuple(math.floor(v / 64.0) for v in s["local"])
    octant = ((s["yaw"] + 0x1000) % 0x10000) // 0x2000
    return (tuple(s["zone"]), s["owner"], s["clause"], g, octant)


def yaw_diff(a, b):
    d = (a - b) & 0xFFFF
    return min(d, 0x10000 - d)


def merge(dirs):
    spots, index, merges = [], {}, 0
    for d in dirs:
        path = os.path.join(d, "blj_spots.jsonl")
        if not os.path.exists(path):
            print(f"{path}: missing, skipped")
            continue
        run = os.path.basename(os.path.normpath(d))
        for line in open(path):
            s = json.loads(line)
            k = key(s)
            i = index.get(k)
            if i is None:
                for j, t in enumerate(spots):
                    if (t["zone"], t["owner"], t["clause"]) == (s["zone"], s["owner"], s["clause"]) and math.dist(t["local"], s["local"]) <= MERGE_DIST and yaw_diff(t["yaw"], s["yaw"]) <= MERGE_YAW:
                        i = j
                        merges += 1
                        break
            if i is None:
                spots.append({"zone": s["zone"], "owner": s["owner"], "clause": s["clause"], "local": s["local"], "yaw": s["yaw"], "world": s["world"],
                              "sources": [], "runs": [], "seen": 0, "attempted": 0, "succeeded": 0, "observed": 0, "best_fv": 0.0})
                i = len(spots) - 1
            index[k] = i
            t = spots[i]
            t["sources"] = sorted(set(t["sources"]) | set(s["sources"]) - {"catalog"})
            for r in [run] + [r for r in s.get("runs", []) if r.startswith("preset:")]:
                if r not in t["runs"]:
                    t["runs"].append(r)
            for c in ("seen", "attempted", "succeeded", "observed"):
                t[c] += s.get(c, 0)
            t["best_fv"] = max(t["best_fv"], s.get("best_fv", 0.0))
    return spots, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", default="ref/blj_spots_catalog.json")
    a = ap.parse_args()
    spots, merges = merge(a.runs)
    doc = ("BLJ spot catalogue (tools/analysis/blj_spots.py): spots from the runs' blj_spots.jsonl, "
           "deduplicated (zone, owner, clause, 64-grid start, facing octant; near-duplicates within 100 / 0x1000 merged). "
           "'runs' lists the run dirs each spot came from.")
    json.dump({"doc": doc, "runs": a.runs, "merges": merges, "spots": spots}, open(a.out, "w"), indent=1)
    by = {}
    for s in spots:
        for src in s["sources"]:
            by[(src, s["clause"])] = by.get((src, s["clause"]), 0) + 1
    print(f"{len(spots)} spots ({merges} near-duplicate merges) -> {a.out}")
    for (src, c), n in sorted(by.items()):
        print(f"  {src:9s} {c:9s} {n}")


if __name__ == "__main__":
    main()
