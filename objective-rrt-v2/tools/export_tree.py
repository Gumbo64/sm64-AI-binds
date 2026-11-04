#!/usr/bin/env python3
"""Exports finished runs for the static tree view (GitHub Pages):

    tools/export_tree.py <run> [<run> ...] [--out DIR]

DIR defaults to sm64-viewer/frontend/public/trees. Writes, per run:
  <run>/nodes.csv, <run>/summary.json, <run>/tree.bin   (replays are rebuilt
                                                         in the browser)
  <run>/info/<k>.json   states k*1000..k*1000+999: the objective that made
                        each and the objectives tried from it (from events.jsonl)
and, shared: levels.json and index.json (the exported runs, newest first).
Level geometry is not exported: the viewer reads it from the visitor's ROM.
"""

import csv
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viz  # noqa: E402  (geometry and level names)

RUNS = os.path.join(HERE, "..", "runs")
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "../../../sm64-viewer/frontend/public/trees"))
CHUNK = 1000
KEEP_ADDED = 6


def export(run, out):
    src = os.path.join(RUNS, run)
    dst = os.path.join(out, run)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, "info"))
    for f in ("nodes.csv", "summary.json", "tree.bin"):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))

    made, tried = {}, {}
    for line in open(os.path.join(src, "events.jsonl")):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        frm = e.get("from", e["start"] if e.get("stage", 0) == 0 else None)
        ev = {k: e.get(k) for k in ("kind", "key", "success", "chain", "stage", "len")}
        if e.get("added") is not None:
            made[e["added"]] = dict(ev, start=frm)
        if frm is not None:
            # Grouped by objective: the panel shows tries, successes and up
            # to KEEP_ADDED new states per objective.
            g = tried.setdefault(frm, {}).setdefault(e["key"], {"kind": e["kind"], "key": e["key"], "n": 0, "ok": 0, "added": []})
            g["n"] += 1
            g["ok"] += bool(e["success"])
            if e.get("added") is not None and len(g["added"]) < KEEP_ADDED:
                g["added"].append(e["added"])

    nodes = list(csv.DictReader(open(os.path.join(src, "nodes.csv"))))
    chunks = {}
    for n in nodes:
        i = int(n["id"])
        chunks.setdefault(i // CHUNK, {})[i] = {"made": made.get(i), "tried": list(tried.get(i, {}).values())}
    for k, v in chunks.items():
        with open(os.path.join(dst, "info", f"{k}.json"), "w") as f:
            json.dump(v, f, separators=(",", ":"))

    # No level geometry: that is Nintendo's data. The viewer reads it from
    # the visitor's own ROM (src/tree/romGeometry.js).
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dst) for f in fs)
    print(f"{run}: {len(nodes)} states, {size / 1e6:.1f} MB -> {dst}")


def main():
    args = sys.argv[1:]
    out = DEFAULT_OUT
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i : i + 2]
    if not args:
        sys.exit(__doc__)
    os.makedirs(out, exist_ok=True)
    for run in args:
        export(run, out)
    with open(os.path.join(out, "levels.json"), "w") as f:
        json.dump(viz.levels(), f)
    runs = []
    for d in os.listdir(out):
        p = os.path.join(out, d, "nodes.csv")
        if os.path.exists(p):
            runs.append({"name": d, "mtime": os.path.getmtime(os.path.join(RUNS, d, "nodes.csv")) if os.path.exists(os.path.join(RUNS, d, "nodes.csv")) else os.path.getmtime(p), "live": False})
    runs.sort(key=lambda r: -r["mtime"])
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(runs, f, indent=1)


if __name__ == "__main__":
    main()
