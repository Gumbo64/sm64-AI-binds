#!/usr/bin/env python3
"""Checks zone-share nodes by replay: for nodes a run's zone sharing added
(summary.json "zone_share" "recent_added"), writes each one's inputs from
power-on (nodepath) and replays them (tasplay), then compares the last frame
with the node's row in nodes.csv (level, area, stars, coins, position, frame).

  tools/zone_share_verify.py <run_dir> [--bin-dir builds/target-share/release] [--n 8] [ids...]

Run from objective-rrt/. Files go to <run_dir>/verify/.
"""
import argparse
import csv
import json
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("ids", nargs="*", type=int)
ap.add_argument("--bin-dir", default="builds/target-share/release")
ap.add_argument("--rom", default="../baserom.us.z64")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

ids = a.ids
if not ids:
    recent = json.load(open(f"{a.run}/summary.json"))["zone_share"]["recent_added"]
    step = max(1, len(recent) // a.n)
    ids = recent[::step][: a.n]
rows = {int(r["id"]): r for r in csv.DictReader(open(f"{a.run}/nodes.csv"))}
os.makedirs(f"{a.run}/verify", exist_ok=True)
bad = 0
for i in ids:
    m64 = f"{a.run}/verify/node-{i}.m64"
    out = f"{a.run}/verify/node-{i}.csv"
    subprocess.run([f"{a.bin_dir}/nodepath", a.run, str(i), m64], check=True, capture_output=True)
    subprocess.run(["nice", "-n", "10", f"{a.bin_dir}/tasplay", a.rom, m64, out], check=True, capture_output=True)
    last = list(csv.DictReader(open(out)))[-1]
    n = rows[i]
    checks = {
        "level": last["level"] == n["level"],
        "area": last["area"] == n["area"],
        "stars": last["stars"] == n["stars"],
        "coins": last["coins"] == n["coins"],
        "pos": all(abs(float(last[k]) - float(n[k])) <= 1.0 for k in "xyz"),
        "frame": int(last["frame"]) == int(n["frame"]),
    }
    ok = all(checks.values())
    bad += not ok
    print(f"node {i} (parent {n['parent']}, via {n['via']}): node L{n['level']}/{n['area']} ({n['x']},{n['y']},{n['z']}) f{n['frame']}; "
          f"replay L{last['level']}/{last['area']} ({last['x']},{last['y']},{last['z']}) f{last['frame']}: "
          + ("OK" if ok else "MISMATCH " + ",".join(k for k, v in checks.items() if not v)))
print(f"{len(ids) - bad}/{len(ids)} match")
raise SystemExit(1 if bad else 0)
