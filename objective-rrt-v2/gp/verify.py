#!/usr/bin/env python3
"""Independent replay check: verify.py <tasplay> <rom> <index file> <kind x y z r>
Replays each dump in <index> (raw inputs from power-on) with tasplay, and finds
the first frame after the start that meets the goal: `ramp` (xz within 400 of the
setup point, not airborne, controllable) or `door` (3D within 350 of the door or
inside the castle, level 16 until then). Compares with what gp claimed."""
import csv, subprocess, sys, os, math, tempfile
tas, rom, index, kind = sys.argv[1:5]
ok = bad = 0
for line in open(index):
    path, frame, done, played = line.split()
    frame, done = int(frame), int(done)
    with tempfile.NamedTemporaryFile(suffix=".csv") as t:
        subprocess.run([tas, rom, path, t.name], check=True, stderr=subprocess.DEVNULL)
        got = -1
        for r in csv.DictReader(open(t.name)):
            f = int(r["frame"])
            if f <= frame:
                continue
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            act = int(r["action"], 16)
            if kind == "ramp":
                hit = math.hypot(x - 1600, z - 4550) <= 400 and not act & 0x800 and r["controllable"] == "1"
            else:
                hit = r["level"] == "6" or (r["level"] == "16" and math.dist((x, y, z), (0, 803, -3155)) < 350)
            if hit:
                got = f - frame
                break
    same = got == done
    ok += same
    bad += not same
    if not same or "-v" in sys.argv:
        print(os.path.basename(path), "claimed", done, "replay", got)
print(f"replay agrees on {ok}/{ok + bad}")
