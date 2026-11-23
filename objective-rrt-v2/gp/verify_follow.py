#!/usr/bin/env python3
"""Independent check of follow plays: verify_follow.py <tasplay> <rom> <index> [limit].
Replays each dump with tasplay and applies the hit rule (in Python, from the CSV) to its .wp."""
import csv, subprocess, sys, tempfile, random
tas, rom, index = sys.argv[1:4]
lim = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
def seg2(a, b, p):
    d = [b[i] - a[i] for i in range(3)]
    l = sum(x * x for x in d)
    u = max(0.0, min(1.0, sum((p[i] - a[i]) * d[i] for i in range(3)) / l)) if l > 0 else 0.0
    return sum((a[i] + u * d[i] - p[i]) ** 2 for i in range(3))
lines = open(index).read().split("\n")[:-1]
random.Random(1).shuffle(lines)
ok = bad = 0
for line in lines[:lim]:
    path, frame, done, played = line.split()
    frame, done = int(frame), int(done)
    way = [tuple(map(float, l.split())) for l in open(path + ".wp")]
    with tempfile.NamedTemporaryFile(suffix=".csv") as t:
        subprocess.run([tas, rom, path, t.name], check=True, stderr=subprocess.DEVNULL)
        rows = [r for r in csv.DictReader(open(t.name)) if int(r["frame"]) >= frame]
    i, got, prev = 0, -1, None
    for r in rows:
        p = (float(r["x"]), float(r["y"]), float(r["z"]))
        if prev is None:
            prev = p
            continue
        z = (float(r["level"]), float(r["area"]))
        live = r["controllable"] == "1" and r["dialog"] == "-1"
        adv = None
        if live and z != way[i][3:5]:
            adv = next((k for k in range(i, len(way)) if way[k][3:5] == z), None)
        else:
            for k in range(min(i + 5, len(way)) - 1, i - 1, -1):
                if way[k][3:5] == z and seg2(prev, p, way[k][:3]) <= 100.0 ** 2:
                    adv = k + 1
                    break
        prev = p
        if adv is not None:
            if adv >= len(way):
                got = int(r["frame"]) - frame
                break
            i = adv
    same = got == done
    ok += same; bad += not same
    if not same:
        print(path.rsplit("/", 1)[-1], "claimed", done, "replay", got)
print(f"replay agrees on {ok}/{ok + bad}")
