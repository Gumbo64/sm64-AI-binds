#!/usr/bin/env python3
"""Plays a finished route through the ending: idle frames after the grand star until the credits are over and the
"THE END" screen holds, checked by replay (builds/tools/tasplay).

    tools/seg120/credits.py ROUTE.m64 OUT.m64 [--hold 600]

ROUTE ends just after the grand star (the final search's exit into the castle grounds, level 16). The ending needs
no input: Peach's scene in the grounds, then the credits (about 222 frames in each of 21 course scenes), then the
grounds again, where the state stops changing for good (the reference movie, used only to learn this sequence:
frames 129550 grounds, 132781-137224 credits, still from 137583 on). The output is ROUTE + idle frames up to
`--hold` frames after the last change of (level, area, play mode). Exit status 0 only if the replay shows the
credits (at least 15 scenes) and then the still screen.
"""
import argparse, csv, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('route'); ap.add_argument('out')
ap.add_argument('--hold', type=int, default=600, help='frames kept after the ending stops changing')
ap.add_argument('--max', type=int, default=20000, help='idle frames tried at most')
A = ap.parse_args()


def body(p):
    b = Path(p).read_bytes()
    return b[0x400:] if b[:4] == b'M64\x1a' else b


def main():
    route = body(A.route); n0 = len(route) // 4
    with tempfile.TemporaryDirectory() as tmp:
        movie, out = Path(tmp) / 'm.m64', Path(tmp) / 'p.csv'
        movie.write_bytes(route + b'\0\0\0\0' * A.max)
        subprocess.run(['nice', '-n', '10', str(HERE / 'builds/tools/tasplay'), str(HERE.parent / 'baserom.us.z64'), str(movie), str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows = list(csv.DictReader(open(out)))
    seq, prev = [], None
    for x in rows[n0 - 1:]:
        k = (int(x['level']), int(x['area']), int(x['play_mode']))
        if k != prev: seq.append((int(x['frame']), k)); prev = k
    scenes = len({k[:2] for f, k in seq if k[0] not in (16, 34, 21, 6)})
    last = seq[-1][0] if seq else n0
    end = last + A.hold
    print(f'route {n0} frames; ending: {len(seq)} changes, {scenes} credit scenes, still from frame {last}; output {end} frames')
    if scenes < 15 or end > len(rows) - 1:
        print('the replay does not show the whole ending: not written', file=sys.stderr); return 1
    Path(A.out).write_bytes(route + b'\0\0\0\0' * (end - n0))
    return 0


sys.exit(main())
