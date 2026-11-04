#!/usr/bin/env python3
"""Writes runs/fixture/: a small hand-made run for developing the GUI
(sm64-viewer ?oss=fixture) without running a search.

    tools/make_fixture.py

objectives.json, cells.jsonl and tree.bin follow the formats the search
writes. The cells are made up; the replays are boot inputs plus random
walking, so they play but don't reach the cells' states.
"""

import json
import os
import random
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "fixture")

A, START = 0x8000, 0x1000
ACT = {"idle": 0x0C400201, "walk": 0x04000440, "jump": 0x03000880, "long_jump": 0x03000888, "freefall": 0x0100088C}


def boot():
    """sm64_ai::util::skip_menu_pads: 1680 frames from power-on to Mario."""
    pads = []
    for i in range(1680):
        b = A if i % 2 == 0 else START if 150 <= i < 160 or 200 <= i < 300 else 0
        pads.append((b, 0, 0))
    return pads


def walk(rng, n):
    pads, x, y = [], 0, 80
    for _ in range(n):
        if rng.random() < 0.05:
            x, y = rng.randint(-80, 80), rng.randint(-80, 80)
        pads.append((A if rng.random() < 0.03 else 0, x, y))
    return pads


def pos(x, y, z, grid_log2):
    """The 10-bit packed position of a cell key."""
    q = [round(c / (1 << grid_log2)) & 0x3FF for c in (x, y, z)]
    return (q[0] << 20) | (q[1] << 10) | q[2]


OBJECTIVES = [
    {"id": 0, "kind": "Root", "name": "root", "doc": "Progress and a coarse position everywhere.",
     "params": {"grid_log2": 10}, "children": [1, 2], "fields": ["level", "keys", "pos"]},
    {"id": 1, "kind": "BljLens", "name": "LobbyBLJ", "doc": "BLJ up the lobby stairs: a fine lens near the steps, then a payoff while fast.",
     "params": {"center": [-2230, 0, 303], "radius": 1500, "grid_log2": 6, "hspeed_step": 4, "yaw_sectors": 16},
     "children": [], "fields": ["pos", "hspeed", "yaw", "action", "payoff"]},
    {"id": 2, "kind": "Region", "name": "LakituSkip", "doc": "The castle grounds bridge strip.",
     "params": {"min": [-700, 800, -2000], "max": [700, 3000, -177], "grid_log2": 7}, "children": [], "fields": ["pos"]},
]


def cells(rng):
    out = []

    def add(key, node, frames):
        visits = rng.randint(1, 60)
        succ = rng.randint(0, visits // 2)
        a, b = 1 + succ, 1 + visits - succ
        out.append({"id": len(out), "key": key, "node": node, "frames": frames,
                    "visits": visits, "succ": succ, "p": round(a / (a + b), 3)})

    grounds, lobby = 16 * 256 + 1, 6 * 256 + 1
    # Castle grounds, some on the bridge.
    for i, (x, y, z) in enumerate([(-1300, 260, 4400), (-300, 260, 2500), (0, 900, -1200), (500, 900, -600), (-400, 1000, -1800)]):
        key = [[0, 0, grounds], [0, 1, 0], [0, 2, pos(x, y, z, 10)]]
        if y > 800:
            key.append([2, 0, pos(x, y, z, 7)])
        add(key, 2 + i % 2, 1700 + 120 * i)
    # The lobby: near the steps the BLJ lens, away from them the payoff when fast.
    for i in range(10):
        x, z = -2230 + rng.randint(-600, 600), 303 + rng.randint(-600, 600)
        y = rng.choice([0, 0, 100, 250])
        hs = rng.choice([-4, -3, -2, -1, 0, 2, 4, 7])
        act = rng.choice(["walk", "long_jump", "long_jump", "jump", "idle"])
        key = [[0, 0, lobby], [0, 1, rng.choice([0, 0, 1])], [0, 2, pos(x, y, z, 10)],
               [1, 0, pos(x, y, z, 6)], [1, 1, hs], [1, 2, rng.randrange(16)], [1, 3, ACT[act]]]
        add(key, 4 + i % 2, 2600 + 90 * i)
    for i in range(5):
        x, y, z = -3500 - 300 * i, 600 + 300 * i, -2000 - 400 * i
        key = [[0, 0, lobby], [0, 1, 0], [0, 2, pos(x, y, z, 10)], [1, 4, 1], [1, 1, -10 - 5 * i]]
        add(key, 5, 3500 + 60 * i)
    return out


def main():
    rng = random.Random(64)
    os.makedirs(OUT, exist_ok=True)
    # tree.bin: (parent, inputs) per record; record 0 is power-on (no inputs)
    # and record 1 the menu skip.
    records = [(-1, []), (0, boot()), (1, walk(rng, 400)), (1, walk(rng, 600)), (2, walk(rng, 700)), (4, walk(rng, 500))]
    with open(os.path.join(OUT, "tree.bin"), "wb") as f:
        f.write(struct.pack("<I", len(records)))
        for parent, pads in records:
            f.write(struct.pack("<iI", parent, len(pads)))
            for b, x, y in pads:
                f.write(struct.pack(">Hbb", b, x, y))
    with open(os.path.join(OUT, "objectives.json"), "w") as f:
        json.dump(OBJECTIVES, f, indent=1)
    with open(os.path.join(OUT, "cells.jsonl"), "w") as f:
        for c in cells(rng):
            f.write(json.dumps(c) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
