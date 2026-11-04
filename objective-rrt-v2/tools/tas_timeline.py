#!/usr/bin/env python3
"""Action timeline of a tasplay dump between two frames: one line per action
change (frame, action, position, speeds, held, Chain Chomp gate-local coords
in BoB): tools/tas_timeline.py <dump.csv> <from> <to>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tas_analyse as T

rows = T.load(sys.argv[1])
lo, hi = int(sys.argv[2]), int(sys.argv[3])
prev = None
for r in rows[lo - 1:hi]:
    name = T.act_name(r["act"])
    if name != prev:
        g = T.gate_local(r["pos"]) if T.lname(r["level"]) == "BoB" else None
        print(f"{r['f']:6} {T.lname(r['level'])}/{r['area']} {name:24} pos {T.fmt(r['pos']):22} fvel {r['fvel']:7.1f} "
              f"hs {r['hs']:6.1f} vy {r['vy']:6.1f} held {int(r['held'])} st {r['stars']} hp {r['health'] >> 8}"
              + (f" gate x {g[0]:5.0f} z {g[2]:5.0f}" if g else ""))
    prev = name
