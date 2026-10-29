#!/usr/bin/env python3
"""Soonest course entries in a run's tree: for each (level, stars) first
reached by a node outside the castle, the earliest game frame and the
node; and the soonest frame of each star count. For runs started from a
post-star castle state (tools/painting_fast_reentry.sh).

usage: first_entries.py runs/<run> [...]
"""
import csv
import sys

CASTLE = {6, 16, 26}
NAMES = {9: 'bob', 24: 'wf', 12: 'jrb', 5: 'ccm', 27: 'pss', 4: 'bbh', 7: 'hmc', 22: 'lll', 8: 'ssl', 23: 'ddd',
         10: 'sl', 11: 'wdw', 36: 'ttm', 13: 'thi', 14: 'ttc', 15: 'rr', 29: 'totwc', 18: 'cotmc', 20: 'vcutm', 17: 'bitdw'}

for run in sys.argv[1:]:
    rows = list(csv.DictReader(open(f'{run}/nodes.csv')))
    root = rows[0]
    f0 = int(root['frame'])
    first = {}
    stars = {}
    for x in rows:
        f, lv, st = int(x['frame']), int(x['level']), int(x['stars'])
        if lv not in CASTLE:
            k = (NAMES.get(lv, lv), st)
            if k not in first or f < first[k][0]:
                first[k] = (f, x['id'])
        if st not in stars or f < stars[st][0]:
            stars[st] = (f, x['id'])
    print(f'# {run}: {len(rows)} nodes, root frame {f0} stars {root["stars"]}')
    print('  entries: ' + ', '.join(f'{k[0]}@{k[1]}st +{v[0] - f0}f (n{v[1]})' for k, v in sorted(first.items(), key=lambda kv: kv[1][0])))
    print('  stars:   ' + ', '.join(f'{k}: +{v[0] - f0}f (n{v[1]})' for k, v in sorted(stars.items())))
