#!/usr/bin/env python3
"""Level-entry start states for per-level tree searches (the 120-star plan). The TAS is used only to REACH
each level's entry funnel (Mario's state there is the same for any route) and for nothing after it: a start
movie is the TAS up to one frame before control, plus one neutral input, so nothing is held at the start.
Writes logs/seg120/starts/<n>-<level>-act<a>.m64 and logs/seg120/visits.csv (one row per level visit)."""
import csv, struct
TAS = 'ref/tas/full-120star-2026-U.m64'
HUB = {6, 16, 26}
LV = {9:'BoB',24:'WF',12:'JRB',5:'CCM',4:'BBH',7:'HMC',22:'LLL',8:'SSL',23:'DDD',10:'SL',11:'WDW',36:'TTM',13:'THI',14:'TTC',15:'RR',27:'PSS',29:'TotWC',28:'CotMC',18:'VCutM',31:'WMotR',20:'SA',17:'BitDW',19:'BitFS',21:'BitS',30:'Bowser1',33:'Bowser2',34:'Bowser3'}
R = list(csv.DictReader(open('logs/funnel/census-tas.csv')))
act = {}
for r in csv.DictReader(open('ref/tas/dumps/full-120star-2026-U.csv')): act[int(r['frame'])] = r['act']
raw = open(TAS, 'rb').read(); head, body = raw[:0x400], raw[0x400:]
visits = []; cur = None
for r in R:
    lvl = int((r['from'] if r['zone'] == 'star' else r['zone']).split(':')[0])
    if r['zone'] == 'star':
        if cur and lvl not in HUB: cur['stars'].append((r['course'], r['star'], int(r['frame']), r['x'], r['y'], r['z']))
        elif lvl in HUB: visits.append(dict(level=lvl, frame=int(prev['frame']), stars=[(r['course'], r['star'], int(r['frame']), r['x'], r['y'], r['z'])], hub=True, sset=prev['star_set'], nstars=prev['stars']))
        continue
    prev = r
    if lvl in HUB: cur = None
    elif cur is None:
        cur = dict(level=lvl, frame=int(r['frame']), stars=[], hub=False, sset=r['star_set'], nstars=r['stars']); visits.append(cur)
w = csv.writer(open('logs/seg120/visits.csv', 'w')); w.writerow(['n', 'level', 'name', 'act', 'entry', 'had', 'hub', 'stars', 'start'])
for n, v in enumerate(visits):
    f = v['frame']; a = act.get(f, '?'); name = LV.get(v['level'], str(v['level']))
    p = f"logs/seg120/starts/{n:03d}-{name}-act{a}.m64"
    b = body[:4 * (f - 1)] + b'\x00\x00\x00\x00'
    open(p, 'wb').write(b)  # raw, 4 bytes a frame from power-on (obrrt2 and legs read raw)
    w.writerow([n, v['level'], name, a, f, v['nstars'], int(v['hub']), ' '.join(f"{c}:{s}@{fr - f}" for c, s, fr, *_ in v['stars']), p])
print(len(visits), 'visits;', sum(len(v['stars']) for v in visits), 'stars;', sum(v['hub'] for v in visits), 'hub stars')
