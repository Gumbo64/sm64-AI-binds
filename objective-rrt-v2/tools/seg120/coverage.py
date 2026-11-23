#!/usr/bin/env python3
"""Which stars did the per-zone trees find? tools/seg120/coverage.py PREFIX
Per run: every star the tree collected that its start state didn't have: (course, star), the earliest frame
after the level entry, and the node. Then per course: found / 7."""
import csv, glob, os, sys, collections
csv.field_size_limit(10**7)
PREFIX = sys.argv[1]
COURSE = {1:'BoB',2:'WF',3:'JRB',4:'CCM',5:'BBH',6:'HMC',7:'LLL',8:'SSL',9:'DDD',10:'SL',11:'WDW',12:'TTM',13:'THI',14:'TTC',15:'RR',16:'BitDW',17:'BitFS',18:'BitS',19:'PSS',20:'CotMC',21:'TotWC',22:'VCutM',23:'WMotR',24:'SA',25:'end',26:'Castle'}
LEVEL = {9:'BoB',24:'WF',12:'JRB',5:'CCM',4:'BBH',7:'HMC',22:'LLL',8:'SSL',23:'DDD',10:'SL',11:'WDW',36:'TTM',13:'THI',14:'TTC',15:'RR',27:'PSS',6:'Castle',16:'Grounds',26:'Courtyard',29:'TotWC',28:'CotMC',18:'VCutM',31:'WMotR',20:'SA',17:'BitDW',19:'BitFS',21:'BitS',30:'Bowser1',33:'Bowser2',34:'Bowser3'}
def flags(s):
    o = {}
    for p in s.split():
        c, m = p.split(':'); o[int(c)] = int(m, 16) & 0x7f
    return o
V = {v['n']: v for v in csv.DictReader(open('logs/seg120/visits.csv'))}
tot = collections.defaultdict(dict)
for d in sorted(glob.glob(f'runs/{PREFIX}-*')):
    if not os.path.exists(d + '/nodes.csv'): continue
    n = os.path.basename(d).split('-')[1]; v = V[str(int(n))]
    rows = list(csv.DictReader(open(d + '/nodes.csv'))); root = flags(rows[0]['star_flags']); f0 = int(rows[0]['frame'])
    best = {}
    for r in rows[1:]:
        fl = flags(r['star_flags'])
        for c, m in fl.items():
            new = m & ~root.get(c, 0)
            for i in range(7):
                if new >> i & 1 and ((c, i + 1) not in best or int(r['frame']) < best[(c, i + 1)][0]): best[(c, i + 1)] = (int(r['frame']), r['id'], r['via'])
    # Other zones the tree got into (level:area and act), earliest first, and keys.
    zones = {}
    for r in rows[1:]:
        z = (r['level'], r['area'], r['act'] if int(r['level']) not in (6, 16, 26) else '-')
        if z[:2] != (rows[0]['level'], rows[0]['area']) and (z not in zones or int(r['frame']) < zones[z]): zones[z] = int(r['frame'])
    keys = max(int(r['keys']) for r in rows) - int(rows[0]['keys'])
    want = v['stars'].split()
    got = ' '.join(f"{COURSE.get(c, c)}:{s}@{f - f0}" for (c, s), (f, _, _) in sorted(best.items(), key=lambda kv: kv[1][0]))
    print(f"{os.path.basename(d):28} nodes {len(rows):6}  TAS here: {' '.join(want):24} found: {got or '-'}")
    if zones or keys: print('      zones entered:', ' '.join(f"{LEVEL.get(int(l), l)}:{ar}{'/act' + ac if ac != '-' else ''}@{f - f0}" for (l, ar, ac), f in sorted(zones.items(), key=lambda kv: kv[1])[:14]), f'| keys +{keys}' if keys else '')
    for (c, s), (f, node, via) in best.items():
        if (c, s) not in tot[c] or f - f0 < tot[c][(c, s)][0]: tot[c][(c, s)] = (f - f0, os.path.basename(d), node)
print()
for c in sorted(tot): print(f"{COURSE.get(c, c):6} {len(tot[c])} stars:", ' '.join(f"{s}@{v[0]}" for (_, s), v in sorted(tot[c].items())))
print('total distinct stars found:', sum(len(v) for v in tot.values()))
