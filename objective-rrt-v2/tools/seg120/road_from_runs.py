#!/usr/bin/env python3
"""Roadmap files (OBRRT_ROAD=<dir>, search/src/road.rs) from the trees of finished runs.

    tools/seg120/road_from_runs.py OUT_DIR RUN_DIR [RUN_DIR ...] [--cell 256] [--max-edge 400]

Each parent -> child pair of nodes.csv in the same level and area, 1..max-edge frames apart, is an edge between
their cells (fewest frames kept, counts summed), merged into OUT_DIR/L<level>A<area>.road. The water level
isn't in nodes.csv: cells get -32000 (unknown; the engine replaces it with its own first state there)."""
import argparse, csv, math, os, collections
csv.field_size_limit(10**8)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('runs', nargs='+')
ap.add_argument('--cell', type=float, default=256); ap.add_argument('--max-edge', type=int, default=400); A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
rep = collections.defaultdict(dict); edges = collections.defaultdict(dict)
for f in os.listdir(A.out):          # merge with what is there
    if not f.endswith('.road'): continue
    z = tuple(int(x) for x in f[1:-5].split('A'))
    for line in open(os.path.join(A.out, f)):
        p = line.split()
        if p[0] == 'r': rep[z][tuple(map(int, p[1:4]))] = (float(p[4]), float(p[5]), float(p[6]), int(p[7]))
        elif p[0] == 'e': edges[z][(tuple(map(int, p[1:4])), tuple(map(int, p[4:7])))] = [int(p[7]), int(p[8]), float(p[9]), float(p[10]), float(p[11])]
c = lambda p: tuple(int(math.floor(v / A.cell)) for v in p)
for run in A.runs:
    path = os.path.join(run, 'nodes.csv')
    if not os.path.exists(path): continue
    rows = list(csv.DictReader(open(path))); N = {r['id']: r for r in rows}; n = 0
    for r in rows:
        z = (int(r['level']), int(r['area'])); p = (float(r['x']), float(r['y']), float(r['z'])); b = c(p)
        rep[z].setdefault(b, p + (-32000,))
        par = N.get(r['parent'])
        if par is None or (int(par['level']), int(par['area'])) != z: continue
        q = (float(par['x']), float(par['y']), float(par['z'])); a = c(q); dt = int(r['frame']) - int(par['frame'])
        if a == b or not 0 < dt <= A.max_edge or math.dist(p, q) > 60 * dt + 500: continue
        e = edges[z].get((a, b))
        if e is None: edges[z][(a, b)] = [dt, 1, *p]
        else:
            e[1] += 1
            if dt < e[0]: e[0] = dt; e[2:] = p
        n += 1
    print(f'{run}: {len(rows)} nodes, {n} edges')
for z in rep:
    if not edges[z]: continue
    with open(os.path.join(A.out, f'L{z[0]}A{z[1]}.road'), 'w') as f:
        for k, v in rep[z].items(): f.write(f'r {k[0]} {k[1]} {k[2]} {v[0]:.0f} {v[1]:.0f} {v[2]:.0f} {v[3]}\n')
        for (a, b), e in edges[z].items(): f.write(f'e {a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]} {e[0]} {e[1]} {e[2]:.0f} {e[3]:.0f} {e[4]:.0f}\n')
    print(f'L{z[0]}A{z[1]}: {len(rep[z])} cells, {len(edges[z])} edges')
