#!/usr/bin/env python3
"""Attempt dataset from objective-rrt-v2 runs: one row per GA attempt (events.jsonl) joined with the node it
started from (nodes.csv), with labels known only in hindsight.

    build.py OUT.parquet [RUNS_DIR]

Labels
  success      the objective was met
  added        a node was added
  useful_star  the subtree under the added node reaches more stars+keys than the start node had
  useful_wip   ... or more red coins in the same level, or (zone-locked runs) a kept exit
  gain_frames  for useful_star: route frames from the start node to the first such descendant
Features are only what the search knows at the time of the attempt: the start node's state, the objective,
its ETA, and running counts of this run so far (per kind, per kind at this node)."""
import csv, json, os, sys, collections
import numpy as np, pandas as pd
csv.field_size_limit(10**8)
OUT = sys.argv[1]
RUNS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../objective-rrt-v2/runs')


def group(name):
    import re
    m = re.match(r'(b\d|f5|f8|fn)-\d+-(\w+)-act\d+$', name)
    if m: return m.group(1), m.group(2)
    m = re.match(r'camp-(\w+)-(\d+)$', name)
    if m: return 'camp-' + m.group(1), m.group(2)
    m = re.match(r'(.*)-s?(\d+)$', name)
    if m: return m.group(1), m.group(2)
    return name, '0'


rows = []
for run in sorted(os.listdir(RUNS)):
    d = os.path.join(RUNS, run)
    if not (os.path.exists(d + '/events.jsonl') and os.path.exists(d + '/nodes.csv')): continue
    try:
        nodes = pd.read_csv(d + '/nodes.csv', dtype={'action': str, 'star_flags': str, 'via': str}, engine='c', on_bad_lines='skip')
    except Exception as e:
        print('skip', run, e, file=sys.stderr); continue
    n = len(nodes)
    if n < 2 or nodes['id'].iloc[-1] != n - 1:
        print('skip (ids)', run, file=sys.stderr); continue
    par = nodes['parent'].to_numpy()
    sk = (nodes['stars'] + nodes['keys']).to_numpy().astype(np.int64)
    red = nodes['red_coins'].to_numpy().astype(np.int64) if 'red_coins' in nodes else np.zeros(n, np.int64)
    lvl = nodes['level'].to_numpy(); frame = nodes['frame'].to_numpy().astype(np.int64)
    retired = nodes['retired'].to_numpy() if 'retired' in nodes else np.zeros(n, np.int64)
    root_lvl = lvl[0]; hub = {6, 16, 26}
    inside = hub if root_lvl in hub else {root_lvl, {17: 30, 19: 33, 21: 34}.get(root_lvl, root_lvl)}
    locked = run.startswith(('b2', 'f5', 'f8', 'fn', 'camp'))
    is_exit = np.array([locked and retired[i] == 1 and lvl[i] not in inside for i in range(n)])
    # Subtree bests (children have larger ids): most stars+keys, and the earliest frame that reaches each count.
    best_sk = sk.copy(); best_red = red.copy(); any_exit = is_exit.copy()
    first = [None] * n   # per node: dict count -> earliest frame in the subtree (only counts above the node's own)
    for i in range(n - 1, 0, -1):
        p = par[i]
        if p < 0: continue
        if best_sk[i] > best_sk[p]: best_sk[p] = best_sk[i]
        if lvl[i] == lvl[p] and best_red[i] > best_red[p]: best_red[p] = best_red[i]
        if any_exit[i]: any_exit[p] = True
    # Earliest frame at which a subtree reaches (its best) stars+keys: one pass keeping the min frame of nodes at best.
    t_best = np.where(sk == best_sk, frame, np.iinfo(np.int64).max)
    for i in range(n - 1, 0, -1):
        p = par[i]
        if p >= 0 and best_sk[i] == best_sk[p] and t_best[i] < t_best[p]: t_best[p] = t_best[i]
    cfg, seed = group(run)
    kind_n = collections.Counter(); kind_s = collections.Counter(); kn_n = collections.Counter(); node_n = collections.Counter()
    total = 0; t = 0
    with open(d + '/events.jsonl') as f:
        for line in f:
            try: e = json.loads(line)
            except Exception: continue
            if 'event' in e: continue
            fr = e.get('from'); kind = e.get('kind')
            if fr is None or fr >= n or kind is None: continue
            a = e.get('added'); ok = bool(e.get('success'))
            a_ok = a is not None and a < n
            us = bool(a_ok and best_sk[a] > sk[fr])
            uw = bool(us or (a_ok and ((lvl[a] == lvl[fr] and best_red[a] > red[fr]) or any_exit[a])))
            ga = e.get('ga') or {}
            nd = nodes.iloc[fr]
            rows.append(dict(
                run=run, cfg=cfg, seed=seed, t=t, frames_before=total, kind=kind, tech=kind.split('::')[0],
                start_eta=e.get('start_eta'), stage=e.get('stage', 0), chain=e.get('chain', 0), plan_len=e.get('len', 1),
                is_b=int(e.get('b') == fr), is_start=int(e.get('start') == fr),
                level=int(nd['level']), area=int(nd['area']), depth=int(nd['depth']), x=float(nd['x']), y=float(nd['y']), z=float(nd['z']),
                stars=int(nd['stars']), keys=int(nd['keys']), coins=int(nd['coins']), red=int(red[fr]), route_frame=int(frame[fr] - frame[0]),
                hspeed=float(nd['hspeed']), fwd_vel=float(nd['fwd_vel']), action=str(nd['action']), via=str(nd['via']).split('~')[0],
                kind_n=kind_n[kind], kind_s=kind_s[kind], kn_n=kn_n[(kind, fr)], node_n=node_n[fr],
                frames=int(e.get('frames') or 0), seg_frames=int(e.get('seg_frames') or 0), generations=ga.get('generations'),
                success=ok, added=a_ok, died=bool(e.get('died')), useful_star=us, useful_wip=uw,
                gain_frames=int(t_best[a] - frame[fr]) if us else -1,
            ))
            kind_n[kind] += 1; kind_s[kind] += ok; kn_n[(kind, fr)] += 1; node_n[fr] += 1
            total += int(e.get('frames') or 0); t += 1
    print(f'{run}: {t} attempts, {n} nodes', file=sys.stderr)
df = pd.DataFrame(rows)
df.to_parquet(OUT)
print(len(df), 'attempts from', df.run.nunique(), 'runs;', 'success', round(df.success.mean(), 3), 'useful_star', round(df.useful_star.mean(), 4), 'useful_wip', round(df.useful_wip.mean(), 4))
