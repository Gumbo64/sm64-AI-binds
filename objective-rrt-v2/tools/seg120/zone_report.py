#!/usr/bin/env python3
"""What a zone tree found: stars beyond the root's, exits, how high it got, where the frames went.
    tools/seg120/zone_report.py runs/NAME [top kinds]"""
import json, csv, collections, sys, os
csv.field_size_limit(10**8)
d = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows = list(csv.DictReader(open(d + '/nodes.csv'))); root = rows[0]
try: s = json.load(open(d + '/summary.json')); print(f'{d}: {s["frames"] / 1e6:.1f} M frames, {s["nodes"]} nodes, {int(s["secs"])} s')
except Exception: print(f'{d}: no summary (still running or stopped); {len(rows)} nodes')
def flags(t):
    return {int(p.split(':')[0]): int(p.split(':')[1], 16) & 0x7f for p in t.split()}
f0 = flags(root['star_flags']); first = {}
for r in rows[1:]:
    for c, m in flags(r['star_flags']).items():
        for i in range(7):
            if (m & ~f0.get(c, 0)) >> i & 1:
                k = (c, i + 1); t = int(r['frame']) - int(root['frame'])
                if k not in first or t < first[k]: first[k] = t
print('  new stars (course, star) @ frames from the root:', sorted(first.items()) or 'none')
out = collections.Counter(r['level'] for r in rows if r['level'] != root['level'])
print('  levels:', dict(out), ' max y', max(float(r['y']) for r in rows), ' red', dict(collections.Counter(r['red_coins'] for r in rows)), ' max coins', max(int(r['coins']) for r in rows))
att = collections.Counter(); suc = collections.Counter(); fr = collections.Counter(); when = {}; at_node = []
for l in open(d + '/events.jsonl'):
    try: e = json.loads(l)
    except Exception: continue
    if e.get('event') == 'best_improved' and e.get('milestone') == 'progress':
        when.setdefault(tuple(e['progress']), e['frames_total'])
    if 'event' in e: continue
    k = e['kind']; att[k] += 1; suc[k] += e['success']; fr[k] += e['frames']
    if e.get('added') is not None: at_node.append((e['added'], sum(fr.values())))
# Attempt frames spent when the first state with each new star was added (the last attempt that added a node at or before it).
import bisect
ids = [a for a, _ in at_node]; seen = {}
for r in rows[1:]:
    for c, m in flags(r['star_flags']).items():
        for i in range(7):
            if (m & ~f0.get(c, 0)) >> i & 1 and (c, i + 1) not in seen:
                j = bisect.bisect_right(ids, int(r['id'])) - 1; seen[(c, i + 1)] = at_node[j][1] if j >= 0 else 0
print('  attempt frames spent when each star was first had:', {k: f'{v / 1e6:.1f} M' for k, v in sorted(seen.items())})
tot = sum(fr.values()) or 1; print(f'  {sum(att.values())} attempts, {tot / 1e6:.1f} M frames in attempts')
for k, v in fr.most_common(top): print(f'    {k:30} {att[k]:6} att {suc[k]:6} ok {100 * v / tot:5.1f}% of frames {v // max(1, att[k]):7} each')
