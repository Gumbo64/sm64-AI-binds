#!/usr/bin/env python3
"""Per objective kind: how often an attempt still meets the objective when no rollout of generation 0 did
(OBRRT_RESCUE_PRIOR). From every single-objective GA attempt in the runs' events.jsonl.

    rescue.py RUNS_DIR OUT.json        (kinds with at least 200 such attempts)"""
import json, collections, glob, sys
K = collections.defaultdict(lambda: [0, 0, 0.0]); T = collections.Counter()
for d in sorted(glob.glob(sys.argv[1] + '/*/events.jsonl')):
    for l in open(d):
        if '"ga":' not in l: continue
        try: e = json.loads(l)
        except Exception: continue
        g = e.get('ga')
        if not g or g.get('gen0_met') is None or e.get('len', 1) != 1: continue
        T['n'] += 1; T['frames'] += e['frames']
        if g['gen0_met'] >= 1: T['met'] += 1; T['met_ok'] += e['success']; continue
        r = K[e['kind']]; r[0] += 1; r[1] += e['success']; r[2] += e['frames']; T['miss'] += 1; T['miss_ok'] += e['success']; T['miss_frames'] += e['frames']
out = {k: round(v[1] / v[0], 4) for k, v in K.items() if v[0] >= 200}
json.dump(out, open(sys.argv[2], 'w'), indent=1, sort_keys=True)
print(f"{T['n']} attempts, {T['frames'] / 1e9:.1f} B frames. Generation 0 met: {T['met']} ({T['met_ok'] / T['met']:.3f} succeed). "
      f"Missed: {T['miss']} ({T['miss_ok'] / T['miss']:.3f} succeed), {T['miss_frames'] / T['frames']:.3f} of frames.")
low = sum(v[2] for k, v in K.items() if k in out and out[k] < 0.05)
print(f'{len(out)} kinds; {sum(1 for v in out.values() if v < 0.05)} under 5% rescue, holding {low / 1e9:.1f} B frames ({low / T["frames"]:.3f} of all)')
