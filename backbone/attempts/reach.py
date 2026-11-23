#!/usr/bin/env python3
"""Per objective kind, the start ETA beyond which attempts have not succeeded in past runs (OBRRT_REACH_PRIOR).

    reach.py DATA.parquet OUT.json

reach = 1.25 x the 95th percentile of the start ETAs of the kind's successes + 30 frames. A kind gets one only
with at least 20 successes, at least 100 attempts from beyond the reach, and under 2% success there."""
import sys, json
import pandas as pd
df = pd.read_parquet(sys.argv[1]); df = df[df.start_eta.notna() & (df.plan_len == 1)]
out = {}; rows = []
for k, d in df.groupby('kind'):
    ok = d[d.success]
    if len(ok) < 20: continue
    r = 1.25 * ok.start_eta.quantile(0.95) + 30
    far = d[d.start_eta > r]
    if len(far) < 100 or far.success.mean() >= 0.02: continue
    out[k] = round(float(r), 1)
    rows.append((far.frames.sum() / 1e6, k, r, len(far), far.success.mean(), far.frames.mean(), far.useful_wip.mean()))
json.dump(out, open(sys.argv[2], 'w'), indent=1, sort_keys=True)
tot = df.frames.sum() / 1e6
print(f'{len(out)} kinds; frames beyond reach {sum(r[0] for r in rows):.0f} M of {tot:.0f} M ({100 * sum(r[0] for r in rows) / tot:.1f}%)')
for fr, k, r, n, s, mf, w in sorted(rows, reverse=True)[:25]:
    print(f'{k:28} reach {r:6.0f}  beyond: {n:6} attempts, success {s:.4f}, {mf / 1e3:5.0f} k frames each, {fr:6.0f} M, pays {w:.3f}')
