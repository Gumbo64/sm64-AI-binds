#!/usr/bin/env python3
"""Rate sampler diagnosis from a run's events.jsonl + summary.json.

    tools/analysis/rate_diag.py runs/<run> [F0=2e5]

Per pool entry (key): attempts, frames spent, successes, kept states; the
attempts/frames distribution vs the prior F0; how many entries ever improved
(approximation: a kept state or a success); per-kind pooled yield (kept per
Mframe) and its spread, i.e. how much a kind-level prior would separate the
entries that no attempt has informed yet.
"""
import json, sys, collections

run = sys.argv[1]
F0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2e5
S = json.load(open(f"{run}/summary.json"))
ent = collections.defaultdict(lambda: [0, 0, 0, 0, ""])  # attempts frames successes kept kind
for line in open(f"{run}/events.jsonl"):
    e = json.loads(line)
    if "key" not in e:
        continue
    r = ent[e["key"]]
    r[0] += 1; r[1] += e.get("frames", 0); r[2] += bool(e.get("success")); r[3] += e.get("added") is not None
    r[4] = e.get("kind", "")
pool = S.get("rate", {}).get("entries", {}).get("entries") or sum(v.get("entries", 0) for v in S.get("entry_weights", {}).values())
n = len(ent)
att = sorted(r[0] for r in ent.values())
fr = sorted(r[1] for r in ent.values())
q = lambda v, f: v[min(len(v) - 1, int(f * (len(v) - 1) + .5))] if v else 0
print(f"run {run}: frames {S['frames']/1e6:.0f}M, attempts {sum(att)}, pool entries {pool}, entries ever attempted {n} ({100*n/max(pool,1):.1f}%)")
print(f"  attempts/entry (attempted): p10 {q(att,.1)} p25 {q(att,.25)} med {q(att,.5)} p75 {q(att,.75)} p90 {q(att,.9)} max {att[-1] if att else 0}")
print(f"  frames/entry  (attempted): p10 {q(fr,.1)/1e3:.0f}k p25 {q(fr,.25)/1e3:.0f}k med {q(fr,.5)/1e3:.0f}k p75 {q(fr,.75)/1e3:.0f}k p90 {q(fr,.9)/1e3:.0f}k; F0 {F0/1e3:.0f}k")
over = sum(1 for f in fr if f >= F0)
print(f"  entries with F >= F0: {over} ({100*over/max(pool,1):.2f}% of pool); with F >= F0/10: {sum(1 for f in fr if f >= F0/10)}")
imp = sum(1 for r in ent.values() if r[3] > 0 or r[2] > 0)
print(f"  entries with any kept state / success: {imp} ({100*imp/max(pool,1):.2f}% of pool, {100*imp/max(n,1):.0f}% of attempted)")
fs = S["frames"] / max(sum(att), 1)
print(f"  frames/attempt {fs/1e3:.1f}k -> F0 = {F0/fs:.1f} attempts")
# Per kind: pooled yield.
kinds = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for r in ent.values():
    k = kinds[r[4]]; k[0] += 1; k[1] += r[0]; k[2] += r[1]; k[3] += r[3]; k[4] += r[2]
pk = S.get("entry_weights", {})
ks = S.get("kinds", {})
print(f"  {'kind':32s} {'pool':>6s} {'tried':>6s} {'att':>6s} {'Mfr':>7s} {'kept':>6s} {'succ':>6s} {'kept/Mfr':>8s} {'nostart':>8s}")
rows = []
for k, (ne, a, f, kept, su) in kinds.items():
    rows.append((kept / max(f, 1) * 1e6, k, ne, a, f, kept, su))
rows.sort(reverse=True)
for y, k, ne, a, f, kept, su in rows:
    print(f"  {k:32s} {pk.get(k,{}).get('entries',0):6d} {ne:6d} {a:6d} {f/1e6:7.1f} {kept:6d} {su:6d} {y:8.2f} {ks.get(k,{}).get('no_start',0):8d}")
ys = sorted(r[0] for r in rows if r[4] >= F0)
if ys:
    print(f"  kind yield (kinds with >= F0 frames, n={len(ys)}): p10 {q(ys,.1):.2f} p25 {q(ys,.25):.2f} med {q(ys,.5):.2f} p75 {q(ys,.75):.2f} p90 {q(ys,.9):.2f} kept/Mframe")
tot_ns = sum(v.get("no_start", 0) for v in ks.values()); tot_s = sum(v.get("samples", 0) for v in ks.values())
print(f"  picks {tot_s}, no-start {tot_ns} ({100*tot_ns/max(tot_s,1):.0f}%); plan secs {S.get('timing',{}).get('plan_secs',0):.0f} of {S['secs']:.0f}")
