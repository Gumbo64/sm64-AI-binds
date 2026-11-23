#!/usr/bin/env python3
"""Exp. 3 grid from `gp eval` output, computed as eval_follow.py's summary():
end, prog (1 if ok else i/(n-1)), hit, median frames/ref over finished plays, died; per spacing,
all, walk and fast; held-out rows only unless --split is given."""
import re, sys, collections, statistics as st
split = "TE"
if "--split" in sys.argv:
    split = sys.argv[sys.argv.index("--split") + 1]
rows = collections.defaultdict(list)
grp = lambda l: (re.search(r"kind=(\w+)", l) or [None, ""])[1]
for l in open(sys.argv[1]):
    if not l.startswith(split): continue
    m = re.search(r",s=(\d+),fast=(\d).*ok=(\d) i=(\d+) n=(\d+) hits=(\d+) frames=(\d+) ref=(\d+) died=(\d)", l)
    if m:
        s, fast, ok, i, n, hits, fr, ref, died = map(int, m.groups())
        rows[(grp(l), s)].append((fast, ok, i, n, hits, fr, ref, died))
def summ(v):
    if not v: return "n=0"
    ok = [x for x in v if x[1]]
    end = sum(x[1] for x in v) / len(v)
    prog = sum(1 if x[1] else x[2] / max(x[3] - 1, 1) for x in v) / len(v)
    hit = sum(min(x[4] / max(x[3], 1), 1) for x in v) / len(v)
    ratio = st.median(x[5] / max(x[6], 1) for x in ok) if ok else float("nan")
    died = sum(x[7] for x in v) / len(v)
    return f"n={len(v):3} end={end:.3f} prog={prog:.3f} hit={hit:.3f} ratio={ratio:.2f} died={died:.3f}"
for k, s in sorted(rows):
    v = rows[(k, s)]
    print(f"{k:5} S={s:3} all  {summ(v)}")
    if not k:
        print(f"      walk {summ([x for x in v if not x[0]])}")
        print(f"      fast {summ([x for x in v if x[0]])}")
