#!/usr/bin/env python3
"""One line per arm: end/prog on the grid at S 16, 64, 256; mean end/prog per rewire kind over its spacings."""
import re, sys, subprocess, collections
for a in sys.argv[1:]:
    out = []
    for suf in (".eval", ".rw"):
        p = f"fol-{a}{suf}"
        try:
            txt = subprocess.run(["python3", "../../gp/follow_summary.py", p], capture_output=True, text=True).stdout
        except Exception:
            continue
        d = collections.defaultdict(list)
        for l in txt.split("\n"):
            m = re.match(r"(\w*)\s+S=\s*(\d+) all\s+n=\s*\d+ end=([\d.]+) prog=([\d.]+)", l)
            if m:
                d[m[1]].append((int(m[2]), float(m[3]), float(m[4])))
        if suf == ".eval":
            for s, e, pr in d[""]:
                if s in (16, 64, 256):
                    out.append(f"{s}:{e:.2f}/{pr:.2f}")
        else:
            for k in ("K15", "K30", "K60", "graft"):
                v = d[k]
                if v:
                    out.append(f"{k}:{sum(x[1] for x in v)/len(v):.2f}/{sum(x[2] for x in v)/len(v):.2f}")
    print(f"{a:10}", " ".join(out))
