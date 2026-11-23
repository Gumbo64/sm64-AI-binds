#!/usr/bin/env python3
"""Paired frames-to-done: compare.py <baseline log> <eval log>...: per split, the movie and
overlay rows of `gp baseline`, then each program's, over the cases all of them finish."""
import re, sys
def rows(path, section=None):
    out = {}; cur = None
    for l in open(path):
        if l.startswith('---'): cur = 'movie' if 'movie' in l else 'overlay'
        m = re.match(r'(tr|TE|VA)\s+(\S+)\s+(-?\d+)\s+(-?\d+)\s+(\S+)\s+(\d+)', l)
        if m and (section is None or cur == section):
            out[m[2]] = (m[1], None if m[5] == '-' else int(m[5]))
    return out
base = sys.argv[1]
arms = {'movie': rows(base, 'movie'), 'overlay': rows(base, 'overlay')}
for p in sys.argv[2:]:
    arms[p.rsplit('/', 1)[-1]] = rows(p)
names = list(arms)
print(f"{'arm':<18}{'tr ok':>7}{'tr mean':>9}{'TE ok':>7}{'TE mean':>9}{'tr paired':>11}{'TE paired':>11}")
for a in names:
    r = []
    for sp in ('tr', 'TE'):
        cs = [c for c, v in arms[a].items() if v[0] == sp]
        ok = [arms[a][c][1] for c in cs if arms[a][c][1] is not None]
        r.append((len(ok), len(cs), sum(ok) / max(len(ok), 1)))
    p = []
    for sp in ('tr', 'TE'):
        common = [c for c, v in arms['movie'].items() if v[0] == sp and v[1] is not None and arms[a].get(c, (0, None))[1] is not None]
        p.append(f"{sum(arms[a][c][1] for c in common) / max(len(common), 1):.1f}/{len(common)}")
    print(f"{a:<18}{r[0][0]:>3}/{r[0][1]:<3}{r[0][2]:>9.1f}{r[1][0]:>3}/{r[1][1]:<3}{r[1][2]:>9.1f}{p[0]:>11}{p[1]:>11}")
