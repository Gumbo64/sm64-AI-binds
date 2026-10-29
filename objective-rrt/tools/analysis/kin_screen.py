#!/usr/bin/env python3
"""Summary of the kinship-unit / dense-thinning screen (tools/kin_arms.sh).

    python3 tools/analysis/kin_screen.py [logs/kn-*-s*.log ...]

Per log: best times (stars), nodes, frames, kin-set size (attempt unit),
pick / no-start seconds and no-starts for want of kin (rate sampler line).
"""
import glob
import re
import sys


def last(lines, prefix):
    for l in reversed(lines):
        if l.lstrip().startswith(prefix):
            return l.strip()
    return ""


def main():
    paths = sys.argv[1:] or sorted(glob.glob("logs/kn-*-s[0-9].log"))
    paths = [p for p in paths if "-id-" not in p and "-det" not in p]
    rows = []
    for p in paths:
        lines = open(p, errors="replace").read().splitlines()
        best = last(lines, "best times:")
        stars = re.findall(r"k0s(\d) (\d+:\d+\.\d)", best)
        st = {int(s): t for s, t in stars}
        prog = last(lines, "==")
        m = re.search(r"nodes (\d+)", prog)
        nodes = int(m.group(1)) if m else 0
        kin = last(lines, "kin sets")
        km = re.search(r"mean (\d+) members \(max (\d+)\).*?([\d.]+)s building \(([\d.]+) us", kin)
        rate = last(lines, "rate sampler: weight")
        rm = re.search(r"picks ([\d.]+)s started ([\d.]+)s no start", rate)
        nk = re.search(r"no-start global (\d+) kin (\d+)", rate)
        tm = last(lines, "timing:")
        pm = re.search(r"plan ([\d.]+)s run ([\d.]+)s", tm)
        done = any(l.startswith("best times") for l in lines) and "timing:" in tm
        rows.append((p.split("/")[-1][3:-4], done, nodes, st, km, rm, nk, pm))
    print(f"{'arm':34} {'done':4} {'nodes':>7} {'1 star':>7} {'2 stars':>7} {'3 stars':>7} {'kin mean':>8} {'kin max':>7} {'build us':>8} {'pick s':>7} {'nost s':>7} {'nost kin':>8} {'plan s':>7} {'run s':>7}")
    for arm, done, nodes, st, km, rm, nk, pm in rows:
        print(
            f"{arm:34} {'y' if done else 'n':4} {nodes:7} {st.get(1, '-'):>7} {st.get(2, '-'):>7} {st.get(3, '-'):>7} "
            f"{km.group(1) if km else '-':>8} {km.group(2) if km else '-':>7} {km.group(4) if km else '-':>8} "
            f"{rm.group(1) if rm else '-':>7} {rm.group(2) if rm else '-':>7} {nk.group(2) if nk else '-':>8} "
            f"{pm.group(1) if pm else '-':>7} {pm.group(2) if pm else '-':>7}"
        )


if __name__ == "__main__":
    main()
