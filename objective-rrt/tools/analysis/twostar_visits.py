#!/usr/bin/env python3
"""Stars per course visit in a run's tree (for OBRRT_TWOSTAR).

    tools/analysis/twostar_visits.py runs/<run> [...]

A visit is a maximal chain of tree nodes in one main-course level (its
entry: the first node whose parent is in another level); a star that ends
it is counted from the first node after the exit. Per run, per
level: visits, most coins reached in any node, how many visits got 1 / 2+
stars, whether the 100-coin star (flag bit 6, `star_flags` "course:hex")
was ever collected, and for the best 2-star visit its entry frame, the
frames from entry to the first and to the second star, and whether one of
the two was the 100-coin star. Also the soonest node per star count.
"""
import csv, sys, collections

MAIN = {9: 1, 24: 2, 12: 3, 5: 4, 4: 5, 7: 6, 22: 7, 8: 8, 23: 9, 10: 10, 11: 11, 36: 12, 13: 13, 14: 14, 15: 15}


def flags(s):
    out = {}
    for part in (s or "").split():
        c, f = part.split(":")
        out[int(c)] = int(f, 16)
    return out


def main(run):
    rows = list(csv.DictReader(open(run + "/nodes.csv")))
    by = {r["id"]: r for r in rows}
    entry = {}
    for r in rows:  # ids are in creation order: parents first
        p = by.get(r["parent"])
        entry[r["id"]] = entry[p["id"]] if p is not None and p["level"] == r["level"] else r["id"]
    lv = collections.defaultdict(lambda: {"visits": set(), "coins": 0, "hund": False})
    visit_stars = collections.defaultdict(list)  # entry id -> [(frame, stars, star_flags)]
    for r in rows:
        L = int(r["level"])
        if L not in MAIN:
            continue
        d = lv[L]
        d["visits"].add(entry[r["id"]])
        d["coins"] = max(d["coins"], int(r["coins"]))
        if flags(r["star_flags"]).get(MAIN[L], 0) & 0x40:
            d["hund"] = True
        visit_stars[entry[r["id"]]].append(r)
    # A star that ends the visit shows first in the node after the exit (the
    # automatic tail plays through the warp): credit it to the visit left.
    for r in rows:
        p = by.get(r["parent"])
        if p is not None and p["level"] != r["level"] and int(p["level"]) in MAIN and int(r["stars"]) > int(p["stars"]):
            visit_stars[entry[p["id"]]].append(r)
    print(f"== {run}: {len(rows)} nodes")
    for L, d in sorted(lv.items()):
        gained = collections.Counter()
        best2 = None
        for e in d["visits"]:
            er = by[e]
            s0 = int(er["stars"])
            nodes = visit_stars[e]
            g = max(int(n["stars"]) - s0 for n in nodes)
            gained[min(g, 2)] += 1
            if g >= 2:
                first = min((int(n["frame"]) for n in nodes if int(n["stars"]) - s0 >= 1))
                second_n = min((n for n in nodes if int(n["stars"]) - s0 >= 2), key=lambda n: int(n["frame"]))
                second = int(second_n["frame"])
                c = MAIN[L]
                hund = bool(flags(second_n["star_flags"]).get(c, 0) & 0x40) and not (flags(er["star_flags"]).get(c, 0) & 0x40)
                cand = (second, int(er["frame"]), first, hund, e)
                best2 = min(best2, cand) if best2 else cand
        line = f"level {L:2} (course {MAIN[L]:2}): visits {len(d['visits'])}, max coins {d['coins']}, visits with 0/1/2+ stars {gained[0]}/{gained[1]}/{gained[2]}, 100c star {'yes' if d['hund'] else 'no'}"
        if best2:
            second, f0, first, hund, e = best2
            line += f"; best 2-star visit: entry node {e} f{f0}, 1st star +{first - f0}, 2nd +{second - f0} (f{second}){', incl. 100c' if hund else ''}"
        print(line)
    soon = {}
    for r in rows:
        s = int(r["stars"])
        if s not in soon or int(r["frame"]) < int(soon[s]["frame"]):
            soon[s] = r
    print("soonest per star count: " + ", ".join(f"{s}: f{soon[s]['frame']}" for s in sorted(soon)))


if __name__ == "__main__":
    for run in sys.argv[1:]:
        main(run)
