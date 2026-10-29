#!/usr/bin/env python3
"""Technique objectives vs dense adds (OBRRT_ADD), per run.

Usage: nice -n 10 python3 tools/analysis/dense_tech.py [--frames N] [--json] RUN_DIR...

Per run (streams nodes.csv and events.jsonl, read only):
  - per technique kind (Tech::*, WallKick::*, TripleJumpWallKick::*, and the
    location tricks Jrb::/Wf::/Ccm::/Opener::/IslandHop::): attempt lines,
    successes, novel successes (success that kept a node), frames;
  - attempts by start-node class: dense ("~d" via) vs ordinary, airborne
    (action & ACT_FLAG_AIR 0x800) vs grounded; their success rate and the
    nodes they added (the kept end node plus that attempt's dense
    intermediates, attributed by id order: Engine writes an attempt's event
    lines, adds its end node, then merges its dense nodes, so ids are
    [end][dense...] per attempt in event order);
  - supply: share of dense / airborne nodes in the tree.
--frames N cuts each run's events at N cumulative attempt frames (the
"frames" field), to compare runs of different length on the same budget.
"""
import json
import sys
from collections import defaultdict

AIR = 0x800
TECH_PREFIX = ("Tech::", "WallKick::", "TripleJumpWallKick::")
LOC_PREFIX = ("Jrb::", "Wf::", "Ccm::", "Opener::", "IslandHop::")


def group(kind):
    if kind.startswith(TECH_PREFIX):
        return "tech"
    if kind.startswith(LOC_PREFIX):
        return "loc"
    return "other"


def load_nodes(path):
    parent, dense, air, via = [], bytearray(), bytearray(), []
    intern = {}
    with open(path) as f:
        hdr = f.readline().strip().split(",")
        ip, iv, ia = hdr.index("parent"), hdr.index("via"), hdr.index("action")
        for line in f:
            c = line.rstrip("\n").split(",")
            if len(c) != len(hdr):  # a run in progress: partial last line
                break
            parent.append(int(c[ip]))
            v = c[iv]
            via.append(intern.setdefault(v, v))
            dense.append(v.endswith("~d"))
            try:
                air.append(bool(int(c[ia], 16) & AIR))
            except ValueError:
                air.append(0)
    return parent, dense, air, via


def cls(dense, air, n):
    return ("dense" if dense[n] else "ord") + ("-air" if air[n] else "-gnd")


def analyse(run, max_frames=None):
    parent, dense, air, via = load_nodes(f"{run}/nodes.csv")
    nn = len(parent)
    # Attempts grouped by `chain` (one plan; stage lines share it).
    attempts = []  # dict(kind, start, success, frames, end)
    cur = None
    cum = 0
    per_kind = defaultdict(lambda: [0, 0, 0, 0])  # lines, succ, novel succ, frames
    with open(f"{run}/events.jsonl") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:  # a run in progress: partial last line
                break
            if max_frames is not None and cum >= max_frames:
                break
            cum += d["frames"]
            k = per_kind[d["kind"]]
            k[0] += 1
            k[1] += d["success"]
            k[2] += bool(d["success"] and d["added"] is not None)
            k[3] += d["frames"]
            if cur is None or d["chain"] != cur["chain"] or d["stage"] == 0:
                cur = {"chain": d["chain"], "kind": d["kind"], "start": d["start"],
                       "success": False, "tech_success": False, "frames": 0, "ends": [],
                       "kinds": set()}
                attempts.append(cur)
            cur["kinds"].add(d["kind"])
            cur["frames"] += d["frames"]
            if d["success"]:
                cur["success"] = True
            if d["added"] is not None:
                cur["ends"].append(d["added"])
    # Attribute node ids to attempts: each attempt owns its end node(s) and
    # the dense nodes that follow until the next attempt's end node.
    owned = [0] * len(attempts)
    nid = 1
    cur_owner = None
    gap_cands = []
    for i, a in enumerate(attempts):
        if a["ends"]:
            e0 = min(a["ends"])
            # nodes in [nid, e0) belong to cur_owner or gap_cands
            assign_gap(nid, e0, cur_owner, gap_cands, attempts, owned, parent, dense, via)
            owned[i] += len(a["ends"])
            nid = max(a["ends"]) + 1
            cur_owner = i
            gap_cands = []
        else:
            gap_cands.append(i)
    end_id = nn if max_frames is None else nid
    assign_gap(nid, end_id, cur_owner, gap_cands, attempts, owned, parent, dense, via)
    # Aggregate.
    res = {"run": run, "nodes": nn, "frames": cum,
           "dense_nodes": sum(dense), "air_nodes": sum(air),
           "dense_air_nodes": sum(1 for n in range(nn) if dense[n] and air[n]),
           "kinds": {k: v for k, v in per_kind.items()}}
    by = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # group -> class -> att, succ, added, frames
    bykind = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # tech kind -> class -> att, succ, added
    for i, a in enumerate(attempts):
        s = a["start"]
        c = cls(dense, air, s) if s < nn else "?"
        g = group(a["kind"])
        for key in (g, "all"):
            r = by[key][c]
            r[0] += 1
            r[1] += a["success"]
            r[2] += owned[i]
            r[3] += a["frames"]
        if g != "other":
            r = bykind[a["kind"]][c]
            r[0] += 1
            r[1] += a["success"]
            r[2] += owned[i]
    res["by_class"] = {g: dict(v) for g, v in by.items()}
    res["tech_by_class"] = {k: dict(v) for k, v in bykind.items()}
    return res


def assign_gap(lo, hi, cur_owner, cands, attempts, owned, parent, dense, via):
    """Dense ids in [lo, hi): cur_owner's dense nodes, then the gap
    attempts' (no end node) in order; a node switches to a later candidate
    when its parent is outside the run so far and equals that candidate's
    start, with a matching kind."""
    seq = ([cur_owner] if cur_owner is not None else []) + cands
    if not seq:
        return
    p = 0
    for n in range(lo, min(hi, len(parent))):
        if not dense[n]:
            continue
        kind = via[n][:-2]
        if parent[n] < lo or len(seq) > 1:
            for j in range(p + 1, len(seq)):
                a = attempts[seq[j]]
                if kind in a["kinds"] and parent[n] == a["start"]:
                    p = j
                    break
        owned[seq[p]] += 1


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "-"


def report(res):
    fr = res["frames"] / 1e8
    print(f"\n=== {res['run']}  frames {res['frames']/1e6:.0f}M  nodes {res['nodes']}  "
          f"dense {pct(res['dense_nodes'], res['nodes'])}  air {pct(res['air_nodes'], res['nodes'])}  "
          f"dense&air {pct(res['dense_air_nodes'], res['nodes'])}")
    print(f"{'kind':28} {'att':>6} {'succ':>5} {'rate':>6} {'novel':>5} {'att/100M':>8} {'succ/100M':>9}")
    for k, v in sorted(res["kinds"].items()):
        if group(k) == "other":
            continue
        print(f"{k:28} {v[0]:6} {v[1]:5} {pct(v[1], v[0]):>6} {v[2]:5} {v[0]/fr:8.1f} {v[1]/fr:9.1f}")
    for g in ("tech", "loc", "all"):
        print(f" [{g}] start class: att succ rate nodes-added/att frames/att")
        for c, r in sorted(res["by_class"].get(g, {}).items()):
            print(f"   {c:10} {r[0]:6} {r[1]:5} {pct(r[1], r[0]):>6} {r[2]/max(r[0],1):7.2f} {r[3]/max(r[0],1):8.0f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    mf = None
    js = False
    if "--frames" in args:
        i = args.index("--frames")
        mf = float(args[i + 1])
        del args[i:i + 2]
    if "--json" in args:
        args.remove("--json")
        js = True
    for run in args:
        r = analyse(run.rstrip("/"), mf)
        if js:
            print(json.dumps(r, default=list))
        else:
            report(r)
