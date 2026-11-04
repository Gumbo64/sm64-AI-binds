#!/usr/bin/env python3
"""Graft screen summary (tools/graft_screen.sh).

    tools/analysis/graft_report.py runs/<run> ...

Per run: states, the last "best times" and "first arrivals" lines of its
log (logs/<run name>.log), the reroute counters (queued / tried / ok /
frames saved; per graft kind queued, tried, kept and the replayed frames of
kept vs rejected ones; forward, relaxed and leg-reuse counters), and the
best route's segments between zone entries (zone_segments.py).
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import zone_segments  # noqa: E402


def last(path, prefix):
    out = ""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(prefix):
                    out = line.strip()
    except OSError:
        pass
    return out


def main():
    for run in sys.argv[1:]:
        name = os.path.basename(run.rstrip("/"))
        log = os.path.join(ROOT, "logs", name + ".log")
        if not os.path.exists(log):
            log = run.rstrip("/") + ".log"
        try:
            s = json.load(open(os.path.join(run, "summary.json")))
        except OSError:
            print(f"== {name}: no summary.json")
            continue
        print(f"== {name}: {s.get('nodes')} states, {s.get('frames')} frames")
        print("  ", last(log, "best times"))
        print("  ", last(log, "first arrivals"))
        r = s.get("reroute")
        if r:
            print(f"   reroute: queued {r['queued']} (dropped {r['dropped']}, waiting {r['queue']}), tried {r['tried']}, ok {r['ok']}, "
                  f"diverged {r['diverged']}, not faster {r['not_faster']}, dup {r['duplicate']}, died {r['died']}; "
                  f"frames {r['frames']}, saved {r['frames_saved']}, depth {r['max_depth']}")
            for k, v in (r.get("by_kind") or {}).items():
                fr = v["frames_kept"] + v["frames_rejected"]
                share = 100.0 * v["frames_kept"] / fr if fr else 0.0
                print(f"     {k:<8} queued {v['queued']:>7} tried {v['tried']:>7} ok {v['ok']:>6}  frames kept {v['frames_kept']:>9} rejected {v['frames_rejected']:>9} ({share:.1f}% kept)")
            for key in ("fwd", "relax"):
                if key in r:
                    print(f"     {key}: {r[key]}")
            if "legs" in r:
                l = dict(r["legs"])
                bz = l.pop("by_zone", {})
                print(f"     legs: {l}")
                print("     legs by zone (spawns/legs/queued/done): " + ", ".join(f"{z} {v['spawns']}/{v['legs']}/{v['queued']}/{v['done']}" for z, v in bz.items()))
        try:
            top, segs = zone_segments.segments(zone_segments.load(run))
            print(f"   route to {top}*: " + " | ".join(
                ("end" if k == ("end",) else f"{k[0]}:{k[1]} {k[2]}*") + f" @{fr} (+{seg}; first {first})" for k, fr, seg, first, _ in segs))
        except OSError:
            pass


if __name__ == "__main__":
    main()
