#!/usr/bin/env python3
"""One round of the Eureka-style loop over spec candidates, minus the LLM:
validate each candidate against a known route, screen it with a short piped
obrrt2 run, and write a feedback summary an LLM can revise from.

  tools/spec/loop.py RUN_DIR --from MILESTONE --to MILESTONE CAND.json [CAND.json ...]
        [--node ID] [--schema SCHEMA.json] [--screen] [--frames 5000000] [--seed 1]
        [--out DIR] [--obrrt2 BIN] [--validate BIN]

MILESTONE as in pack.py (f<frame>, n<id>, stars=<n>, zone=<l[:a]>, flag=<c>:<hex>, start, end).

Per candidate:
1. validate: spec-validate CAND RUN_DIR --node <route end> --json ... (gate, Done vs expect,
   calibration, ETA jumps, Lost on the route, schema separation with --schema).
2. screen (--screen): the inputs to the segment's start node become a start .m64, and obrrt2
   runs from there (OBRRT_BOOT_NOWAIT=1, 2 threads, OBRRT_MEM_MB=1500, tools/screen.sh's base
   env with the P arm) with OBRRT_SPEC=CAND for --frames. Scored by the frames from the start
   to the first node that meets the segment's end milestone, the node count, and the spec's
   attempts/successes (summary.json `spec`).
3. feedback.md + feedback.json in --out: per candidate the problems found, the calibration,
   the screen score, and a reflection prompt; best.json is the best candidate (fewest problems,
   then the fastest segment end, then the best calibration).

The LLM step is `propose()`: given the brief (pack.py) and the last feedback, return new
candidate JSON texts. It is not implemented; wire a model in there.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import pack  # noqa: E402

ENV_P = {
    "OBRRT_SOFT_DEDUP": "1", "OBRRT_MEM_MB": "1500", "OBRRT_ROUND": "64",
    "OBRRT_TECH": "all", "OBRRT_NEW_ELIGIBLE": "1", "OBRRT_STARSET_KEY": "1", "OBRRT_PICK_BY_B": "0", "OBRRT_SWIM_HOLD": "1",
    "OBRRT_GP_CATCH": "1", "OBRRT_KEEP_WIP": "1", "OBRRT_MOVE_WEIGHT": "0.75",
    "OBRRT_CHAIN": "route", "OBRRT_SAMPLER": "rate",
    "OBRRT_MOVES": "1", "OBRRT_MOVE_GA": "1", "OBRRT_MOVE_GA_EXTRA": "1", "OBRRT_CALM": "1", "OBRRT_LAKITU_ARC": "1",
    "OBRRT_PAINTING_FAST": "1", "OBRRT_ADD": "last_gen", "OBRRT_KINSHIP_MULT": "10",
    "RAYON_NUM_THREADS": "2", "OBRRT_BOOT_NOWAIT": "1",
}


def propose(brief_md, feedback_md, k):
    """The LLM step: return k candidate spec JSON texts for the segment.

    Inputs: the brief (tools/spec/pack.py) and the previous round's feedback.md (empty on the
    first round). Prompt sketch: "Write k spec objectives (FORMAT.md) for this segment; the
    previous candidates and their validator/screen results are below; fix the listed problems
    (ETA calibration, jumps, Lost on the route, Done far from the milestone, unseparated
    phases) and keep what scored best." Parse each JSON block, then call this script on them.
    """
    raise NotImplementedError("wire an LLM here (see README.md, 'The Eureka loop')")


def node_inputs(run, node):
    b = open(os.path.join(run, "tree.bin"), "rb").read()
    n = struct.unpack_from("<I", b, 0)[0]
    at, rec = 4, []
    for _ in range(n):
        p, k = struct.unpack_from("<iI", b, at)
        rec.append((p, b[at + 8:at + 8 + 4 * k]))
        at += 8 + 4 * k
    chain, x = [node + 1], node
    while True:
        p = rec[x + 1][0]
        chain.append(p + 1 if p >= 0 else 0)
        if p < 0:
            break
        x = p
    return b"".join(rec[r][1] for r in reversed(chain))


def end_met(row, spec, start_frame):
    """Whether a nodes.csv row meets the end milestone (stars=, zone=, flag=, f<frame> relative)."""
    if spec.startswith("stars="):
        return int(row["stars"]) >= int(spec[6:])
    if spec.startswith("zone="):
        l, a = pack.parse_zone(spec[5:])
        return int(row["level"]) == l and (a is None or int(row["area"]) == a)
    if spec.startswith("flag="):
        c, _, h = spec[5:].partition(":")
        return pack.flags_have(row, int(c), int(h, 16))
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", required=True)
    ap.add_argument("--node", default=None)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--frames", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="spec-loop")
    ap.add_argument("--obrrt2", default=os.environ.get("OBRRT2", os.path.join(V2, "target", "release", "obrrt2")))
    ap.add_argument("--validate", default=os.environ.get("SPEC_VALIDATE", os.path.join(V2, "target", "release", "spec-validate")))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    route = pack.load_route(a.run, a.node)
    i0 = pack.find(route, a.frm)
    i1 = pack.find(route, a.to, i0)
    # The screen starts where play goes on: skip nodes followed by a menu or
    # dialog step (obrrt2's boot waits for a controllable state).
    j = i0
    while j < i1 and route[j + 1]["via"].split("~")[0].startswith(("Menu::", "Dialog::")):
        j += 1
    start, end = route[j], route[i1]
    route_frames = int(end["frame"]) - int(start["frame"])
    results = []
    for c in a.cands:
        name = os.path.splitext(os.path.basename(c))[0]
        r = {"candidate": c, "name": name}
        vj = os.path.join(a.out, f"{name}-validate.json")
        cmd = [a.validate, c, a.run, "--node", route[-1]["id"], "--json", vj]
        if a.schema:
            cmd += ["--schema", a.schema]
        p = subprocess.run(cmd, capture_output=True, text=True)
        r["validate_text"] = p.stdout[-6000:]
        if p.returncode != 0:
            r["error"] = (p.stderr or p.stdout)[-3000:]
            results.append(r)
            continue
        rep = json.load(open(vj))
        r["specs"] = rep["specs"]
        r["problems"] = sum(len(s.get("problems", [])) for s in rep["specs"])
        meds = [s["calibration"]["median"] for s in rep["specs"] if s.get("calibration", {}).get("median")]
        r["calibration_err"] = max((abs(__import__("math").log(m)) for m in meds), default=None)
        if a.screen:
            sd = os.path.join(a.out, f"{name}-screen")
            shutil.rmtree(sd, ignore_errors=True)
            m64 = os.path.join(a.out, f"start-{start['id']}.m64")
            if not os.path.exists(m64):
                open(m64, "wb").write(node_inputs(a.run, int(start["id"])))
            env = dict(os.environ, **ENV_P, OBRRT_MAX_FRAMES=str(a.frames), OBRRT_SPEC=os.path.abspath(c))
            log = open(sd + ".log", "w")
            subprocess.run(["nice", "-n", "10", a.obrrt2, os.path.join(V2, "..", "baserom.us.z64"), sd, str(a.seed), "600", m64],
                           env=env, stdout=log, stderr=subprocess.STDOUT, cwd=V2)
            rows = list(pack.csv.DictReader(open(os.path.join(sd, "nodes.csv"))))
            f0 = int(rows[0]["frame"])
            hit = [int(x["frame"]) - f0 for x in rows if end_met(x, a.to, f0)]
            summ = json.load(open(os.path.join(sd, "summary.json")))
            r["screen"] = {"segment_end_frames": min(hit) if hit else None, "route_frames": route_frames,
                           "nodes": len(rows), "spec": summ.get("spec", {}).get("attempts", {})}
        results.append(r)

    def score(r):
        s = r.get("screen", {}).get("segment_end_frames")
        return (r.get("error") is not None, r.get("problems", 99), s if s is not None else 1e9, r.get("calibration_err") or 9)

    results.sort(key=score)
    lines = [f"# Spec loop feedback: {a.run} {a.frm} -> {a.to} (route segment {route_frames} frames)\n"]
    for r in results:
        lines.append(f"## {r['name']} ({r['candidate']})\n")
        if r.get("error"):
            lines.append("**Rejected: the spec did not load or the validator failed:**\n\n```\n" + r["error"].strip() + "\n```\n")
            continue
        for s in r["specs"]:
            cal = s.get("calibration", {})
            lines.append(f"- {s['name']}{'/' + s['variant'] if s.get('variant') else ''}: gate {s['gate'] and s['gate']['frame']}, "
                         f"done {s.get('done') and s['done']['frame']}, expect {s.get('expect') and s['expect']['frame']}, "
                         f"done-expect {s.get('done_minus_expect')}, calibration median {cal.get('median')} (p10 {cal.get('p10')}, p90 {cal.get('p90')}), "
                         f"jumps {len(s.get('jumps', []))}, lost {len(s.get('lost', []))}")
            for pr in s.get("problems", []):
                lines.append(f"  - problem: {pr}")
        if "screen" in r:
            sc = r["screen"]
            lines.append(f"- screen ({a.frames} frames from node {start['id']}): segment end after {sc['segment_end_frames']} frames "
                         f"(route: {route_frames}); {sc['nodes']} nodes; spec attempts {json.dumps(sc['spec'])}")
        lines.append("\nReflection: which phase's ETA is off (see calibration and jumps), whether Done lands on the milestone, "
                     "whether the gate starts where the route can use it, and what to change.\n")
    open(os.path.join(a.out, "feedback.md"), "w").write("\n".join(lines) + "\n")
    json.dump(results, open(os.path.join(a.out, "feedback.json"), "w"), indent=1)
    if results and not results[0].get("error"):
        shutil.copy(results[0]["candidate"], os.path.join(a.out, "best.json"))
    print("\n".join(lines))
    print(f"wrote {a.out}/feedback.md, feedback.json, best.json")


if __name__ == "__main__":
    main()
