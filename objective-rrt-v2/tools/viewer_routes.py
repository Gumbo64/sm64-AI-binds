#!/usr/bin/env python3
"""Put the notable King Bob-omb routes and v2 tree runs in sm64-viewer's replay list, with one region per
leg. Copies the inputs to <viewer>/obrrt2/ and replaces this tool's entries in index.json, keeping the rest.

    tools/viewer_routes.py [viewer_replays_dir]
"""
import csv, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
VIEWER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "sm64-viewer/frontend/public/replays"
PREFIX = "King route: "
TREE = "Objective RRT v2: "
COLORS = ["#8b93a7", "#4c8bf5", "#2bb673", "#f5a623", "#e0568a", "#9b6bf2", "#35b8c9"]
ROUTES = [
    ("record, star at 3610", "legs-kingloop-0113-r1-x1-blj-tas_ramp0", True, "Best King Bob-omb kill: star at frame 3610 (2:00.3). Elevator BLJ, two tree redirects, wall kick and double jumps to the summit, as the 2026 TAS's chain, then the fight per hit."),
    ("tree and hyperjump, star at 3829", "legs-kingloop-0061-r1-x2-tree-d", False, "BLJ, tree redirect, a hyperjump above the slippery east slope, then the walked upper path. Star at frame 3829 (2:07.6)."),
    ("walked, star at 4075", "legs-kingloop-0094-r1-x1-hits", False, "The walked route, refined by the loop. Star at frame 4075 (2:15.8)."),
    ("speed chain, unfinished", "legs-speed-blj-fresh-s1", False, "Unfinished: the spawn slope's speed chain to the BLJ setup point (161 frames from control; TAS 147), then both trees, done at frame 2719, 24 frames ahead of the record. Failed the leg to the high ledge."),
    ("VCutM from the 7th coin, switch first", "legs-tas-switch-first-2362-s2", False, "The 2026 TAS's first star with the vanish cap switch pressed before the 8th red coin: star at 2642, against the TAS's 2569. Landing after the switch opens its dialog, about 84 frames."),
]
TREES = [("q15-pw1-s2", 4, "Parent window on every attempt: second star at 2:49.4, third at 3:46.6."), ("q15-pw1-s3", 5, "Parent window on every attempt: five stars, the fifth at 7:11.4.")]


def regions(route):
    out, at = [{"from": 0, "label": "Boot", "note": "The TAS's boot inputs, to the castle grounds.", "color": COLORS[0]}], 1646
    for i, l in enumerate(route["legs"]):
        if l.get("frames") is None:
            continue
        out.append({"from": at, "label": l["name"], "note": f"{l['frames']} frames", "color": COLORS[1 + i % (len(COLORS) - 1)]})
        at += l["frames"]
    return out


def best_node(run, stars):
    rows = list(csv.DictReader(open(HERE / "runs" / run / "nodes.csv")))
    by = {r["id"]: r for r in rows}
    return min((r for r in rows if r["id"] in by and int(r["stars"]) == stars and by.get(r["parent"]) and int(by[r["parent"]]["stars"]) < stars), key=lambda r: int(r["frame"]))


def main():
    out = VIEWER / "obrrt2"; out.mkdir(exist_ok=True)
    index = [e for e in json.loads((VIEWER / "index.json").read_text()) if not e.get("title", "").startswith((PREFIX, TREE))]
    date = "2026-11-04T00:00:00Z"
    for title, run, important, doc in ROUTES:
        d = HERE / "oss2/runs" / run
        shutil.copyfile(d / "route.m64", out / f"{run}.m64")
        e = {"date": date, "title": PREFIX + title, "description": doc, "file": f"obrrt2/{run}.m64", "start": 1646, "regions": regions(json.load(open(d / "route.json")))}
        if important:
            e["important"] = True
        index.append(e)
    for run, stars, doc in TREES:
        n = best_node(run, stars); f = out / f"{run}-star{stars}.m64"
        subprocess.run([str(HERE / "target/release/nodepath"), str(HERE / "runs" / run), n["id"], str(f)], check=True, stdout=subprocess.DEVNULL)
        index.append({"date": date, "title": TREE + f"{run}, {stars} stars", "description": f"{doc} Node {n['id']}, frame {n['frame']}.", "file": f"obrrt2/{f.name}", "start": 1646})
    (VIEWER / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(len(index), "entries;", sum(p.stat().st_size for p in out.iterdir()) // 1024, "KB in obrrt2/")


main()
