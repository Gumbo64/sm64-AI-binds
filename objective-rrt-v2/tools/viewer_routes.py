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
]
# Mended routes (spec/src/bin/mend.rs): the same route, only faster. Regions come from a replay.
MENDED = [
    ("King's star at 3541", "king-3541", True, "The King route after mending: 3610 -> 3541. Trim, window GA and repair, accepted only if the star came sooner."),
    ("4th star at 8074", "q15-s2-star4-8074", True, "q15-pw1-s2's route after mending: the 4th star 9390 -> 8074 (-1316), same star order. Compare with the tree's own run below; most of the saving is in Whomp's Fortress."),
    ("5th star at 11274", "q15-s3-star5-11274", True, "q15-pw1-s3's route after mending: the 5th star 12782 -> 11274 (-1508), 6:15.8, same star order. Compare with the tree's own run below."),
]
# Composed from the segment library (logs/funnel/compose.py): segments of many runs joined at zone entries.
COMPOSED = [
    (6, 14023, True, "Fastest 6 stars: 7:47.4. Composed with no search from segments of 8 different runs, joined at zone entries where Mario's state is identical across routes. Best single run before: 17037."),
    (5, 9793, False, "Fastest 5 stars: 5:26.4, composed from 7 runs. Best single route before (mended): 10963."),
    (4, 7778, False, "Fastest 4 stars: 4:19.3, composed from 6 runs. Best single route before (mended): 7937."),
    (1, 2888, False, "Fastest first star: 1:36.3. The King route's opening into BoB, then a tree run's Chain Chomp star."),
]
TREES = [("q15-pw1-s2", 4, "Parent window on every attempt: second star at 2:49.4, third at 3:46.6."), ("q15-pw1-s3", 5, "Parent window on every attempt: five stars, the fifth at 7:11.4.")]


def replay_regions(m64):
    "One region per zone visit and star, from a replay; returns (regions, {stars: frame})."
    names = {16: "Castle grounds", 6: "Castle", 9: "Bob-omb Battlefield", 24: "Whomp's Fortress", 26: "Courtyard", 5: "Cool, Cool Mountain", 18: "Vanish Cap", 27: "Princess's Slide", 3: "Jolly Roger Bay"}
    tmp = VIEWER / ".replay.csv"
    subprocess.run([str(HERE / "builds/tools/tasplay"), str(HERE.parent / "baserom.us.z64"), str(m64), str(tmp)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out, stars, zone, got, n = [], {}, None, 0, 0
    for r in csv.DictReader(open(tmp)):
        f, l, s = int(r["frame"]), int(r["level"]), int(r["stars"])
        if s > got:
            got = s; stars[s] = f
            out.append({"from": f, "label": f"Star {s}", "note": f"Frame {f}, {f // 1800}:{f % 1800 / 30:04.1f}", "color": "#f5a623"})
            n += 1
        if l != zone and l:
            zone = l
            out.append({"from": f, "label": names.get(l, f"Level {l}"), "note": "", "color": COLORS[1 + n % (len(COLORS) - 1)]})
            n += 1
    tmp.unlink()
    return out, stars


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
    index = [e for e in json.loads((VIEWER / "index.json").read_text()) if not e.get("title", "").startswith((PREFIX, TREE, "Mended: ", "Composed: "))]
    date = "2026-11-04T00:00:00Z"
    for title, run, important, doc in ROUTES:
        d = HERE / "oss2/runs" / run
        shutil.copyfile(d / "route.m64", out / f"{run}.m64")
        e = {"date": date, "title": PREFIX + title, "description": doc, "file": f"obrrt2/{run}.m64", "start": 1646, "regions": regions(json.load(open(d / "route.json")))}
        if important:
            e["important"] = True
        index.append(e)
    for k, frame, important, doc in COMPOSED:
        src = HERE / f"logs/funnel/composed-{k}.m64"
        shutil.copyfile(src, out / f"composed-{k}.m64")
        regs, stars = replay_regions(src)
        assert stars.get(k) == frame, (k, stars)
        e = {"date": date, "title": f"Composed: {k} star{'s' if k > 1 else ''} at {frame}", "description": doc + f" Replay: stars at {list(stars.values())}.", "file": f"obrrt2/composed-{k}.m64", "start": 1646, "regions": regs}
        if important:
            e["important"] = True
        index.append(e)
    for title, name, important, doc in MENDED:
        src = HERE / "logs/mend" / f"{name}.m64"
        shutil.copyfile(src, out / f"{name}.m64")
        regs, stars = replay_regions(src)
        e = {"date": date, "title": "Mended: " + title, "description": doc + f" Replay: stars at {list(stars.values())}.", "file": f"obrrt2/{name}.m64", "start": 1646, "regions": regs}
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
