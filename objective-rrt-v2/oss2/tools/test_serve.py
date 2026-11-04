#!/usr/bin/env python3
"""Checks tools/serve.py's paged views against cells.jsonl itself:

    tools/test_serve.py <port> <run> <stride> [rare]

Fetches /api/view and every /api/cells page, and checks that the kept cells
are exactly every stride-th cell plus every cell of an area with at most
`rare` cells (the server's OSS_RARE), that each line is the cells.jsonl line
plus `vp` = the nearest kept ancestor and `from` = the parent's area, and that
the summary's per-area counts match.
"""

import json
import math
import os
import sys
import time
import urllib.request

port, run, stride = sys.argv[1], sys.argv[2], int(sys.argv[3])
rare = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
base = f"http://localhost:{port}"
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", run)


def get(path):
    with urllib.request.urlopen(base + path) as r:
        return r.read()


t0 = time.time()
view = json.loads(get(f"/api/view?run={run}&stride={stride}"))
t_view = time.time() - t0
cells = [json.loads(l) for l in open(os.path.join(root, "cells.jsonl"))]
assert view["total"] == len(cells), (view["total"], len(cells))
zone = [c["key"][0][2] if c["key"] and c["key"][0][:2] == [0, 0] else None for c in cells]
counts = {}
for z in zone:
    counts[z] = counts.get(z, 0) + 1
assert {a["zone"]: a["n"] for a in view["summary"]["areas"]} == {z: n for z, n in counts.items() if z is not None}, "summary counts"
want = [i for i in range(len(cells)) if i % stride == 0 or counts[zone[i]] <= rare]
assert view["kept"] == len(want), (view["kept"], len(want))
keep = set(want)
got, offset, t_pages, pages = [], 0, 0.0, 0
while offset < view["kept"]:
    t0 = time.time()
    body = get(f"/api/cells?run={run}&stride={stride}&v={view['v']}&offset={offset}&limit=5000")
    t_pages += time.time() - t0
    pages += 1
    lines = [json.loads(l) for l in body.splitlines()]
    assert lines, "empty page"
    got += lines
    offset += len(lines)
assert [c["id"] for c in got] == want, "kept ids"
for c in got:
    i = c["id"]
    p = cells[i]["parent"]
    q = p
    while q >= 0 and q not in keep:
        q = cells[q]["parent"]
    assert c["vp"] == q, (i, c["vp"], q)
    assert c["from"] == (zone[p] if p >= 0 else None), (i, c["from"])
    stripped = {k: v for k, v in c.items() if k not in ("vp", "from")}
    assert stripped == cells[i], i
print(f"ok: {len(got)} of {len(cells)} cells kept at stride {stride} (rare {rare}); view {t_view*1000:.0f} ms, {pages} pages {t_pages*1000:.0f} ms")

# ---------- region queries (needs cells.meta v2 positions and cells.grid) ----------
import struct

meta = open(os.path.join(root, "cells.meta"), "rb").read()
n = len(cells)
if int.from_bytes(meta[4:8], "little") >= 2:
    pos = list(struct.iter_unpack("<hhh", meta[24 + 20 * n:24 + 26 * n]))
    zones_m = memoryview(meta)[24 + 12 * n:24 + 16 * n].cast("I")
    big = max(view["summary"]["areas"], key=lambda a: a["n"])["zone"]
    inz = [i for i in range(n) if zones_m[i] == big]
    xs = sorted(pos[i][0] for i in inz)
    zs = sorted(pos[i][2] for i in inz)
    box = [xs[len(xs) // 4], -20000, zs[len(zs) // 4], xs[len(xs) // 2], 20000, zs[len(zs) // 2]]
    expect = sorted(i for i in inz if box[0] <= pos[i][0] <= box[3] + 255 and box[2] <= pos[i][2] <= box[5] + 255)
    for cap, label in ((10 ** 6, "all"), (max(1, len(expect) // 10), "thinned")):
        q = f"/api/region?run={run}&stride={stride}&v={view['v']}&zone={big}&box={','.join(map(str, box))}&max={cap}"
        with urllib.request.urlopen(base + q) as r:
            total, step = int(r.headers["X-Region-Total"]), int(r.headers["X-Region-Step"])
            lines = [json.loads(l) for l in r.read().splitlines()]
        ids = [c["id"] for c in lines]
        inbox = [i for i in inz if all(math.floor(b / 256) * 256 <= pos[i][k] < (math.floor(e / 256) + 1) * 256 for k, b, e in ((0, box[0], box[3]), (1, box[1], box[4]), (2, box[2], box[5])))]
        assert total == len(inbox), (label, total, len(inbox))
        assert set(ids) <= set(inbox), label
        if label == "all":
            assert ids == sorted(inbox) and step == 1, label
        else:
            assert step > 1 and len(ids) <= 2 * cap + 1000, (label, step, len(ids), cap)
        pick = set(ids)
        for c in lines:
            q2 = cells[c["id"]]["parent"]
            while q2 >= 0 and q2 not in keep and q2 not in pick:
                q2 = cells[q2]["parent"]
            assert c["vp"] == q2, (label, c["id"])
        print(f"region {label}: {len(ids)} of {total} cells in the box (step {step}) ok")
