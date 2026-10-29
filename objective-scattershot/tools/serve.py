#!/usr/bin/env python3
"""Data server for the objective-scattershot GUI (sm64-viewer, ?oss=).

    tools/serve.py [port]          (default 8091)

The viewer's dev server reaches it through its /oss proxy (vite.config.js).

  /api/runs          runs in runs/ (dirs with cells.jsonl), newest first,
                     and whether each is live (cells.jsonl written < 45 s ago)
  /api/view?run=R&stride=S
                     a sampled view of run R's cells, without sending them: the
                     version `v`, `total` cells, `kept` (every S-th cell plus
                     every cell of an area with at most RARE cells) and the exact
                     summary.json (per-area totals and transitions). 404 when the
                     run has no cells.meta (written by the search since the
                     paged viewer); 409 while the search is mid-write.
  /api/cells?run=R&stride=S&v=V&offset=O&limit=L
                     kept cells O..O+L of that view, as cells.jsonl lines, each
                     with `vp` (the nearest kept ancestor cell, -1 none) and
                     `from` (the area of its parent cell, null none). 410 when
                     version V is gone (start again from /api/view).
  /api/region?run=R&stride=S&v=V&zone=Z&box=x0,y0,z0,x1,y1,z1&max=N
                     the cells of area Z inside the box (from cells.grid),
                     thinned to about N (every k-th of each 256-unit bucket,
                     at least one per bucket), as /api/cells lines; `vp` is the
                     nearest ancestor that is in the base view or in this
                     answer. Headers X-Region-Total (cells in the box) and
                     X-Region-Step (k). 404 without cells.grid.
  /runs/<run>/*      the search's outputs: objectives.json, cells.jsonl, tree.bin

Everything is read from disk on request, so a running search shows up live.
"""

import gzip
import json
import math
import mmap
import struct
import os
import sys
import threading
import time
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def runs():
    out = []
    for d in os.listdir(RUNS) if os.path.isdir(RUNS) else []:
        # A run that has started but not yet written its first snapshot has
        # objectives.json only.
        cells = next((f for f in (os.path.join(RUNS, d, n) for n in ("cells.jsonl", "objectives.json")) if os.path.exists(f)), None)
        if cells:
            t = os.path.getmtime(cells)
            out.append({"name": d, "mtime": t, "live": time.time() - t < 45})
    return sorted(out, key=lambda r: -r["mtime"])


# gzip of the text files (cells.jsonl is tens of MB and shrinks about 8x),
# kept per (file, mtime, size) so polling doesn't recompress it.
_gz = {}


def gzipped(path):
    st = os.stat(path)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _gz.get(path)
    if hit is None or hit[0] != stamp:
        with open(path, "rb") as f:
            hit = (stamp, gzip.compress(f.read(), compresslevel=3))
        _gz[path] = hit
    return hit[1]


# ---------- sampled views of cells.jsonl, from cells.meta ----------

RARE = int(os.environ.get("OSS_RARE", "30000"))  # every cell of an area with at most this many is kept
_views = OrderedDict()  # (run, v, stride) -> View, the most recent few
_views_lock = threading.Lock()


class Gone(Exception):
    pass


class Busy(Exception):
    pass


class View:
    """The kept cells of one version of a run's cells.jsonl. Holds the file
    open (and mapped), so pages keep working after the search replaces it."""

    def __init__(self, run_dir, stride):
        path = os.path.join(run_dir, "cells.jsonl")
        f = open(path, "rb")
        size = os.fstat(f.fileno()).st_size
        with open(os.path.join(run_dir, "cells.meta"), "rb") as m:
            meta = m.read()
        if meta[:4] != b"OSSM":
            raise ValueError("cells.meta: bad magic")
        n = int.from_bytes(meta[8:16], "little")
        if int.from_bytes(meta[16:24], "little") != size:
            raise Busy()  # cells.meta of a newer write than this cells.jsonl
        with open(os.path.join(run_dir, "summary.json"), "rb") as sfile:
            self.summary = json.loads(sfile.read())
        if self.summary.get("cells_jsonl_bytes") != size:
            raise Busy()
        mv = memoryview(meta)
        self.offsets = mv[24:24 + 8 * n].cast("Q")
        self.parents = mv[24 + 8 * n:24 + 12 * n].cast("i")
        self.zones = mv[24 + 12 * n:24 + 16 * n].cast("I")
        self.grid = None
        gpath = os.path.join(run_dir, "cells.grid")
        if int.from_bytes(meta[4:8], "little") >= 2 and os.path.exists(gpath):
            with open(gpath, "rb") as gf:
                g = gf.read()
            if g[:4] == b"OSSG" and int.from_bytes(g[24:32], "little") == size:
                size_u, nb = struct.unpack_from("<II", g, 8)
                buckets = list(struct.iter_unpack("<IhhhHII", g[32:32 + 20 * nb]))
                by_zone = {}
                for j, b in enumerate(buckets):
                    lo, hi = by_zone.get(b[0], (j, j))
                    by_zone[b[0]] = (lo, j + 1)
                self.grid = (size_u, buckets, by_zone, memoryview(g)[32 + 20 * nb:].cast("I"))
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if size else b""
        self.file = f
        self.size, self.n = size, n
        rare = {a["zone"] for a in self.summary["areas"] if a["n"] <= RARE}
        keep = bytearray(n)
        for i in range(0, n, stride):
            keep[i] = 1
        if rare:
            zones = self.zones
            for i in range(n):
                if zones[i] in rare:
                    keep[i] = 1
        self.kept = [i for i in range(n) if keep[i]] if rare else list(range(0, n, stride))
        self.keep = keep

    def line(self, i):
        start = self.offsets[i]
        end = self.offsets[i + 1] - 1 if i + 1 < self.n else self.size - 1
        return self.mm[start:end]

    def page(self, offset, limit):
        parents, zones, keep = self.parents, self.zones, self.keep
        out = []
        for i in self.kept[offset:offset + limit]:
            p = parents[i]
            q = p
            while q >= 0 and not keep[q]:
                q = parents[q]
            frm = str(zones[p]).encode() if p >= 0 else b"null"
            line = self.line(i)
            out.append(line[:-1] + b',"vp":' + str(q).encode() + b',"from":' + frm + b"}")
        return b"\n".join(out) + (b"\n" if out else b"")


    def region(self, zone, box, cap):
        """Cells of `zone` in `box` thinned to about `cap`: (lines, total, step)."""
        size, buckets, by_zone, ids = self.grid
        lo, hi = by_zone.get(zone, (0, 0))
        b0 = [math.floor(v / size) for v in box[:3]]
        b1 = [math.floor(v / size) for v in box[3:]]
        hits = [(b[5], b[6]) for b in buckets[lo:hi]
                if b0[0] <= b[1] <= b1[0] and b0[1] <= b[2] <= b1[1] and b0[2] <= b[3] <= b1[2]]
        total = sum(c for _, c in hits)
        step = max(1, math.ceil(total / max(1, cap)))
        picks = []
        for start, count in hits:
            picks.extend(ids[start:start + count][::step].tolist())
        picks.sort()
        inset = set(picks)
        parents, zones, keep = self.parents, self.zones, self.keep
        out = []
        for i in picks:
            p = parents[i]
            q = p
            while q >= 0 and not keep[q] and q not in inset:
                q = parents[q]
            frm = str(zones[p]).encode() if p >= 0 else b"null"
            out.append(self.line(i)[:-1] + b',"vp":' + str(q).encode() + b',"from":' + frm + b"}")
        return b"\n".join(out) + (b"\n" if out else b""), total, step


def version(run_dir):
    st = os.stat(os.path.join(run_dir, "cells.jsonl"))
    return f"{st.st_size}-{st.st_mtime_ns}"


def view(run, stride, v=None):
    run_dir = os.path.join(RUNS, os.path.basename(run))
    if not os.path.exists(os.path.join(run_dir, "cells.meta")):
        raise FileNotFoundError(run)
    cur = version(run_dir)
    want = v or cur
    with _views_lock:
        hit = _views.get((run, want, stride))
        if hit:
            _views.move_to_end((run, want, stride))
            return want, hit
    if want != cur:
        raise Gone()
    built = View(run_dir, stride)
    if version(run_dir) != cur:
        raise Busy()
    with _views_lock:
        _views[(run, cur, stride)] = built
        while len(_views) > 6:
            _views.popitem(last=False)
    return cur, built


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", gz=False):
        if gz and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, compresslevel=1)
        else:
            gz = False
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, val in getattr(self, "extra", {}).items():
            self.send_header(k, val)
        self.send_header("Access-Control-Expose-Headers", "X-Region-Total, X-Region-Step")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def compressible(self, path):
        """JSON, and cells.jsonl up to 200 MB (bigger ones would take seconds to
        compress; the paged /api/cells serves those)."""
        if path.endswith(".json"):
            return True
        f = self.translate_path(path)
        return path.endswith(".jsonl") and os.path.isfile(f) and os.path.getsize(f) < 200 << 20

    def api_view(self, q, page):
        run = q.get("run", [""])[0]
        stride = max(1, int(q.get("stride", ["1"])[0]))
        try:
            v, vw = view(run, stride, q.get("v", [None])[0] if page else None)
        except FileNotFoundError:
            return self.send(404, b'{"error":"no cells.meta"}')
        except Busy:
            return self.send(409, b'{"error":"the search is writing; try again"}')
        except Gone:
            return self.send(410, b'{"error":"that version is gone"}')
        if not page:
            body = json.dumps({"v": v, "total": vw.n, "kept": len(vw.kept), "stride": stride, "summary": vw.summary,
                               "grid": vw.grid[0] if vw.grid else None}).encode()
            return self.send(200, body, gz=True)
        if page == "region":
            if not vw.grid:
                return self.send(404, b'{"error":"no cells.grid"}')
            box = [float(t) for t in q.get("box", [""])[0].split(",")]
            if len(box) != 6:
                return self.send(400, b'{"error":"box=x0,y0,z0,x1,y1,z1"}')
            body, total, step = vw.region(int(q.get("zone", ["0"])[0]), box, min(200000, int(q.get("max", ["50000"])[0])))
            self.extra = {"X-Region-Total": str(total), "X-Region-Step": str(step)}
            return self.send(200, body, "application/x-ndjson", gz=True)
        offset = int(q.get("offset", ["0"])[0])
        limit = min(100000, int(q.get("limit", ["20000"])[0]))
        self.send(200, vw.page(offset, limit), "application/x-ndjson", gz=True)

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        if path in ("/api/view", "/api/cells", "/api/region"):
            return self.api_view(parse_qs(url.query), {"/api/view": None, "/api/cells": "cells", "/api/region": "region"}[path])
        if path == "/api/runs":
            body = json.dumps(runs()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/runs/") and self.compressible(path) and "gzip" in self.headers.get("Accept-Encoding", ""):
            f = self.translate_path(path)
            if not os.path.isfile(f):
                return self.send_error(404)
            body = gzipped(f)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/runs/"):
            super().do_GET()
        else:
            self.send_error(404)

    def do_HEAD(self):
        if urlparse(self.path).path.startswith("/runs/"):
            super().do_HEAD()
        else:
            self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
    print(f"objective-scattershot data on http://localhost:{port} (GUI: http://localhost:5173/sm64-viewer/?oss=)")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
