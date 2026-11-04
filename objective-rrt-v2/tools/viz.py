#!/usr/bin/env python3
"""Live tree visualiser for Objective-based RRT runs.

    tools/viz.py [port]          (default 8090)

The page itself is part of the replay viewer (sm64-viewer, ?tree=), which
reaches this server through its dev server's /obrrt proxy; / redirects there
and /standalone serves the original page (viz/index.html, the source that
tools/port_tree.py ports into the viewer).

Serves viz/index.html and, per run in runs/:
  /api/runs                          runs, newest first, and whether each is live
  /api/levels                        level number -> name (from core/src/layout.rs)
  /runs/<run>/nodes.csv|summary.json the engine's outputs (rewritten every 30 s)
  /api/node?run=&id=                 the objective that made a state and the
                                     attempts made from it (from events.jsonl)
  /api/m64?run=&node=                a replay from power-on to that state (tree.bin)
  /api/geometry?level=&area=         the area's collision mesh, from the decomp

Everything is read from disk on request, so a running search shows up live.
"""

import json
import os
import re
import struct
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECOMP = os.path.normpath(os.path.join(ROOT, "../../sm64-binds/sm64-port"))
RUNS = os.path.join(ROOT, "runs")
VIZ = os.path.join(ROOT, "viz")


def levels():
    names = {}
    with open(os.path.join(ROOT, "core/src/layout.rs")) as f:
        for m in re.finditer(r"pub const LEVEL_(\w+): u32 = (\d+);", f.read()):
            names[int(m[2])] = m[1].replace("_", " ").title()
    return names


class Events:
    """events.jsonl indexed incrementally (it only grows): per node, the
    event that added it and the attempts that started from it."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.made = {}
        self.tried = {}
        self.lock = threading.Lock()

    def update(self):
        with self.lock:
            try:
                size = os.path.getsize(self.path)
            except OSError:
                return
            if size < self.offset:  # the run was restarted
                self.__init__(self.path)
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
            end = data.rfind(b"\n") + 1
            self.offset += end
            for line in data[:end].splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                ev = {k: e.get(k) for k in ("kind", "key", "success", "added", "frames", "progress", "chain", "stage", "len", "died")}
                frm = e.get("from", e["start"] if e.get("stage", 0) == 0 else None)
                if frm is not None:
                    self.tried.setdefault(frm, []).append(ev)
                if e.get("added") is not None:
                    self.made[e["added"]] = dict(ev, start=frm)


GEOMETRY = {}


def geometry(level, area):
    """An area's collision mesh from the decomp (levels/<dir>/areas/<n>/
    collision.inc.c): vertices, triangles with their surface type, and water
    boxes. In game units, the same space as Mario's position."""
    key = (level, area)
    if key in GEOMETRY:
        return GEOMETRY[key]
    name = levels().get(level, "").lower().replace(" ", "_")
    folder = {"castle": "castle_inside"}.get(name, name)
    path = os.path.join(DECOMP, "levels", folder, "areas", str(area), "collision.inc.c")
    out = {"v": [], "t": [], "s": [], "types": [], "water": []}
    if os.path.exists(path):
        types = {}
        surface = 0
        with open(path) as f:
            for line in f:
                m = re.match(r"\s*(COL_\w+)\(([^)]*)\)", line)
                if not m:
                    continue
                op, args = m[1], [a.strip() for a in m[2].split(",") if a.strip()]
                if op == "COL_VERTEX":
                    out["v"] += [int(a, 0) for a in args]
                elif op == "COL_TRI_INIT":
                    surface = types.setdefault(args[0], len(types))
                elif op in ("COL_TRI", "COL_TRI_SPECIAL"):
                    out["t"] += [int(a, 0) for a in args[:3]]
                    out["s"].append(surface)
                elif op == "COL_WATER_BOX":
                    x1, z1, x2, z2, y = (int(a, 0) for a in args[1:6])
                    out["water"].append([x1, z1, x2, z2, y])
        out["types"] = sorted(types, key=types.get)
    GEOMETRY[key] = out
    return out


EVENTS = {}


def events(run):
    ev = EVENTS.setdefault(run, Events(os.path.join(RUNS, run, "events.jsonl")))
    ev.update()
    return ev


def m64(run, node):
    """Boot inputs plus every input on the path to `node`, raw (4 bytes a frame)."""
    with open(os.path.join(RUNS, run, "tree.bin"), "rb") as f:
        data = f.read()
    (count,) = struct.unpack_from("<I", data, 0)
    pos, recs = 4, []
    for _ in range(count):
        parent, n = struct.unpack_from("<iI", data, pos)
        recs.append((parent, pos + 8, n))
        pos += 8 + 4 * n
    chain, i = [], node
    while i >= 0:
        chain.append(i)
        i = recs[i + 1][0]
    out = [data[recs[0][1] : recs[0][1] + 4 * recs[0][2]]]
    for i in reversed(chain):
        _, at, n = recs[i + 1]
        out.append(data[at : at + 4 * n])
    return b"".join(out)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                # The tree view lives in the replay viewer now; this server is its data.
                host = (self.headers.get("Host") or "localhost").split(":")[0]
                self.send_response(302)
                self.send_header("Location", f"http://{host}:5173/sm64-viewer/?tree=")
                self.end_headers()
                return
            if u.path in ("/standalone", "/index.html"):
                with open(os.path.join(VIZ, "index.html"), "rb") as f:
                    return self.send(f.read(), "text/html; charset=utf-8")
            if u.path == "/api/runs":
                runs = []
                for d in os.listdir(RUNS):
                    nodes = os.path.join(RUNS, d, "nodes.csv")
                    if os.path.exists(nodes):
                        t = os.path.getmtime(nodes)
                        runs.append({"name": d, "mtime": t, "live": time.time() - t < 45})
                runs.sort(key=lambda r: -r["mtime"])
                return self.send(runs)
            if u.path == "/api/levels":
                return self.send(levels())
            if u.path == "/api/geometry":
                return self.send(geometry(int(q["level"]), int(q["area"])))
            if u.path == "/api/node":
                ev = events(q["run"])
                i = int(q["id"])
                return self.send({"made": ev.made.get(i), "tried": ev.tried.get(i, [])})
            if u.path == "/api/m64":
                return self.send(m64(q["run"], int(q["node"])), "application/octet-stream")
            if u.path.startswith("/runs/"):
                return super().do_GET()
        except (KeyError, ValueError, OSError, IndexError) as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(e).encode())
            return
        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
