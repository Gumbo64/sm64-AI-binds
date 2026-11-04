"""Writes a search as an objective-rrt run directory, so sm64-viewer's tree
view (objective-rrt/tools/viz.py) shows it live: nodes.csv, tree.bin and
summary.json, each replaced atomically (temp file + rename) so a reader never
sees half a file.

tree.bin: u32 count, then per record i32 parent (-1 for the boot), u32 n and
n inputs of 4 bytes (big-endian buttons, stick x, stick y); record 0 is the
boot, record i+1 node i. nodes.csv: objective-rrt's columns (unused ones 0).
"""

import io
import json
import os
import struct
import time

import numpy as np

COLS = ["id", "parent", "depth", "x", "y", "z", "level", "area", "stars", "keys", "coins", "frame", "via",
        "as_b", "as_start", "successes", "added", "barren", "hspeed", "fwd_vel", "action"]


def _atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _be(pads):
    p = np.asarray(pads, np.uint32)
    return np.stack([(p >> 24) & 0xFF, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF], -1).astype(np.uint8).tobytes()


class LiveRun:
    def __init__(self, path, boot_pads, every=2.0):
        os.makedirs(path, exist_ok=True)
        self.path, self.every = path, every
        self.boot = np.asarray(boot_pads, np.uint32)
        self.parent, self.pads, self.rows = [], [], []
        self.depth = []
        self.t0 = time.time()
        self.last = 0.0
        self.summary = {}

    def add(self, parent, pads, st, via, frame):
        """Adds a node: parent id (-1 for a child of the boot), the inputs from
        the parent, its status (dict of scalars: pos, level, ...)."""
        i = len(self.parent)
        self.parent.append(parent)
        self.pads.append(np.asarray(pads, np.uint32))
        d = 0 if parent < 0 else self.depth[parent] + 1
        self.depth.append(d)
        p = st["pos"]
        hs = float(np.hypot(st["vel"][0], st["vel"][2]))
        self.rows.append(f"{i},{parent},{d},{p[0]:.0f},{p[1]:.0f},{p[2]:.0f},{st['level']},{st['area']},{st['stars']},"
                         f"{st['keys']},{st['coins']},{frame},{via},0,0,0,0,0,{hs:.1f},{st['forward_vel']:.1f},{st['action']:#x}")
        return i

    def flush(self, force=False, **summary):
        self.summary.update(summary)
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        _atomic(os.path.join(self.path, "nodes.csv"), (",".join(COLS) + "\n" + "\n".join(self.rows) + "\n").encode())
        b = io.BytesIO()
        b.write(struct.pack("<I", len(self.parent) + 1))
        b.write(struct.pack("<iI", -1, len(self.boot)))
        b.write(_be(self.boot))
        for par, pads in zip(self.parent, self.pads):
            b.write(struct.pack("<iI", par, len(pads)))
            b.write(_be(pads))
        _atomic(os.path.join(self.path, "tree.bin"), b.getvalue())
        s = dict(self.summary)
        s.setdefault("nodes", len(self.parent))
        s["nodes"] = len(self.parent)
        s["secs"] = round(now - self.t0, 1)
        s.setdefault("kinds", {})
        _atomic(os.path.join(self.path, "summary.json"), json.dumps(s, indent=1).encode())

    def path_pads(self, node):
        """All inputs from power-on to `node`."""
        chain = []
        while node >= 0:
            chain.append(self.pads[node])
            node = self.parent[node]
        return np.concatenate([self.boot] + chain[::-1]) if chain else self.boot.copy()
