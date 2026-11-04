#!/usr/bin/env python3
"""Copies one head's weights from one checkpoint onto another (a test of
how far the shared encoder drifted between them):

    .venv/bin/python graft_head.py <base.pt> <donor.pt> <out.pt> [prefix=pol_out.]

Every parameter whose name starts with `prefix` comes from the donor, the
rest from the base. The default moves only the policy output layer, not the
goal tail it shares with reachability.
"""

import sys

import torch


def main():
    base, donor, out = sys.argv[1:4]
    prefix = sys.argv[4] if len(sys.argv) > 4 else "pol_out."
    b = torch.load(base, map_location="cpu")
    d = torch.load(donor, map_location="cpu")
    moved = [k for k in b["model"] if k.startswith(prefix)]
    if not moved:
        sys.exit(f"no parameters start with {prefix!r}")
    for k in moved:
        b["model"][k] = d["model"][k]
    b["grafted"] = {"prefix": prefix, "from": donor, "donor_step": d["step"], "base_step": b["step"]}
    torch.save(b, out)
    print(f"{out}: {len(moved)} tensors under {prefix!r} from step {d['step']} onto step {b['step']}")


if __name__ == "__main__":
    main()
