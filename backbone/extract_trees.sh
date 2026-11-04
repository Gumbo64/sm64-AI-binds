#!/bin/bash
# v4 data: states from more objective-rrt trees (sm64_AI_backbone.md 5),
# about 200k each, level-balanced (select_nodes.py), extracted with
# objective-rrt's dataset tool into backbone/data/<run>/.
cd "$(dirname "$0")/.."
for r in "$@"; do
  [ -d backbone/data/$r/s0 ] && { echo "$r: already extracted"; continue; }
  backbone/.venv/bin/python backbone/select_nodes.py objective-rrt/runs/$r backbone/data/select-$r.bin 200000 \
    && objective-rrt/target/release/dataset tree baserom.us.z64 objective-rrt/runs/$r backbone/data/select-$r.bin backbone/data/$r 8 2>&1 | grep -E "done|rror"
  du -sh backbone/data/$r; df -h / | tail -1
done
