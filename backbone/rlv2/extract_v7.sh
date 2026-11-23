#!/bin/bash
# v7 backbone data: v2-era, fast play. TAS (already in data/tas120: same
# schema), best routes, 80k samples each from two v2 trees. Held out from it:
# runs/q15-pw1-s3 and legs-fast-king-walk-s3 (the follower's test traces).
cd /workspaces/sm64-combined/sm64-AI-binds
D=backbone/rlv2/dataset_v2
V2=objective-rrt-v2
for m in logs/kingloop/best-fast-king-blj-tas.ramp0 logs/kingloop/best-fast-king-tree-d logs/kingloop/best-fast-king-superjump-c oss2/runs/legs-speed-blj-fresh-s1/route oss2/runs/legs-fast-king-walk-s1/route; do
  n=$(echo $m | tr '/' '_')
  [ -d backbone/data/v7/r-$n/s0 ] || $D tas baserom.us.z64 $V2/$m.m64 backbone/data/v7/r-$n 2>&1 | tail -1
done
for r in q15-pw1-s1 q8-g995-s1; do
  [ -d backbone/data/v7/$r/s0 ] && continue
  backbone/.venv/bin/python backbone/select_nodes.py $V2/runs/$r backbone/data/v7/select-$r.bin 80000 \
   && $D tree baserom.us.z64 $V2/runs/$r backbone/data/v7/select-$r.bin backbone/data/v7/$r 3 2>&1 | grep -E "done|rror" | tail -2
done
du -sh backbone/data/v7/*; df -h / | tail -1
