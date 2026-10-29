#!/bin/bash
# The mode / technique policies on (OBRRT_MOVES=1 OBRRT_MOVE_GA=1
# OBRRT_MOVES_MODES=1, base env): determinism (seed 1, 10 M frames, 2 vs 4
# threads: nodes.csv / events.jsonl / tree.bin must match), then a 20 M
# smoke (4 threads) whose summary.json move_ga stats are printed per mode
# policy.
#   tools/modes_det_smoke.sh <bin> [tag]
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BIN=${1:?bin}; TAG=${2:-ms}
MW="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
ON="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVES_MODES=1"
ok=0
for t in 4 2; do
  out=runs/md-$TAG-det-t$t; rm -rf $out
  env $MW $ON OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=10000000 RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM $out 1 60 > logs/md-$TAG-det-t$t.log 2>&1
done
for f in nodes.csv events.jsonl tree.bin; do
  if cmp -s runs/md-$TAG-det-t2/$f runs/md-$TAG-det-t4/$f; then echo "det $f identical"; else echo "det $f DIFFERS"; ok=1; fi
done
echo "det nodes: $(($(wc -l < runs/md-$TAG-det-t4/nodes.csv) - 1))"
out=runs/md-$TAG-smoke; rm -rf $out
env $MW $ON OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=20000000 RAYON_NUM_THREADS=4 nice -n 10 $BIN $ROM $out 1 60 > logs/md-$TAG-smoke.log 2>&1
python3 - "$out/summary.json" <<'P'
import json,sys
s=json.load(open(sys.argv[1]))
m=s.get('move_ga',{})
mode=('cannon-','swim-','fly-','slide-','pole-','speed-kick','edge-dive','high-dive')
print('smoke: attempts',m.get('attempts'),'stars',s.get('stars',s.get('best_stars')),'nodes',s.get('nodes'))
for k,e in sorted(m.get('policy',{}).items()):
    if k.startswith(mode):
        print(f"  {k}: considered {e['considered']} pruned {e['pruned']} simulated {e['simulated']} gen0_best {e['gen0_best']} wins {e['wins']} lineage {e['lineage_wins']} repolicy {e['repolicy']}/{e['repolicy_wins']}")
P
exit $ok
