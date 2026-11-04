#!/usr/bin/env bash
# Speed-mode evaluation: pipe an obrrt2 run's best route into oss2 and try to
# make it faster for FRAMES emulated frames, then print the milestone table.
#
#   tools/speed_eval.sh RUN_DIR FRAMES SEED
#
# Writes oss2/runs/speed-<run>-s<seed>/ (viewable in sm64-viewer ?oss=; the
# route, fastest path and cells as for any oss run, plus route.m64 and
# fastest.m64, both raw from power-on). Env (defaults in brackets):
#   OSS2_BIN   [oss2/../target/release/oss2]
#   SCHEMA     [oss2/schemas/star120.json]: a Stage with a 256-unit grid in
#              every level and area, mission counters and Root's
#              level/keys/stars; zero_star.json has no stage in the courses
#              (BoB, WF, CCM would be keyed on Root's 1024-unit cubes only).
#   THREADS [2]  GA [1] (OSS_GA)  CORRIDOR [600]  PROP [0.1]  NODE [best]
#   OUT_NAME   [speed-<run>-s<seed>]
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 RUN_DIR FRAMES SEED" >&2; exit 2; }
here=$(cd "$(dirname "$0")/.." && pwd)
run=$(cd "$1" && pwd)
frames=$2
seed=$3
name=${OUT_NAME:-speed-$(basename "$run")-s$seed}
out=$here/runs/$name
bin=${OSS2_BIN:-$here/../target/release/oss2}
schema=${SCHEMA:-$here/schemas/star120.json}
mkdir -p "$out"
echo "$name: $run -> $out ($frames frames, seed $seed, ${THREADS:-2} threads)"
OSS_GA=${GA:-1} OSS_GA_TARGET=route OSS_SELECT=route OSS_CORRIDOR=${CORRIDOR:-600} OSS_PROP=${PROP:-0.1} \
  RAYON_NUM_THREADS=${THREADS:-2} nice -n 10 "$bin" "$schema" 0 "$seed" --frames "$frames" \
  --pipe "$run" ${NODE:+--pipe-node "$NODE"} --out "$out" > "$out/log.txt" 2>&1 \
  || { tail -20 "$out/log.txt" >&2; exit 1; }
python3 - "$out/stats.json" <<'EOF'
import json, sys
st = json.load(open(sys.argv[1]))
s = st["speed"]
fmt = lambda f: "-" if f is None else f"{f // 1800}:{f % 1800 / 30:04.1f}"
print(f"route: {s['pipe']} node {s['node']} ({s['route_frames']} frames); frames {st['frames']}, cells {st['cells']}")
print(f"{'milestone':<18} {'route':>8} {'oss2':>8} {'saved':>7}")
for m in s["milestones"]:
    sv = m["saved_frames"]
    print(f"{m['name']:<18} {fmt(m['route_frame']):>8} {fmt(m['best_frame']):>8} {'-' if sv is None else f'{sv / 30:+.2f}s':>7}")
print(f"final ({s['final']}): saved {s['saved_s'] or 0:.2f} s; route keys faster {s['route_keys_faster']}/{s['route_keys']}"
      f"; improvements {s['improvements_per_mframe']:.1f}/Mframe")
p = s["prop"]
print(f"propagation: queued {p['queued']} tried {p['tried']} improved {p['improved']} added {p['added']} frames {p['frames']} dropped {p['dropped']}")
g = st["ga"]
print(f"ga: expansions {g['expansions']} met {g['met']} improved {g['improved']} fallbacks {g['fallbacks']}")
EOF
