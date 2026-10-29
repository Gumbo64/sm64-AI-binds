#!/bin/bash
# tools/bench_replay.sh <replay_max> [seconds] [threads]: cells found and saved-state
# memory at (seconds - 3) s, interpolated, with OSS_REPLAY_MAX set. Prints "<replay_max> <cells> <MB>".
# Run from objective-scattershot/. Writes runs/bench_<n>/ (delete afterwards).
n=$1; secs=${2:-30}; threads=${3:-8}
sed "s/\"zero_star\"/\"zero_star\"/" schemas/zero_star.json > schemas/bench_$n.json
OSS_REPLAY_MAX=$n RAYON_NUM_THREADS=$threads nice -n 19 ./target/release/examples/oss schemas/bench_$n.json $(python3 -c "print($secs/60)") 0 --noise pink 2>&1 \
  | grep -E "^\[" | sed -E "s/^\[([0-9]+)s\] cells ([0-9]+) .*saves ([0-9]+)MB.*/\1 \2 \3/" \
  | awk -v n=$n -v T=$((secs-3)) '{ if ($1<=T) {pt=$1; pc=$2; pm=$3} else if (!d) {d=1; f=(T-pt)/($1-pt); printf "%d %d %d\n", n, pc+f*($2-pc), pm+f*($3-pm)} } END { if (!d) printf "%d %d %d\n", n, pc, pm }'
rm -r runs/bench_$n; rm schemas/bench_$n.json
