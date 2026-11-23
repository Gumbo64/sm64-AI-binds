#!/bin/bash
# A private copy of the obrrt2 sources for one subagent, with its own build dir (release, no debuginfo, only
# obrrt2), so parallel agents don't break each other's builds. Usage: tools/agent_ws.sh NAME -> prints the dir.
#   build:  cd DIR && CARGO_TARGET_DIR=DIR/target CARGO_PROFILE_RELEASE_DEBUG=false CARGO_BUILD_JOBS=2 nice cargo build --release -p obrrt-run --bin obrrt2
set -e
src="$(cd "$(dirname "$0")/.." && pwd)"; dst=/workspaces/sm64-combined/agentws/$1
mkdir -p "$dst"
rsync -a --delete --exclude target --exclude runs --exclude logs --exclude builds --exclude target-agent \
  --exclude 'policy/*' --include 'policy/' --exclude 'oss2/*' --exclude 'agent/*' --exclude pyenv/target --exclude scattershot/target \
  "$src/Cargo.toml" "$src/Cargo.lock" "$src/core" "$src/search" "$src/objectives" "$src/run" "$src/spec" "$src/gp" "$src/pyenv" "$src/scattershot" "$dst/"
ln -sfn "$(readlink -f "$src/ref")" "$dst/ref"
for m in policy oss2 agent; do mkdir -p "$dst/$m"; rsync -a "$src/$m/Cargo.toml" "$src/$m/src" "$dst/$m/"; done
ln -sfn "$src/../rust" /workspaces/sm64-combined/agentws/rust 2>/dev/null || true
echo "$dst"
