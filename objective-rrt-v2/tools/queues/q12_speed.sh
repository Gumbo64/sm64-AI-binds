#!/bin/bash
# Queue 12 (2026-11-02): oss2 speed mode (builds/oss2b) on obrrt2 best routes,
# 100 M frames each, 8 threads, two at a time. Waits for q11.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q11.pid)" 2>/dev/null; do sleep 30; done
B=$PWD/builds/oss2b/oss2
OSS2_BIN=$B THREADS=8 oss2/tools/speed_eval.sh runs/q6-p-s1 100000000 1 > logs/q12-q6-p-s1.txt 2>&1 &
OSS2_BIN=$B THREADS=8 oss2/tools/speed_eval.sh runs/q8-g995-s2 100000000 1 > logs/q12-g995-s2.txt 2>&1 &
wait
echo "q12 done"
