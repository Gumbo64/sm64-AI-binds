#!/bin/bash
# Kills the largest obrrt2/oss2 run when MemAvailable drops below $1 MB (default
# 1500), noting it in logs/memguard.txt; exits when the queue PID file's process ($2) ends.
cd "$(dirname "$0")/.."
LOW=${1:-1500}; Q=${2:?queue pid file}
while kill -0 "$(cat "$Q")" 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$LOW" ]; then
    read -r pid rss args <<<"$(ps -eo pid=,rss=,args= --sort=-rss | grep -E 'obrrt2|oss2' | grep -v grep | head -1)"
    [ -n "$pid" ] && kill "$pid" && echo "$(date -u +%H:%M:%S) avail ${avail} MB: killed $pid (rss $((rss/1024)) MB) $args" >> logs/memguard.txt
    sleep 30
  fi
  sleep 10
done
