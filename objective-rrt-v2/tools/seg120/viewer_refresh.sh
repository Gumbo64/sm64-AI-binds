#!/bin/bash
# Keeps a campaign's viewer entry current while the campaign runs: the progress file (stars by samples, for the live
# chart; cheap) every 20 s, and the whole entry (a replay of the route: star board, regions) when a step has ended,
# at most every 5 minutes. PID in logs/viewer/refresh-NAME.pid; stop with: kill $(cat logs/viewer/refresh-NAME.pid)
cd "$(dirname "$0")/../.."
name=${1:-t1}
echo $$ > logs/viewer/refresh-$name.pid
step() { python3 -c "import json; print(json.load(open('logs/campaign/$name/state.json'))['step'])" 2>/dev/null; }
last=; at=0
while kill -0 "$(cat logs/seg120/camp-$name.pid)" 2>/dev/null; do
  python3 tools/seg120/progress_json.py "$name" >> logs/viewer/refresh-$name.log 2>&1
  s=$(step); t=$(date +%s)
  if [ -n "$s" ] && [ "$s" != "$last" ] && [ $((t - at)) -ge 300 ]; then
    nice -n 15 python3 tools/seg120/viewer_publish.py "$name" >> logs/viewer/refresh-$name.log 2>&1 && { last=$s; at=$t; }
  fi
  timeout 20 tail -f /dev/null
done
python3 tools/seg120/progress_json.py "$name" >> logs/viewer/refresh-$name.log 2>&1
