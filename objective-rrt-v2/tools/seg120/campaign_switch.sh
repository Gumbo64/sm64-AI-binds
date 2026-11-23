#!/bin/bash
# Restart the campaign between steps with another build or settings: waits for the running step to be
# recorded, stops the campaign's process group, clears its skip list, starts it again.
#   [SKIP=12,24] tools/seg120/campaign_switch.sh NAME "campaign.py arguments after NAME"
# SKIP: levels the restarted campaign starts with on its skip list (default none); SKIP=keep leaves the list as it is.
# UNSKIP=29,22: levels taken off the list after that. PATCH=file.py: Python run on the state `s` before the restart. COINUNSKIP=9: levels taken off the coin-legs skip list.
cd "$(dirname "$0")/../.."
name=$1; args=$2
step() { python3 -c "import json; print(json.load(open('logs/campaign/$name/state.json'))['step'])"; }
old=$(cat logs/seg120/camp-$name.pid); s0=$(step)
while kill -0 $old 2>/dev/null && [ "$(step)" = "$s0" ]; do timeout 5 tail -f /dev/null; done
kill -- -$old 2>/dev/null; timeout 3 tail -f /dev/null
python3 - <<P
import json
p='logs/campaign/$name/state.json'; s=json.load(open(p)); k='${SKIP:-}'; s['skip']=s['skip'] if k=='keep' else [int(x) for x in k.split(',') if x]; s['skip']=[l for l in s['skip'] if str(l) not in '${UNSKIP:-}'.split(',')]; s['mult']=1.0; s['coin_skip']=[l for l in s.get('coin_skip',[]) if str(l) not in '${COINUNSKIP:-}'.split(',')]
exec(open('${PATCH}').read()) if '${PATCH:-}' else None
json.dump(s, open(p,'w'), indent=1)
P
echo "$(date '+%m-%d %H:%M:%S') restart at step $(step): $args" >> logs/campaign/$name/log.txt
setsid nohup bash -c "echo \$\$ > logs/seg120/camp-$name.pid; exec python3 tools/seg120/campaign.py $name $args" > logs/seg120/camp-$name.out 2>&1 &
