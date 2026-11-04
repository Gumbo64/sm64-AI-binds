#!/bin/bash
# Keeps king_loop.py alive: if it ever dies, it restarts from logs/kingloop/state.json. Stop: kill -- -$(cat logs/king_loop.pid)
cd "$(dirname "$0")/../.."
while true; do python3 tools/queues/king_loop.py >> logs/king_loop.out 2>&1; echo "$(date) king_loop exited $?, restarting" >> logs/king_loop.txt; timeout 60 tail -f /dev/null; done
