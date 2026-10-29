#!/bin/bash
# get.sh Page -> raw/<Page>.html + .url ; prints status
D=/tmp/claude-1000/-workspaces-sm64-combined/85a1d413-6634-4235-ba1a-3998b313ad34/scratchpad/starstrat_q7
p="$1"; safe=$(python3 -I -c "import sys,urllib.parse;print(urllib.parse.unquote(sys.argv[1]).replace('/','_').replace('?','').replace(':','_'))" "$p")
out="$D/raw/$safe.html"
[ -s "$out" ] && { echo "cached $safe $(cat "$out.url")"; exit 0; }
for y in 2026 2026 2026 2025 2025 2024 2023 2022 2021; do
  r=$(curl -sS -L --max-time 90 -H "Accept-Encoding: identity" -o "$out.tmp" -w "%{http_code} %{url_effective}" "https://web.archive.org/web/$y/https://ukikipedia.net/wiki/$p" 2>&1)
  code=${r%% *}; url=${r#* }
  if [ "$code" = 200 ] && grep -q 'mw-parser-output' "$out.tmp"; then mv "$out.tmp" "$out"; echo "$url" > "$out.url"; echo "ok $safe $url"; exit 0; fi
  if [ "$code" = 000 ] || [ "$code" = 429 ] || [ "$code" = 503 ]; then
    n=$((${n:-0}+1)); [ $n -le 5 ] && { sleep $((15*n)); r=retry; }
  fi
  [ "$code" = 404 ] && [ "$y" = 2024 ] && break
  sleep 4
done
rm -f "$out.tmp"; echo "MISSING $p $code"
