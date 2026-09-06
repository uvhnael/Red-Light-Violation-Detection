#!/usr/bin/env bash
# Xuất toàn bộ diagram .mmd trong docs/diagrams/ sang PNG qua Kroki API.
# Yêu cầu: curl. Output: docs/diagrams/<tên>.png (nền trắng).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR"
FAIL=0
shopt -s nullglob
for mmd in "$DIR"/*.mmd; do
    name="$(basename "$mmd" .mmd)"
    ok=0
    for attempt in 1 2 3; do
        if curl -sf --max-time 60 -X POST -H "Content-Type: text/plain" \
             --data-binary @"$mmd" "https://kroki.io/mermaid/png" -o "$OUT/$name.png"; then
            if [ -s "$OUT/$name.png" ] && [ "$(head -c4 "$OUT/$name.png" | file -b - | grep -c PNG)" = "1" ]; then
                ok=1; break
            fi
        fi
        sleep 2
    done
    if [ "$ok" = 1 ]; then
        echo "OK  $name.png ($(du -h "$OUT/$name.png" | cut -f1))"
    else
        echo "FAIL $name"
        rm -f "$OUT/$name.png"
        FAIL=1
    fi
done
exit $FAIL
