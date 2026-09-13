#!/usr/bin/env bash
# Dev server cho web (Next.js 16) — nạp JWT_SECRET từ .env gốc để proxy.ts
# xác thực được chữ ký HS256 của Central.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
rm -rf .next
exec npx next dev -p "${PORT:-3111}"
