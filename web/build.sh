#!/usr/bin/env bash
# Build production cho web (Next.js 16) — nạp JWT_SECRET từ .env gốc để
# proxy.ts xác thực được chữ ký HS256 lúc build/collect page data.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
exec npx next build
