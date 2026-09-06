#!/usr/bin/env bash
# ============================================================
# RLVD Central Server — Backup PostgreSQL + MinIO evidence
#
# Backup:
#   ./scripts/backup_central.sh [output_dir]
# Restore (Postgres):
#   docker exec -i <postgres-container> psql -U rlvd -d rlvd_central < backup_XXXX/violations.sql
# Restore (MinIO):
#   docker run --rm -v <output_dir>/minio:/data -v rlvd_full_minio_data:/target minio/mc \
#     cp --recursive /data/ local/target/
#
# Yêu cầu: stack đang chạy qua docker-compose.full.yml.
# ============================================================
set -euo pipefail

OUT_DIR="${1:-./backups}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$OUT_DIR/backup_$STAMP"
mkdir -p "$DEST"

COMPOSE_FILE="$(dirname "$0")/../docker-compose.full.yml"
cd "$(dirname "$0")/.."

echo "== RLVD backup -> $DEST"

# ---- PostgreSQL: pg_dump custom format (nén, restore chọn bảng được) ----
docker compose -f "$COMPOSE_FILE" exec -T postgres \
  pg_dump -U "${DB_USER:-rlvd}" -d "${DB_NAME:-rlvd_central}" -Fc \
  > "$DEST/rlvd_central.dump"
echo "PostgreSQL: $(du -h "$DEST/rlvd_central.dump" | cut -f1)"

# ---- MinIO: copy bucket 'violations' ra volume trung gian ----
# Dùng mc (MinIO client) container — không cần cài trên host.
docker compose -f "$COMPOSE_FILE" exec -T minio sh -c \
  "mc alias set local http://localhost:9000 \$MINIO_ROOT_USER \$MINIO_ROOT_PASSWORD >/dev/null && mc mirror local/violations /backup" \
  2>/dev/null || {
    # Fallback: tar trực tiếp volume (cần root或有 docker volume)
    VOL="$(docker compose -f "$COMPOSE_FILE" config --format json 2>/dev/null \
      | python3 -c 'import json,sys;print(json.load(sys.stdin)["volumes"])' 2>/dev/null || true)"
    echo "mc mirror failed — fallback tar volume minio_data"
    docker run --rm -v "$(docker volume ls --format '{{.Name}}' | grep minio_data | head -1):/data:ro" \
      -v "$PWD/$DEST":/backup alpine tar czf /backup/minio_violations.tar.gz -C /data .
  }

echo "== Backup done: $DEST"
echo "Giữ tối đa 14 bản gần nhất:"
ls -1dt "$OUT_DIR"/backup_* 2>/dev/null | tail -n +15 | xargs -r rm -rf
echo "OK"
