# RLVD — Hướng dẫn triển khai Production

## 1. Yêu cầu hạ tầng

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| GPU (edge) | — | NVIDIA + nvidia-container-toolkit (YOLO FP16 ~73 FPS) |
| Disk | 50 GB | 200 GB+ (ảnh bằng chứng MinIO) |
| OS | Linux 64-bit (Docker) | Ubuntu 22.04+ |

Phần mềm: Docker + Docker Compose plugin; GPU cần nvidia-container-toolkit
(`sudo bash scripts/install_nvidia_container_toolkit.sh`).

## 2. Cấu hình môi trường (BƯỚC BẮT BUỘC)

Tạo `.env` ở thư mục gốc (mẫu đầy đủ trong `.env.sample` ở root — file .env duy nhất dùng chung cho cả 3 compose):

```bash
# SINH SECRET AN TOÀN:
openssl rand -base64 32   # → JWT_SECRET (≥32 ký tự, BẮT BUỘC)
openssl rand -hex 16      # → INGEST_TOKEN
openssl rand -hex 16      # → EDGE_API_TOKEN
openssl rand -base64 24   # → DB_PASS
openssl rand -base64 24   # → MINIO_ROOT_PASSWORD / MINIO_SECRET_KEY

DB_PASS=<mật khẩu mạnh>          # KHÔNG dùng default rlvd_secret
MINIO_ROOT_USER=rlvdadmin
MINIO_ROOT_PASSWORD=<mật khẩu mạnh>
MINIO_ACCESS_KEY=rlvdadmin      # phải khớp MINIO_ROOT_USER
MINIO_SECRET_KEY=<mật khẩu mạnh>  # phải khớp MINIO_ROOT_PASSWORD

JWT_SECRET=<≥32 ký tự ngẫu nhiên>
INGEST_TOKEN=<ngẫu nhiên>        # edge gửi hồ sơ qua header X-Ingest-Token
EDGE_API_TOKEN=<ngẫu nhiên>      # central forward khi gọi /action/* trên edge
EDGE_REQUIRE_TOKEN=true          # BẬT guard cho control-plane edge
EDGE_ALLOWED_ORIGINS=https://your-domain.vn

ADMIN_USERNAME=admin
ADMIN_PASSWORD=<mật khẩu mạnh — KHÔNG dùng admin123>

GEMINI_API_KEY=<AIza... từ aistudio.google.com>
```

Lưu ý bảo mật:
- `JWT_SECRET` thiếu hoặc <32 ký tự → central FAIL-FAST không khởi động (đúng thiết kế).
- `INGEST_TOKEN` trống → ingest mở không xác thực + log cảnh báo (chỉ chấp nhận khi dev).
- `ADMIN_PASSWORD` trống → seed về `admin123` (chỉ dùng dev).

## 3. Build + khởi động

```bash
./start.sh                 # build + up full stack (5 container)
./start.sh minimal         # chỉ web+central+db (không edge/GPU)
./start.sh status          # trạng thái
./start.sh logs            # tail logs
./start.sh rebuild edge-pipeline   # build lại 1 service sau khi sửa code
```

Cổng host: web :3000 · central :8002 · edge :8082 · postgres :5432 · minio :9010/:9011.

Weights model phải có trong `models/` (không nằm trong git):
`yolo26m_vehicle.pt` (bắt buộc), `traffic_light_cls.pt`, `license_plate_yolo26.pt`.

## 4. Reverse proxy + HTTPS (production internet)

Đặt Nginx/Caddy trước web-dashboard, TLS termination:

```nginx
server {
    listen 443 ssl http2;
    server_name rlvd.your-domain.vn;
    ssl_certificate     /etc/letsencrypt/live/.../fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/.../privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
        # HLS stream qua /edge-api — tăng buffer:
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
server {
    listen 80;
    server_name rlvd.your-domain.vn;
    return 301 https://$host$request_uri;
}
```

Sau khi có domain: cập nhật `EDGE_ALLOWED_ORIGINS`, `web` build args trỏ đúng URL nội bộ Docker
(`CENTRAL_SERVER_URL=http://central-server:8000` — đã mặc định trong compose).

## 5. Kiểm tra sau triển khai (verification checklist)

```bash
# 1. Container đủ + healthy
./start.sh status   # central-server (healthy), web-dashboard, postgres, minio, edge-pipeline

# 2. Health endpoints
curl -s http://localhost:8002/api/health | grep ok
curl -s http://localhost:8082/health    # 503 degraded khi chưa có camera thật = bình thường

# 3. Đăng nhập được với mật khẩu mới
curl -s -X POST http://localhost:3000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"<mật khẩu mới>"}' | grep access_token

# 4. Edge đã register
curl -s http://localhost:8002/api/v1/edge-nodes -H "Authorization: Bearer <JWT>" | grep node_id

# 5. Kẻ vạch dừng trên web /nodes/{nodeId} → dòng "violations DISABLED" biến mất trong log edge

# 6. Ingest idempotent: gửi 2 lần cùng event_id → lần 2 duplicates=1, không lỗi
```

## 6. Vận hành thường xuyên

### Backup (hàng ngày + trước mỗi thay đổi lớn)
```bash
./scripts/backup_central.sh           # pg_dump + MinIO mirror → ./backups/backup_YYYYmmdd_HHMMSS
```
Giữ 14 bản tự động. Restore hướng dẫn trong header script.
Lên lịch crontab: `0 2 * * * cd /path/to/RLVD && ./scripts/backup_central.sh >> backups/cron.log 2>&1`

### Restore test (quý)
1. `docker compose -f docker-compose.full.yml down`
2. Xoá volume cũ: `docker volume rm <project>_postgres_data`
3. `docker compose ... up -d postgres` → `cat backup.sql | docker exec -i <pg> psql -U rlvd -d rlvd_central`
4. Verify: `SELECT count(*) FROM violations;`

### Monitoring / Logging
- Mỗi service có `restart: unless-stopped`; healthcheck central mỗi 10s.
- Xem log: `./start.sh logs` hoặc `docker compose -f docker-compose.full.yml logs -f central-server`.
- Edge tự báo `pipeline_stale > 10s → 503` tại `/health` (metrics in-process: frames_processed, violations_detected, fps).
- Chưa có Prometheus/Grafana — giám sát hiện qua Docker healthcheck + log (ghi nhận trong readiness checklist).

### Rollback
```bash
git checkout <commit-trước-đó>
./start.sh                 # rebuild từ code cũ (image tag theo commit)
# DB: backup gần nhất theo hướng dẫn restore ở trên
```

## 7. Bảo mật vận hành

- Đổi toàn bộ secret trong `.env` (không giữ default nào ở mục 2).
- Không expose cổng postgres/minio ra internet — chỉ mở qua firewall/UFW cho docker network.
- Dashboard qua HTTPS duy nhất; JWT 12h + refresh rotation tự động.
- Kiểm tra log `Revoked refresh token reuse detected` — dấu hiệu token bị đánh cắp (hệ thống đã tự revoke all).
- `EDGE_REQUIRE_TOKEN=true` + whitelist origin chặn gọi control-plane từ domain lạ.
