# RLVD — Changelog (Pre-Production Audit 2026-09-06)

Audit toàn diện theo `pre_production.md`. Mọi thay đổi được liệt kê với: vấn đề → nguyên nhân → giải pháp → file.

## Fixed (đợt audit này)

### 1. [HIGH] Hard-code credentials hạ tầng trong docker-compose.full.yml
- **Vấn đề**: `POSTGRES_PASSWORD: rlvd_secret`, `MINIO_ROOT_USER/PASSWORD: minioadmin`, `DB_PASS`, `MINIO_ACCESS_KEY/SECRET_KEY` hard-code — stack production sẽ chạy với mật khẩu mặc định công khai.
- **Nguyên nhân**: compose viết thời dev chưa parameterize.
- **Giải pháp**: chuyển toàn bộ sang `${VAR:-default}` — dev vẫn chạy được với default, production đặt trong `.env`.
- **File**: `docker-compose.full.yml` (postgres, minio, central-server).

### 2. [HIGH] getViolationsByStatus quét toàn bảng
- **Vấn đề**: `repository.findAll().stream().filter(...)` tải TOÀN BỘ bảng violations vào RAM rồi mới lọc status — bảng vài trăm nghìn dòng sẽ treo request.
- **Nguyên nhân**: code viết trước khi có index idx_status + paged query; không ai rà lại khi DB lớn dần.
- **Giải pháp**: dùng `repository.findByStatus(normalized)` — query có index.
- **File**: `central_server/.../service/ViolationService.java`.
- **Tại sao phù hợp production**: endpoint này là query path chính (lọc theo status); IO giảm từ O(toàn bảng) → O(kết quả).

### 3. [MEDIUM→HIGH] normalizeStatus cho chuỗi tuỳ ý vào DB
- **Vấn đề**: `default -> normalized` cho phép status bất kỳ (VD "hacked") ghi thẳng vào DB qua PATCH status.
- **Giải pháp**: default → "pending"; status chỉ ∈ {pending, approved, rejected}.
- **File**: `ViolationService.java`.

### 4. [MEDIUM] Batch ingest check duplicate tải cả entity
- **Vấn đề**: `findByEventId(...).isPresent()` mỗi item trong batch — SELECT full row chỉ để check tồn tại.
- **Giải pháp**: thêm `existsByEventId` vào repository, dùng cho cả single lẫn batch — query boolean nhẹ hơn.
- **File**: `ViolationRepository.java`, `ViolationService.java`.

### 5. [MEDIUM] Không có connection pool tuning + graceful shutdown
- **Vấn đề**: HikariCP dùng mặc định không khai báo trần; tắt central đang đứt request giữa chừng (batch ingest, upload media).
- **Giải pháp**: khai báo `hikari.maximum-pool-size=${DB_POOL_MAX:10}`, `minimum-idle=2`, timeouts; bật `server.shutdown: graceful` + `spring.lifecycle.timeout-per-shutdown-phase: 30s`.
- **File**: `central_server/src/main/resources/application.yml`.

### 6. [HIGH] Không có phương án backup/restore
- **Vấn đề**: checklist production readiness thiếu hoàn toàn backup — mất volume postgres_data là mất toàn bộ hồ sơ vi phạm.
- **Giải pháp**: thêm `scripts/backup_central.sh` — pg_dump custom format + mirror bucket MinIO, tự giữ 14 bản gần nhất, hướng dẫn restore trong header.
- **File**: `scripts/backup_central.sh` (mới, syntax check PASS, compose config validate PASS).

### 7. [LOW] render.sh diagram lỗi biến vòng lặp
- **Vấn đề**: glob rỗng khiến `--data-binary @` đọc file literal `*`.
- **Giải pháp**: `shopt -s nullglob` + xác nhận PNG magic bytes; render qua Kroki GET (deflate+base64).
- **File**: `docs/diagrams/render.sh`.

## Added (tài liệu + diagram)

- `docs/diagrams/*.mmd` + `.png` — 10 diagram Mermaid render thành công:
  system-architecture, three-tier, deployment, api-surface, erd, use-case,
  edge-lifecycle, sequence-edge-data, sequence-auth, sequence-calibration.
- `docs/test-report.md` — báo cáo kiểm thử với số liệu thực tế.
- `docs/deployment/production-guide.md` — hướng dẫn triển khai production.
- `docs/api/README.md` — tài liệu API đầy đủ.
- `docs/production-readiness.md` — checklist + chấm điểm.
- `docs/final-report.docx` — báo cáo đồ án hoàn chỉnh (Times New Roman).

## Verified (không cần sửa — bằng chứng trong test-report)

- 85/85 pytest PASS; Maven test-compile PASS (Docker chính thức); tsc + next build PASS.
- Không có secret hard-code trong source; `.env` gitignore đúng.
- Kiến trúc 3 tầng tách biệt sạch: web không import pg/genai; service không phụ thuộc controller; không SQL trong controller.
- Durable outbox + idempotent ingest (event_id UNIQUE 2 phía) — chịu mất mạng/restart, không duplicate.
