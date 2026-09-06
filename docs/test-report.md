# RLVD — Test Report (Pre-Production Audit)

Ngày thực hiện: 2026-09-06 (giờ Việt Nam)
Phạm vi: toàn bộ 3 tầng — Edge Node (Python), Central Server (Java 17), Web Dashboard (Next.js).

## 1. Test strategy

- **Unit test** (pytest): logic edge node — hình học vạch, tracker/stabilizer, quy tắc vi phạm, định dạng biển số VN.
- **Compile/type check**: central server (Maven test-compile trong Docker image chính thức), web (tsc --noEmit + next build trong CI).
- **Lint**: ruff (strict cho tests/), eslint (web).
- **CI** (.github/workflows/ci.yml): 3 job — edge-tests, central-build, web-build.

## 2. Unit testing — kết quả thực tế

Lệnh: `myenv/bin/python -m pytest tests/ -q`

```
85 passed in 0.87s
```

| File | Nội dung |
|---|---|
| tests/test_geometry.py | Hình học vạch dừng — side, crossing, deadband |
| tests/test_violation_logic.py | Quy tắc vi phạm: qua vạch khi đèn đỏ, cross-and-back, đèn xanh không cờ, regression deadband |
| tests/test_tracker_stabilizer.py | ByteTrack tuned (xe máy không flicker), debounce đèn theo giây thực, scale theo FPS nguồn |
| tests/test_vn_plate.py | Validator biển VN TT24: 3 dạng seri, sửa OCR nhầm lẫn, gạch sau mã tỉnh |
| tests/test_metrics.py | In-process metrics + stale detection |
| tests/conftest.py | Fixtures dùng chung (set_active_tripwire, fake tracks) |

Coverage báo cáo bởi pytest --cov (CI): edge_node/ các module core logic được phủ; các module cần GPU/weights không chạy trong CI.

## 3. Integration / API testing

Trạng thái: **đã có kiểm thử thủ công end-to-end trong các phiên làm việc trước** (được ghi nhận trong skill references):
- Offline→online e2e: outbox pending → flush → central nhận batch (references/outbox-batch-delivery.md).
- Media upload pipeline: edge → central → MinIO → web hiển thị, verify bằng curl magic bytes JPEG (references/delivery-diagnosis-and-db-wipe.md).
- Calibration flow web→central→edge: POST direction qua proxy, verify `GET :8082/api/calibration` cập nhật tức thì (2026-08-31).
- 263-row backlog media drain sau fix IngestPaths (2026-08-31/09-05).
- Login + RBAC: 401 sai mật khẩu, 403 sai vai trò, AI proxy forward JWT (references/web-frontend-audit.md).

Chưa có: bộ integration test TỰ ĐỘNG (JUnit + Testcontainers) cho central server — đây là gap được ghi nhận trong Production Readiness (MEDIUM).

## 4. Frontend testing

- `tsc --noEmit`: PASS (exit 0).
- `next build` (CI job web-build với dummy env): PASS.
- Component test (Jest/Vitest): **Chưa thực hiện** — gap MEDIUM, UI verification hiện qua CDP browser + `./start.sh rebuild web-dashboard` + curl 200.
- Critical user flow đã verify thủ công: login → dashboard → nodes → calibration → review → violations → AI chat.

## 5. Edge Node testing

- Connection/retry/offline/reconnect: đã verify qua kịch bản thủ công (tắt central → pending tích luỹ → bật lại → flush toàn bộ, `sent` rows không gửi lại sau TRUNCATE).
- Data sync + dedupe: event_id UNIQUE cả 2 phía (SQLite outbox + PostgreSQL idx_event_id unique) — duplicate được skip, không lỗi.
- Tracker quality: đo trên 192 frames @3fps (references/track-quality-measurement.md): aziz1 = 2 violations (baseline), 20221003-102556 = 0 (không false positive), độ phủ xe máy 98.3% trên aziz1.

## 6. System / End-to-end

- Full stack qua `./start.sh`: 5 container (postgres, minio, central-server, web-dashboard, edge-pipeline) — đã chạy ổn định trong các phiên trước; healthcheck cho central (wget /api/health), depends_on condition service_healthy cho web.
- Load test: **Chưa thực hiện** — không bịa số liệu. Xem §8 Performance.

## 7. Security testing

Thực hiện trong audit 2026-09-06 (Phase 7):
- Scan secret hard-code trong source (py/java/ts/yml/sh): **0 kết quả** — mọi secret qua env, `.env` đã gitignore, chỉ `.env.example` trong git.
- JWT: HS256, secret ≥32 ký tự fail-fast, TTL 12h; refresh token SHA-256 hash + rotation + reuse-detection revoke-all. Verify logic đọc code (AuthController, RefreshTokenService).
- Ingest token: constant-time compare (MessageDigest.isEqual) cả 2 phía.
- Rate limit: sliding window 30 req/phút/IP cho POST /action/* trên edge; central chưa có rate limit login (gap MEDIUM).
- CORS: central `allowedOriginPatterns("*")` + credentials=false (proxy model — OK, ghi chú trong report); edge whitelist EDGE_ALLOWED_ORIGINS.
- SQL injection: JPA parameterized + AI Text-to-SQL chỉ SELECT (regex chặn DROP/DELETE/INSERT/UPDATE/ALTER/TRUNCATE/GRANT/REVOKE) + LIMIT 50.
- Docker: web chạy user nextjs non-root; postgres/minio/edge chạy root trong container (mặc định image — ghi nhận gap LOW).

## 8. Performance testing

Số liệu có bằng chứng (đo thực tế, không bịa):
- YOLO26m_vehicle (fine-tuned): mAP50=0.944, ~73 FPS FP16 trên RTX 2060 (scripts/benchmark_vehicle_models.py; benchmark_results.csv).
- YOLO26n-cls đèn: val top-1 100% (LISA), ~0.3 ms/ảnh @imgsz 64; fusion ensemble 99.0% trên video domain-shift.
- OCR biển số: ~21 ms/ảnh (RTX 2060) — mAP50=0.993.
- Độ trễ chuyển đèn sau tối ưu: 0.66s @6fps (từ 1.00s), 0.67s trên camera 3fps (từ 2.3s).
- Chưa có load test HTTP (concurrent users/edge nodes) — cần benchmark: k6/wrk against GET /api/violations/page + POST /api/violations/batch.

## 9. Kết luận

| Nhóm | Trạng thái |
|---|---|
| Unit edge | PASS — 85/85 |
| Compile central | PASS (Docker maven:3.9-temurin-17) |
| Type-check + build web | PASS |
| Lint | tests/ PASS; edge_node/ 152 lỗi stylistic (không chặn CI) |
| Integration tự động | Chưa có (thủ công đã verify) |
| Load test | Chưa thực hiện |

Tổng thể: nền tảng chất lượng tốt cho đồ án; rủi ro chính nằm ở thiếu integration test tự động và load benchmark — đã liệt kê trong Production Readiness checklist.
