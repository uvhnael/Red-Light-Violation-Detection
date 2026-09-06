# RLVD — Production Readiness Checklist (2026-09-06)

Chấm điểm theo từng mục, tổng 100 điểm. CRITICAL chưa giải quyết ⇒ KHÔNG kết luận production ready.

## Checklist

| # | Mục | Trạng thái | Điểm | Ghi chú |
|---|---|---|---|---|
| 1 | Build production | ✅ | 5/5 | Maven test-compile PASS trong Docker chính thức; next build PASS trong CI |
| 2 | Environment configuration | ✅ | 5/5 | Toàn bộ secret qua env; fail-fast JWT_SECRET; compose đã parameterize (fix audit) |
| 3 | Docker image | ✅ | 4/5 | 3 Dockerfile multi-stage; edge 10.4GB (torch cu124) — lớn nhưng đúng bản chất GPU; web non-root |
| 4 | Docker Compose | ✅ | 5/5 | healthcheck + depends_on condition; volumes đúng; validate PASS |
| 5 | Database migration | ⚠️ | 2/5 | ddl-auto=update (không Flyway) — chấp nhận được cho đồ án; production lớn cần Flyway |
| 6 | Database backup | ✅ | 4/5 | Đã có scripts/backup_central.sh (pg_dump + MinIO, giữ 14 bản) — chưa chạy tự động |
| 7 | Health check | ✅ | 5/5 | /api/health public + actuator; edge /health với pipeline_stale→503 |
| 8 | Logging | ✅ | 4/5 | SLF4J cấu trúc tốt, warn级别 đúng; chưa có centralized log (ELK) |
| 9 | Monitoring | ⚠️ | 3/5 | Docker healthcheck + metrics in-process edge; chưa có Prometheus/Grafana + alert |
| 10 | Error tracking | ⚠️ | 2/5 | Log file only — chưa có Sentry/đẩy alert Telegram tự động |
| 11 | Reverse proxy | ⚠️ | 3/5 | Hướng dẫn Nginx đầy đủ trong deployment guide — chưa cấu hình thực tế |
| 12 | HTTPS | ⚠️ | 3/5 | Kế hoạch TLS qua Nginx (không có self-signed trong stack dev) |
| 13 | CORS | ✅ | 4/5 | Proxy model: browser không gọi trực tiếp; central patterns=* credentials=false; edge whitelist |
| 14 | Authentication | ✅ | 5/5 | JWT HS256 + BCrypt + refresh rotation + reuse-detection revoke-all |
| 15 | Authorization | ✅ | 5/5 | RBAC 3 vai trò endpoint-level + client-side guard |
| 16 | Rate limiting | ⚠️ | 3/5 | Edge có (30 req/ph/IP); CENTRAL chưa rate limit login — brute force còn khả thi |
| 17 | Resource limits | ⚠️ | 2/5 | Chưa đặt mem/cpu limits trong compose; GPU reserved đúng |
| 18 | Restart policy | ✅ | 5/5 | unless-stopped toàn bộ 5 service |
| 19 | Persistent volumes | ✅ | 5/5 | postgres_data, minio_data, outbox_data, hls_data |
| 20 | Backup (đã test restore?) | ⚠️ | 3/5 | Script có; hướng dẫn restore có; CHƯA chạy restore test thực tế |
| 21 | CI/CD | ✅ | 4/5 | GitHub Actions 3 job pass (edge tests, central compile, web build); chưa có auto-deploy |
| 22 | Rollback strategy | ✅ | 4/5 | Git checkout + rebuild + restore DB theo guide |

**Tổng: 78/100**

## Phân loại issue

### CRITICAL (0 — đã dọn trong audit)
Không còn. (Credential hard-code compose và full-table-scan đã sửa trong đợt này.)

### HIGH (2 — cần làm trước khi public internet)
1. **Central chưa rate-limit /api/auth/login** — brute force mật khẩu admin còn khả thi. Đề xuất: bucket4j hoặc filter sliding-window giống edge.
2. **Resource limits + restore-test chưa thực hiện** — cần đặt mem_limit cho compose và chạy thử backup→restore 1 lần.

### MEDIUM (5 — nên có trong giai đoạn vận hành)
3. Migration Flyway thay cho ddl-auto=update.
4. Centralized monitoring (Prometheus + Grafana) và alert.
5. Integration test tự động central (JUnit + Testcontainers).
6. Frontend component test (Vitest).
7. Rate limit login + lockout account sau N lần sai.

### LOW (3)
8. edge_node/ 152 lỗi ruff stylistic (UP045 Optional→X\|None...) — CI chỉ enforce tests/.
9. Image edge 10.4GB — cân nhắc prune layers hoặc chia image CPU/GPU.
10. postgres/minio container chạy root (mặc định image chính thức).

## Kết luận

**Điểm 78/100 — sẵn sàng cho đồ án + demo vận hành nội bộ (LAN/lab), CHƯA sẵn sàng public internet** cho đến khi xử lý 2 issue HIGH (rate limit login, restore test + resource limits). Không có CRITICAL nào còn tồn tại sau đợt audit này.
