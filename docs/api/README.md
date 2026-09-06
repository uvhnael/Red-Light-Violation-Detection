# RLVD — API Documentation

Base URL (qua Next.js proxy của web dashboard): `http://<web-host>:3000`
Trực tiếp central server (Docker nội bộ): `http://central-server:8000` · host `:8002`
Trực tiếp edge node: `http://<edge-host>:8080` · host `:8082`

Quy ước: mọi response JSON, snake_case. Timestamp ISO-8601 múi giờ Asia/Ho_Chi_Minh.

## 1. Authentication — /api/auth

| Method | Path | Auth | Role | Mô tả |
|---|---|---|---|---|
| POST | /api/auth/login | — | — | `{username,password}` → `{access_token, refresh_token, role, expires_in}` |
| POST | /api/auth/refresh | — | — | `{refreshToken}` → xoay vòng cặp token mới (revoke cũ) |
| POST | /api/auth/logout | — | — | `{refreshToken}` → revoke (idempotent) |
| GET | /api/auth/me | Bearer JWT | mọi role | Thông tin user hiện tại |

Roles: ADMIN (toàn quyền) > OPERATOR (hiệu chuẩn node, không duyệt) > OFFICER (duyệt hồ sơ, hỏi AI).

## 2. Violations — ingest từ Edge Node (X-Ingest-Token)

| Method | Path | Mô tả |
|---|---|---|
| POST | /api/violations | 1 hồ sơ. Header `X-Node-ID`, `X-Ingest-Token`. Trùng event_id → 409 |
| POST | /api/v1/violations | alias versioned |
| POST | /api/violations/batch | `{"violations":[...]}` batch tối đa từ outbox. Idempotent: `{accepted, duplicates, failed, acceptedEventIds, duplicateEventIds}` |
| POST | /api/v1/violations/batch | alias versioned |
| POST | /api/v1/violations/{eventId}/media | multipart `file` (image/jpeg) → MinIO, trả `{object_name, url}` |

Body ViolationRequest (rút từ ViolationRequest.java):
```json
{
  "event_id": "rlvd-a1b2c3d4-00000021-2",
  "light_state": "red",
  "light_confidence": 0.91,
  "track_id": 2,
  "frame_index": 21,
  "timestamp_ms": 700.0,
  "crossing_point": {"x": 960.0, "y": 720.0},
  "previous_point": {"x": 950.0, "y": 700.0},
  "bbox_xyxy": [900, 650, 1020, 780],
  "previous_side": 1, "current_side": -1,
  "plate": {"text": "30-H12345", "confidence": 0.87},
  "metadata": {"direction": "positive_to_negative"}
}
```

## 3. Violations — query cho Dashboard (JWT)

| Method | Path | Role | Mô tả |
|---|---|---|---|
| GET | /api/violations/page?page=0&size=20&status=pending&nodeId=&plateText= | mọi role | Phân trang server-side → `{content,page,size,total_elements,...}`. status nhận comma-separated (`pending,approved`) |
| GET | /api/violations/counts | mọi role | `{total, pending, approved, rejected}` — COUNT queries rẻ |
| GET | /api/violations/{id} | mọi role | 1 hồ sơ theo DB id |
| GET | /api/violations/event/{eventId} | mọi role | 1 hồ sơ theo event id |
| GET | /api/violations?status=&nodeId=&plateText= | mọi role | Toàn bộ theo filter (legacy — cẩn thận bảng lớn) |
| GET | /api/violations/{id}/media/blob | permit-only* | Stream ảnh bằng chứng từ MinIO |

(*) blob permit vì web proxy đã authenticate phía browser; không cần ingest token.

| Method | Path | Role | Mô tả |
|---|---|---|---|
| PATCH/PUT | /api/violations/{id}/status | ADMIN, OFFICER | `{"status":"approved"\|"rejected"\|"confirmed"}` — human-in-the-loop |
| DELETE | /api/violations/{id} | ADMIN | Xoá hồ sơ |

## 4. Statistics

| Method | Path | Mô tả |
|---|---|---|
| GET | /api/stats | Tổng hợp COUNT/GROUP BY: total, theo trạng thái, theo node, hourly_trend (lọc hôm nay), theo light_state |

## 5. Edge Nodes — quản lý

| Method | Path | Role | Mô tả |
|---|---|---|---|
| POST | /api/v1/edge-nodes/register | X-Ingest-Token | Đăng ký/heartbeat: `{node_id,name,ip_address,status,settings{...}}` — upsert theo node_id |
| GET | /api/v1/edge-nodes | ADMIN, OPERATOR | Danh sách node (online = last_ping < 2 phút) |
| GET | /api/v1/edge-nodes/{nodeId} | ADMIN, OPERATOR | Chi tiết node |
| PUT | /api/v1/edge-nodes/{nodeId} | ADMIN, OPERATOR | Cập nhật name/ip/status/settings |
| POST | /api/v1/edge-nodes/{nodeId}/calibration/stop-line | ADMIN, OPERATOR | Proxy → edge POST /action/stop-line `{x1,y1,x2,y2,direction}` — hiệu lực live |
| POST | /api/v1/edge-nodes/{nodeId}/calibration/light-roi | ADMIN, OPERATOR | Proxy → edge POST /action/light-roi `{x,y,w,h}` |
| GET | /api/v1/edge-nodes/{nodeId}/calibration/snapshot | ADMIN, OPERATOR | Proxy → edge JPEG frame để vẽ calibration |

direction: `any` | `positive_to_negative` | `negative_to_positive`.

## 6. AI Text-to-SQL

| Method | Path | Role | Mô tả |
|---|---|---|---|
| POST | /api/ai/query | ADMIN, OPERATOR, OFFICER | `{question}` (tiếng Việt, ≤500 ký tự) → Gemini sinh SQL SELECT-only (chặn DROP/DELETE/INSERT/UPDATE/ALTER/TRUNCATE/GRANT/REVOKE, LIMIT 50) → chạy PostgreSQL → `{sql, columns, rows, count, chartType}` |

## 7. Health

| Method | Path | Mô tả |
|---|---|---|
| GET | /api/health | Public — `{status:ok, timestamp}` ISO múi giờ VN |
| GET | /actuator/health, /actuator/info | Public — Spring actuator |

## 8. Edge Node Control API (trực tiếp :8080 hoặc qua /edge-api)

| Method | Path | Auth | Mô tả |
|---|---|---|---|
| GET | /health | — | 503 degraded nếu pipeline_stale >10s; metrics frames/violations/fps |
| GET | /api/cameras | — | Danh sách camera (HLS từ video input) |
| GET | /api/cameras/{id}/stream | — | HLS m3u8 playlist |
| GET | /api/cameras/{id}/stream/{file} + /api/cameras/{id}/{file} | — | Segment .ts (full path + relative RFC3986 fallback) |
| GET | /api/cameras/{id}/snapshot | — | JPEG frame hiện tại |
| GET | /api/calibration | — | Stop line + light ROI hiện tại |
| GET | /api/light-state | — | Trạng thái đèn realtime (debounced) |
| GET | /api/light-roi | — | ROI đèn hiện tại |
| POST | /action/stop-line | X-Edge-Token | `{x1,y1,x2,y2,direction}` — rate limit 30 req/ph/IP |
| POST | /action/light-roi | X-Edge-Token | `{x,y,w,h}` |
| POST | /action/restart | X-Edge-Token | Restart pipeline (Docker-aware SIGTERM) |
| GET | /action/stop-line | — | Tripwire hiện tại |

## 9. Mã lỗi chuẩn

| Code | Ý nghĩa |
|---|---|
| 400 | Body sai cấu trúc / thiếu trường bắt buộc (@Valid) |
| 401 | Thiếu/sai JWT, sai mật khẩu, sai ingest/edge token |
| 403 | Đúng JWT nhưng role không đủ (RBAC) |
| 404 | Không tìm thấy resource (violation/node/camera) |
| 409 | event_id đã tồn tại (single ingest) |
| 429 | Vượt rate limit edge (30 req/ph/IP trên /action/*) |
| 502 | Central không gọi được edge (EdgeProxyService) |
