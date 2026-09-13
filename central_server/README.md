# Central Server — Red-Light Violation Detection

Backend trung tâm (Java Spring Boot 3) nhận hồ sơ vi phạm từ các Edge Node, lưu metadata vào PostgreSQL và media vào MinIO, đồng thời cung cấp REST API cho web dashboard. Kèm engine **Text-to-SQL** (Gemini) cho phép hỏi dữ liệu vi phạm bằng tiếng Việt.

## Kiến trúc

```
                 ┌────────────────────────────────────────────┐
 Edge Node ─────▶│              CENTRAL SERVER (:8000)         │
 (violation      │                                             │
  batch + media) │  Controller ──▶ Service ──▶ Repository      │
                 │      │                          │           │
                 │      │                          ▼           │
                 │      │                     PostgreSQL       │
                 │      │                    (metadata)        │
                 │      └──▶ MinioStorageService ──▶ MinIO     │
                 │                                  (media)    │
                 │  AiQueryService ──▶ Gemini (Text-to-SQL)    │
                 │  EdgeProxyService ──▶ Edge Node API         │
                 └────────────────────────────────────────────┘
                              ▲
                              │  /api/*  (Next.js rewrite)
                        Web Dashboard
```

## Tech stack

- **Spring Boot 3.3.2** trên **Java 17** (web, data-jpa, validation, actuator)
- **PostgreSQL 16** — metadata vi phạm + edge node (JPA `ddl-auto: update`, timezone `Asia/Ho_Chi_Minh`, Jackson `SNAKE_CASE`)
- **MinIO** (S3-compatible) — lưu ảnh toàn cảnh + ảnh crop biển số, bucket `violations`
- **Gemini** (`gemini-2.5-flash`) — Text-to-SQL cho AI query
- Build bằng **Maven**, đóng gói Docker multi-stage (`maven:3.9-eclipse-temurin-17` → `temurin:17-jre-alpine`)

## Cấu trúc

```
central_server/src/main/java/com/rlvd/centralserver/
├── CentralServerApplication.java
├── config/
│   ├── MinioConfig.java          # MinIO client bean
│   └── WebConfig.java            # CORS
├── controller/
│   ├── ViolationController.java  # CRUD + batch + stats + health
│   ├── MediaController.java      # Upload/stream/info/delete media
│   ├── EdgeNodeController.java   # Register + calibration proxy
│   └── AiQueryController.java    # Text-to-SQL
├── service/
│   ├── ViolationService.java     # Nghiệp vụ vi phạm
│   ├── MinioStorageService.java  # Put/get object MinIO
│   ├── EdgeNodeService.java      # Quản lý node + heartbeat
│   ├── EdgeProxyService.java     # Proxy request xuống edge node
│   └── AiQueryService.java       # Gemini → SQL → chạy → chart type
├── repository/                   # Spring Data JPA
├── entity/                       # Violation, EdgeNode
└── dto/                          # Request/Response
```

## API Endpoints

### Violations (`/api`)

| Endpoint | Method | Mô tả |
|---|---|---|
| `/api/violations` / `/api/v1/violations` | POST | Nhận 1 vi phạm từ edge |
| `/api/violations/batch` / `/api/v1/violations/batch` | POST | Nhận batch vi phạm |
| `/api/violations` | GET | Danh sách (filter) |
| `/api/violations/page` | GET | Phân trang (server-side) |
| `/api/violations/counts` | GET | Đếm theo trạng thái |
| `/api/violations/{id}` | GET | Chi tiết 1 vi phạm |
| `/api/violations/event/{eventId}` | GET | Tra theo event_id |
| `/api/violations/{id}/status` | PATCH/PUT | Duyệt / từ chối (human-in-the-loop) |
| `/api/violations/{id}` | DELETE | Xoá |
| `/api/stats` | GET | Thống kê tổng hợp |
| `/api/health` | GET | Health check |

### Media (`/api/v1/violations/{eventId}/media`)

| Endpoint | Method | Mô tả |
|---|---|---|
| `.../media` | POST | Upload ảnh/video (multipart) lên MinIO |
| `.../media/blob` | GET | Stream binary media |
| `.../media` | GET | Thông tin media (URL, size) |
| `.../media` | DELETE | Xoá media |

### Edge Nodes (`/api/v1/edge-nodes`)

| Endpoint | Method | Mô tả |
|---|---|---|
| `/register` | POST | Edge node đăng ký (+ heartbeat) |
| `` | GET | Danh sách node |
| `/{nodeId}` | GET | Chi tiết node |
| `/{nodeId}/settings` | PUT | Cập nhật cấu hình node |
| `/{nodeId}/calibration` | GET | Trạng thái calibration (proxy xuống edge) |
| `/{nodeId}/calibration/snapshot` | GET | Frame JPEG (proxy) |
| `/{nodeId}/calibration/stop-line` | POST | Đặt vạch dừng + hướng (proxy) |
| `/{nodeId}/calibration/light-roi` | POST | Đặt vùng đèn (proxy) |

### AI Query (`/api/ai`)

| Endpoint | Method | Mô tả |
|---|---|---|
| `/api/ai/query` | POST | Hỏi bằng tiếng Việt → Gemini sinh SQL SELECT → chạy → trả rows + chart type |

**An toàn Text-to-SQL**: chỉ cho phép `SELECT`/`WITH`; chặn `DROP/DELETE/INSERT/UPDATE/ALTER/CREATE/TRUNCATE/GRANT/REVOKE` bằng regex; tự thêm `LIMIT 50`.

## Cấu hình (env var)

| Variable | Default | Mô tả |
|---|---|---|
| `DB_HOST` / `DB_PORT` / `DB_NAME` | `localhost` / `5432` / `rlvd_central` | PostgreSQL |
| `DB_USER` / `DB_PASS` | `rlvd` / `rlvd_secret` | Credentials |
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` | Credentials |
| `MINIO_BUCKET` | `violations` | Bucket lưu media |
| `GEMINI_API_KEY` | – | API key cho Text-to-SQL |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model Gemini |
| `server.port` | `8000` | Cổng HTTP |

## Bảo mật & phân quyền (auth)

Hai luồng xác thực tách bạch:

1. **Người dùng dashboard — JWT**: đăng nhập `POST /api/auth/login {username, password}` → nhận JWT HS256 (TTL 12h) kèm role. Mọi request API gắn `Authorization: Bearer <token>`. Mật khẩu băm BCrypt trong bảng `users`.
2. **Node biên — ingest token**: các endpoint nhận hồ sơ (POST `/api/violations*`, `/api/v1/edge-nodes/register`) yêu cầu header `X-Ingest-Token` khớp `INGEST_TOKEN` (so sánh constant-time). Không dùng JWT cho edge vì edge là máy chủ, không phải user.

Phân quyền theo vai trò (Spring Security `hasRole`):

| Vai trò | Được phép |
|---|---|
| `ADMIN` | Toàn quyền (gồm DELETE hồ sơ, quản lý user) |
| `OPERATOR` | Hiệu chuẩn node (`/api/v1/edge-nodes/**`), xem dữ liệu, hỏi AI |
| `OFFICER` | Duyệt/từ chối hồ sơ (`PATCH .../status`), xem dữ liệu, hỏi AI |

Public (không cần token): `POST /api/auth/login`, `GET /api/health`, `/actuator/health`, `GET .../media/blob` + `.../calibration/snapshot` (proxy từ web đã xác thực), và POST ingest (chỉ cần X-Ingest-Token).

### Biến môi trường bảo mật

Đặt trong `.env` ở **gốc repo** (file duy nhất dùng chung cho cả 3 compose — docker compose tự nạp theo project directory; không còn `central_server/.env` riêng).

| Variable | Ý nghĩa |
|---|---|
| `JWT_SECRET` | Khóa ký JWT (≥ 32 ký tự) — **bắt buộc**, server fail-fast nếu thiếu. Sinh: `openssl rand -base64 32` |
| `JWT_TTL_SECONDS` | Thời gian sống token (mặc định 43200 = 12h) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | Tài khoản admin seed lần chạy đầu tiên — **đổi trước khi triển khai** |
| `INGEST_TOKEN` | Token node biên dùng khi đẩy hồ sơ — đặt trùng giá trị ở edge (`INGEST_TOKEN`) |

Tài khoản admin được tự tạo lần khởi động đầu (UserSeeder); các user khác thêm trực tiếp vào bảng `users` (mật khẩu BCrypt).

## Chạy

### Docker (khuyến nghị — dùng chung stack)

```bash
# Từ gốc project: dựng cả postgres + minio + central + web + edge
./start.sh

# Chỉ central + hạ tầng (không edge)
./start.sh minimal
```

Chạy riêng compose của central (postgres + minio + central-server, không web) — từ gốc repo, `--project-directory` để nạp root `.env` chung:

```bash
docker compose --project-directory . -f central_server/docker-compose.yml up -d
```

### Chạy local với Maven

```bash
cd central_server
# Cần PostgreSQL + MinIO đang chạy (hoặc dùng docker compose cho 2 service này)
mvn spring-boot:run
```

### Build image

```bash
docker build -t rlvd-central central_server/
```

## Schema chính

**violations**: `id, event_id, node_id, track_id, frame_index, timestamp_ms, light_state (red/yellow/green/unknown), light_confidence, plate_text, plate_confidence, status (pending/approved/rejected), created_at, updated_at`

**edge_nodes**: `id, node_id, name, ip_address, status (online/offline/maintenance), online, last_ping, created_at`
