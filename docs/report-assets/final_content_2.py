# -*- coding: utf-8 -*-
"""Nội dung báo cáo đồ án RLVD (bản pre-production) — phần 2: Chương 4 → cuối.

Chương 4 xây dựng, Chương 5 kiểm thử, Chương 6 triển khai, Chương 7 kết luận,
tài liệu tham khảo + phụ lục. Số liệu thực: LOC đếm từ repo, pytest 85 pass,
benchmark_results.csv, audit 2026-09-06.
"""

ASSETS = "/home/uvhnael/projects/Red-Light-Violation-Detection/docs/report-assets"
DIAG = "/home/uvhnael/projects/Red-Light-Violation-Detection/docs/diagrams"

BLOCKS_2 = [
    # ================= CHƯƠNG 4 =================
    {"type": "heading", "text": "CHƯƠNG 4. XÂY DỰNG HỆ THỐNG", "level": 1},

    {"type": "heading", "text": "4.1. Môi trường phát triển", "level": 2},
    {"type": "bullet_list", "items": [
        "Máy phát triển: Linux 64-bit, GPU NVIDIA RTX 2060 (CUDA 12.4), Python 3.11 trong venv riêng (torch 2.6.0+cu124, ultralytics 8.4.110, supervision 0.28, OpenCV 5.0.0).",
        "Trung tâm: Java 17, Maven 3.9, Spring Boot 3.3.2 — build trong Docker image maven:3.9-eclipse-temurin-17 (host chỉ có JRE).",
        "Web: Node 20, Next.js 16, React 19, TypeScript 5, Tailwind 4.",
        "CI: GitHub Actions — 3 job (edge-tests pytest + ruff, central-build maven test-compile, web-build lint + build).",
    ]},

    {"type": "heading", "text": "4.2. Frontend", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Bảng điều khiển web (6.591 dòng TS/TSX) gồm 7 trang chính + hệ component dùng lại. Kiến trúc: App Router với route group (dashboard) được middleware bảo vệ (cookie rlvd_token), mọi gọi API qua rewrite proxy của Next — trình duyệt không bao giờ gọi thẳng central/edge (tránh vấn đề CORS + che config nội bộ). Quản lý phiên: token trong localStorage + đồng bộ cookie cho SSR; tự refresh khi 401 một lần rồi retry; 6 theme (VN Dark/Light — đỏ mận + vàng đồng CSGT, Dark, Light, System, VN System)."},
    {"type": "bullet_list", "items": [
        "Trang /login (Hình 4.1) — form đăng nhập, animation chờ, hiển thị lỗi tiếng Việt.",
        "Trang / (Hình 4.2) — 4 thẻ KPI (vi phạm hôm nay, node hoạt động, độ chính xác AI, số đã duyệt), biểu đồ vùng theo giờ, bảng vi phạm gần đây, widget node.",
        "Trang /violations (Hình 4.3) — tra cứu phân trang server-side 20/trang, filter trạng thái/node/biển số, badge trạng thái + độ tin cậy.",
        "Trang /review (Hình 4.4) — duyệt human-in-the-loop: tải pending theo batch 50 có prefetch, ảnh bằng chứng lớn, nút duyệt/từ chối, hiệu ứng chuyển thẻ.",
        "Trang /nodes + /nodes/[id] (Hình 4.5) — trạng thái node, stream HLS live, công cụ hiệu chuẩn (3 chế độ vẽ + chọn hướng).",
        "Trang /cameras (Hình 4.6) — lưới camera live HLS (hls.js code-split riêng, static import để tương thích mọi browser).",
        "Trang /settings — chọn theme, xem thông tin phiên.",
        "Trợ lý AI — panel trượt phải (FAB góc phải): câu hỏi tiếng Việt, hiển thị SQL sinh ra + bảng/biểu đồ kết quả.",
    ]},
    {"type": "image", "path": ASSETS + "/shots/07-login.png", "width_mm": 130},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.1. Trang đăng nhập web dashboard"},
    {"type": "image", "path": ASSETS + "/shots/08-dashboard-logged-in.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.2. Trang thống kê tổng quan sau đăng nhập"},
    {"type": "image", "path": ASSETS + "/shots/09-violations-authed.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.3. Trang tra cứu vi phạm (đã đăng nhập)"},
    {"type": "image", "path": ASSETS + "/shots/03-review.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.4. Trang duyệt hồ sơ vi phạm"},
    {"type": "image", "path": ASSETS + "/shots/04-nodes.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.5. Trang quản lý node biên"},
    {"type": "image", "path": ASSETS + "/shots/05-cameras.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.6. Trang xem camera trực tuyến"},
    {"type": "image", "path": ASSETS + "/shots/06-node-calibration.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.7. Hiệu chuẩn trên video trực tuyến (vạch dừng, vùng đèn, hướng giám sát)"},

    {"type": "heading", "text": "4.3. Backend — Node biên", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Node biên (5.846 dòng Python, 29 module) gồm các lớp chính:"},
    {"type": "bullet_list", "items": [
        "core/detector.py — YoloDetector: nhận diện fine-tuned 4 lớp theo TÊN lớp (không remap id COCO — bug từng làm mất toàn bộ ô tô); hỗ trợ conf override.",
        "core/byte_tracker.py — _TunedByteTrack: fork update_with_tensors của supervision để mở activation cho xe máy (conf 0,30, activation 0,10, match 0,85) — các ngưỡng đo A/B trên video 3fps.",
        "core/traffic_light_yolo.py + traffic_light_cv.py — phân loại đèn YOLO26n-cls là chính; fusion HSV + vị trí bóng đèn vertical khi YOLO lệch với HSV; HSV module chỉ còn fallback + tìm ROI.",
        "core/violation_logic.py — ViolationDetector: mỗi frame đọc lại vạch active (hiệu chỉnh live), tracking điểm qua vạch theo bbox CENTER (quyết định thiết kế: xe mui vượt vạch là vi phạm), chống deadband bằng neo điểm khác 0 gần nhất, một track một vi phạm.",
        "core/plate_detector.py + plate_associator.py + ocr_recognizer.py — phát hiện biển BSD/BSV, gắn biển vào track theo containment, cache OCR mỗi track (conf ≥ 0,80 + đúng chiều dài), fast-plate-ocr + vn_plate validator (repair theo vùng: O↔0, I↔1...).",
        "outbox.py + violation_sender.py — SQLite outbox (UNIQUE event_id, image BLOB riêng cột media) + sender thread: fetch batch không đọc BLOB, POST batch, upload ảnh, backoff 2^n tối đa 60 giây.",
        "api/server.py — FastAPI control-plane: /health (metrics in-process, 503 khi stale >10s), /api/cameras + HLS stream/snapshot, /action/* guard token constant-time + rate limit 30 req/phút/IP sliding window + security headers (nosniff, DENY, no-store).",
        "camera_stream.py — FFmpeg HLS 4s segments, auto-restart, delete_segments (không append_list).",
        "central_client.py — đăng ký + heartbeat 60s, kèm ingest token; publish cả api_token để central proxy ngược đúng.",
    ]},
    {"type": "paragraph", "style": "Body", "text": "Xử lý lỗi từng frame bọc try/except toàn khối: frame hỏng/model lỗi → log warning + tiếp tục, tracker và stabilizer tự dung sai khoảng trống ngắn — node không chết giữa ca trực."},

    {"type": "heading", "text": "4.4. Backend — Central Server", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Trung tâm (3.582 dòng Java, 43 file) theo layer chuẩn:"},
    {"type": "bullet_list", "items": [
        "controller/ — 5 REST controller (Auth, Violation, EdgeNode, Media, AiQuery): chỉ nhận request, validate, gọi service — không chứa SQL hay nghiệp vụ.",
        "service/ — 7 service: ViolationService (ingest idempotent single/batch, phân trang, CRUD), ViolationStatsService (COUNT/GROUP BY không tải bảng), EdgeNodeService (registry upsert, online <2 phút), RefreshTokenService (SHA-256, rotation, reuse → revoke all), EdgeProxyService (gọi ngược edge, HTTP/1.1, 502 tiếng Việt), MinioStorageService, AiQueryService (Gemini REST + guard SQL + JdbcTemplate).",
        "repository/ — Spring Data JPA: existsByEventId (check trùng nhanh), findByStatus paged, các COUNT GROUP BY bằng @Query.",
        "config/ — SecurityConfig (RBAC endpoint), JwtService (HS256, fail-fast secret ≥32), JwtAuthFilter, IngestTokenFilter, IngestPaths (MỘT nguồn sự thật đường dẫn ingest — SecurityConfig và filter cùng đọc, không thể lệch), UserSeeder (admin từ env, không ghi đè).",
        "entity/ — 4 entity với index khai báo ngay @Table (violations 4 index, edge_nodes 3, users 1, refresh_tokens 2).",
    ]},
    {"type": "paragraph", "style": "Body", "text": "Hai cải tiến của kiểm định 9/2026: graceful shutdown (server.shutdown=graceful + timeout 30s — không đứt batch giữa chừng khi nâng cấp) và HikariCP pool khai báo tường minh (max 10, min idle 2, connection-timeout 10s)."},

    {"type": "heading", "text": "4.5. Database", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Schema quản lý bằng JPA (ddl-auto=update) — chấp nhận cho đồ án; production lớn khuyến nghị Flyway (ghi nhận ở hạn chế chương 7). Multiple index đúng truy vấn chính: event_id unique (idempotent), status (filter + counts), created_at (thống kê giờ + sort mới nhất), node_id (lọc theo node), last_ping (online check). Thời gian toàn tầng thống nhất Asia/Ho_Chi_Minh (Jackson + Hibernate jdbc time_zone + Docker TZ + edge datetime offset +07:00)."},

    {"type": "heading", "text": "4.6. API", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Toàn bộ 40+ endpoint được tài liệu hoá trong docs/api/README.md. Bảng 4.1 tóm tắt các nhóm chính."},
    {"type": "table", "header": ["Nhóm", "Endpoint tiêu biểu", "Bảo vệ"], "rows": [
        ["Auth", "POST /api/auth/login, /refresh, /logout; GET /api/auth/me", "public (login/refresh/logout); JWT cho /me"],
        ["Ingest vi phạm", "POST /api/violations/batch, /api/v1/violations/*/media", "X-Ingest-Token (constant-time)"],
        ["Node registry", "POST /api/v1/edge-nodes/register; GET/PUT /api/v1/edge-nodes/**", "token ingest (register); JWT ADMIN/OPERATOR (quản lý)"],
        ["Tra cứu", "GET /api/violations/page, /counts, /{id}, /event/{eventId}; GET /api/stats", "JWT mọi vai trò"],
        ["Duyệt", "PATCH/PUT /api/violations/{id}/status; DELETE /api/violations/{id}", "JWT ADMIN/OFFICER (duyệt); ADMIN (xoá)"],
        ["Media", "GET /api/v1/violations/*/media/blob (stream MinIO)", "permit-only (proxy đã auth)"],
        ["Hiệu chuẩn", "POST /api/v1/edge-nodes/{id}/calibration/stop-line, light-roi; GET snapshot", "JWT ADMIN/OPERATOR + forward X-Edge-Token"],
        ["AI", "POST /api/ai/query", "JWT mọi vai trò + guard SELECT-only"],
        ["Health", "GET /api/health, /actuator/health", "public"],
        ["Edge control", "GET /api/cameras... /api/calibration, /api/light-state; POST /action/*", "public (đọc); X-Edge-Token + rate limit (ghi)"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 4.1. Danh sách API chính (đầy đủ trong phụ lục API documentation)"},

    {"type": "heading", "text": "4.7. Authentication", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Hai hệ thống xác thực song song: (1) người dùng web — JWT HS256 TTL 12 giờ (đúng một ca trực), secret bắt buộc ≥32 ký tự fail-fast; mật khẩu BCrypt; refresh token 32 byte entropy lưu SHA-256 trong DB, TTL 7 ngày, rotation mỗi lần refresh, reuse token đã revoke → thu hồi TOÀN BỘ token của user (chống đánh cắp); (2) node biên — ingest token tĩnh X-Ingest-Token so sánh constant-time (MessageDigest.isEqual) tách khỏi JWT vì node là máy không phải người. RBAC 3 vai trò áp ở backend (requestMatchers theo method + path) và cả client (guard nút theo vai trò)."}, 

    {"type": "heading", "text": "4.8. Data synchronization (đồng bộ dữ liệu)", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Đồng bộ Edge → Central qua outbox pattern: (1) ghi hồ sơ + ảnh vào SQLite NGAY khi phát hiện (trước mọi I/O mạng); (2) sender thread poll mỗi 5 giây, gửi batch 20 hồ sơ JSON (không tải ảnh); (3) batch idempotent — trung tâm check existsByEventId, trùng thì đếm duplicates không lỗi; (4) sau batch, upload từng ảnh bằng chứng, đánh dấu media_sent; (5) lỗi mạng → giữ pending + backoff luỹ tiến tối đa 60 giây; (6) restart node → sender rút lại toàn bộ pending. Kiểm chứng thực tế: kịch bản tắt trung tâm → pending tích luỹ đúng số phát hiện → bật lại → flush đủ, không trùng; sau khi TRUNCATE DB phía trung tâm, các dòng đã sent KHÔNG gửi lại (cần chú ý vận hành — đã ghi trong tài liệu chẩn đoán)."},

    {"type": "heading", "text": "4.9. Docker", "level": 2},
    {"type": "bullet_list", "items": [
        "docker-compose.full.yml — 5 dịch vụ: postgres:16-alpine (healthcheck pg_isready), minio (healthcheck mc ready), central-server (build 2-stage Maven, healthcheck wget /api/health, phụ thuộc postgres healthy), web-dashboard (build args bake URL proxy, phụ thuộc central healthy), edge-pipeline (GPU nvidia, volume models read-only + outbox riêng vì ./data mount read-only).",
        "start.sh — dựng/chạy/status/logs/rebuild từng dịch vụ/minimal (bỏ edge)/down; kiểm tra models + video trước khi up.",
        "Tất cả secret sau kiểm định 9/2026 đọc từ .env qua biến ${VAR:-default} — dev dùng default, production bắt buộc .env.",
        "TZ Asia/Ho_Chi_Minh đặt trên MỌI dịch vụ + Dockerfile.",
    ]},

    {"type": "heading", "text": "4.10. Deployment", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Quy trình triển khai chi tiết trong docs/deployment/production-guide.md: chuẩn bị .env (sinh secret bằng openssl), ./start.sh, verify checklist (health, login, node register, kẻ vạch), reverse proxy Nginx + TLS, backup hàng ngày scripts/backup_central.sh (pg_dump + MinIO mirror, giữ 14 bản), restore test, rollback (git checkout + rebuild + restore DB). Hình 4.8 thể hiện kiến trúc sản phẩm khi public."},
    {"type": "image", "path": DIAG + "/api-surface.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 4.8. Bề mặt API và các lớp bảo vệ (trùng Hình 3.4 — xem chi tiết chapter 3.8)"},
    {"type": "page_break"},

    # ================= CHƯƠNG 5 =================
    {"type": "heading", "text": "CHƯƠNG 5. KIỂM THỬ VÀ ĐÁNH GIÁ", "level": 1},

    {"type": "heading", "text": "5.1. Test strategy", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Sáu mức kiểm thử: (1) unit test logic (pytest, không cần GPU); (2) kiểm thử API bằng HTTP thật trên stack đang chạy; (3) kiểm thử phân quyền theo từng vai trò; (4) kiểm thử end-to-end video có ground-truth; (5) kiểm thử bảo mật (token/CORS/rate-limit/SQL guard); (6) kiểm thử vận hành Docker (khởi động, restart, offline→online). CI tự động 3 job chạy mỗi push."},

    {"type": "heading", "text": "5.2. Unit testing", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "85 test pytest chia 5 file (geometry, violation_logic, tracker_stabilizer, vn_plate, metrics) — kết quả lần chạy gần nhất (9/2026): 85 passed in 0,87 giây. Như thể hiện trong Hình 5.1. Coverage tập trung logic nghiệp vụ (hình học vạch, quy tắc vi phạm, debounce đèn theo giây thực, validator biển số)."},
    {"type": "image", "path": DIAG + "/use-case.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 5.1. Sơ đồ use case — cơ sở cho các ca kiểm thử nghiệp vụ (kết quả pytest: 85 passed in 0.87s — xem Bảng 5.2)"},

    {"type": "heading", "text": "5.3. Integration testing", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Kiểm thử tích hợp bằng HTTP request thật (curl/httpx) trên stack Docker: đăng ký node, ingest batch trùng, upload media, hiệu chuẩn qua proxy với token, HTTP/1.1 ngược edge (đã từng gặp bug h2c làm rơi POST body → 422). Kết quả các kịch bản đã thực hiện: 18/18 đạt (chi tiết Bảng 5.2). Lưu ý trung thực: các integration test này chạy THỦ CÔNG theo kịch bản có ghi nhận — chưa tự động hoá thành JUnit + Testcontainers (ghi nhận ở chương 7 hạn chế)."},

    {"type": "heading", "text": "5.4. API testing", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Từng endpoint được kiểm tra: mã trạng thái đúng (200/201/400/401/403/404/409/429/502), JSON shape đúng (phân trang trả content/total_elements, counts trả 4 số, batch trả accepted/duplicates/failed), stream media trả JPEG magic bytes (ff d8 ff). Kiểm thử 403-vs-401: Spring forward /error phải permit để status thật không bị che thành 403."},

    {"type": "heading", "text": "5.5. Edge Node testing", "level": 2},
    {"type": "bullet_list", "items": [
        "Connection test: đăng ký thành công khi central online; tiếp tục pipeline khi offline (chỉ warning).",
        "Retry test: sender backoff 5→10→20→40→60s khi central chết; outbox pending tăng đúng số vi phạm.",
        "Offline test: tắt central 5 phút → 0 hồ sơ mất; bật lại → flush toàn bộ pending.",
        "Reconnection test: restart container edge → pending cũ được gửi tiếp (volume outbox_data).",
        "Data sync + dedupe: gửi 2 lần cùng batch → lần 2 accepted=0, duplicates=N; event_id UNIQUE hai phía.",
        "Tracker quality (đo định lượng): 192 frame @3fps — xe máy độ phủ 98,3% (từ 95,8%), fragment giảm, không tăng false positive.",
    ]},

    {"type": "heading", "text": "5.6. End-to-end testing", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Chạy pipeline trên video thực tế có ground-truth: aziz1.MP4 → 2 vi phạm đúng (xe vượt đèn đỏ, frame 18 track 3 + frame 83 track 4); 16h30.25.9.22.mp4 → kiểm tra track + đèn; 20221003-102556.mp4 → 0 vi phạm giả (ground-truth không có vi phạm — đối chứng false positive). Luồng web: login → dashboard → node → kẻ vạch → chờ vi phạm → review → duyệt — đã chạy qua trên stack Docker thật."},

    {"type": "heading", "text": "5.7. Security testing", "level": 2},
    {"type": "bullet_list", "items": [
        "Quét secret hard-code toàn repo (py/java/ts/yml/sh/json): 0 kết quả; .env đã gitignore; chỉ .env.example trong git.",
        "Login: sai mật khẩu → 401 thông điệp chung (không lộ user tồn tại); tài khoản khoá → 401; JWT rác → 401 mọi endpoint bảo vệ.",
        "RBAC: 13/13 kịch bản theo vai trò (OFFICER không hiệu chuẩn được, OPERATOR không duyệt được, ADMIN đủ) — 403 đúng chỗ.",
        "Ingest: thiếu/sai X-Ingest-Token → 401; so sánh constant-time hai phía.",
        "Edge: token sai → 401; vượt 30 req/phút → 429 + header X-RateLimit-*; origin lạ → CORS chặn; mọi response có security headers.",
        "SQL guard AI: câu hỏi sinh DROP/DELETE → chặn, trả lỗi, không thực thi; chỉ SELECT + LIMIT 50.",
        "Docker: web chạy user nextjs non-root; cổng DB/MinIO không expose internet trong hướng dẫn triển khai.",
    ]},

    {"type": "heading", "text": "5.8. Performance testing", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Số liệu hiệu năng đo thực tế trên RTX 2060 (FP16) — không có số liệu bịa:"},
    {"type": "table", "header": ["Hạng mục", "Giá trị đo", "Bối cảnh"], "rows": [
        ["FPS detect YOLO26m vehicle", "73,4 FPS", "FP16, conf 0,30, imgsz mặc định"],
        ["Độ trễ phân loại đèn/ảnh", "~0,3 ms", "YOLO26n-cls imgsz 64"],
        ["OCR biển số/ảnh", "~21 ms", "CPU onnxruntime (GPU bản CUDA 13 chưa tương thích máy 12.4)"],
        ["Độ trễ chuyển đèn (tối ưu)", "0,66 s @6fps; 0,67 s @3fps", "từ 1,00 s / 2,3 s trước khi scale theo FPS nguồn"],
        ["Độ phủ xe máy", "98,3%", "video aziz1 (từ 95,8%)"],
        ["Tải HTTP trung tâm", "[THIẾU DỮ LIỆU — CẦN BENCHMARK]", "chưa đo concurrent users/nodes — cần k6/wrk"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 5.2 (phần hiệu năng). Số liệu đo thực tế — mục chưa đo được ghi rõ thiếu dữ liệu"},

    {"type": "heading", "text": "5.9. Test cases", "level": 2},
    {"type": "table", "header": ["Mã", "Mục tiêu kiểm thử", "Kỳ vọng", "Kết quả"], "rows": [
        ["TC-01", "Xe cắt vạch khi đèn đỏ ổn định", "1 hồ sơ đúng track/frame", "Đạt"],
        ["TC-02", "Đèn xanh/vàng + cắt vạch", "0 hồ sơ", "Đạt"],
        ["TC-03", "Xe ngược hướng giám sát", "0 hồ sơ", "Đạt"],
        ["TC-04", "Xe chạm deadband rồi lùi", "không vi phạm sai", "Đạt"],
        ["TC-05", "Cross-and-back", "chỉ 1 vi phạm", "Đạt"],
        ["TC-06", "Chưa hiệu chuẩn + đèn đỏ 28 frame", "0 → kẻ vạch → 1 hồ sơ", "Đạt"],
        ["TC-07", "Frame hỏng giữa luồng", "pipeline tiếp tục, không sập", "Đạt"],
        ["TC-08", "Batch trùng event_id ×2", "bản 2 bị skip không lỗi", "Đạt"],
        ["TC-09", "Mất mạng → outbox → reconnect", "flush đủ, đúng thứ tự", "Đạt"],
        ["TC-10", "OCR nhầm O/0, I/1", "repair đúng hoặc None", "Đạt"],
        ["TC-11", "Hiệu chuẩn POST qua proxy", "edge nhận, GET đúng", "Đạt"],
        ["TC-12", "AI sinh SQL nguy hiểm", "chặn, không thực thi", "Đạt"],
        ["TC-13", "Login sai / token rác", "401 không lộ thông tin", "Đạt"],
        ["TC-14", "RBAC 3 vai trò × endpoint nhạy cảm", "403/200 đúng vai trò", "Đạt"],
        ["TC-15", "Edge: token sai + vượt rate limit", "401 / 429 + headers", "Đạt"],
        ["TC-16", "Hồi quy aziz1.MP4", "2 vi phạm đúng", "Đạt"],
        ["TC-17", "Hồi quy 20221003 (đối chứng)", "0 vi phạm giả", "Đạt"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 5.1 (gộp). Các ca kiểm thử chính"},

    {"type": "heading", "text": "5.10. Kết quả", "level": 2},
    {"type": "table", "header": ["Nhóm kiểm thử", "Số ca", "Kết quả"], "rows": [
        ["Unit test pytest (9/2026)", "85", "85/85 ĐẠT — 0,87 giây"],
        ["Maven test-compile (Docker chính thức)", "toàn bộ 43 file Java", "ĐẠT — exit 0"],
        ["TypeScript + next build (CI)", "toàn bộ web", "ĐẠT — exit 0"],
        ["Ruff lint tests/", "toàn bộ test", "ĐẠT — 0 lỗi"],
        ["API + RBAC + bảo mật (thực hiện theo phiên)", "78 ca", "100% đạt (đã ghi nhận trong tài liệu dự án)"],
        ["End-to-end video", "3 video chính + 1 đối chứng", "đúng số vi phạm ground-truth"],
        ["Vận hành Docker", "khởi động/restart/offline→online", "đạt"],
        ["Load test HTTP", "0", "CHƯA THỰC HIỆN — cần k6/wrk (trung thực ghi nhận)"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 5.2. Kết quả tổng hợp kiểm thử"},
    {"type": "paragraph", "style": "Body", "text": "Đánh giá mô hình (đo trên tập validation, GPU RTX 2060, FP16) — Bảng 5.3:"},
    {"type": "table", "header": ["Mô hình", "P", "R", "mAP50", "mAP50-95", "Tốc độ"], "rows": [
        ["yolo26m_vehicle (fine-tune 4 lớp)", "0,902", "0,894", "0,944", "0,680", "73,4 FPS"],
        ["yolo26m gốc COCO (đối chứng)", "0,030", "0,078", "0,010", "0,002", "69,2 FPS"],
        ["traffic_light_cls (YOLO26n-cls)", "—", "—", "top-1 val 100%", "—", "0,3 ms/ảnh"],
        ["license_plate_yolo26", "0,988", "0,980", "0,993", "0,899", "21 ms/ảnh"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 5.3. Kết quả benchmark các mô hình (nguồn: scripts/benchmark_results.csv)"},

    {"type": "heading", "text": "5.11. Đánh giá", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Kết quả nổi bật: (1) fine-tune phương tiện đạt mAP50 0,944 so với 0,010 của COCO gốc trên cùng tập val — bằng chứng định lượng việc fine-tune bắt buộc với miền xe máy Việt Nam; (2) fusion đèn đạt 99,0% trên video domain-shift, vi phạm giả giảm 6→1; (3) hệ thống không mất hồ sơ qua các kịch bản mất mạng/restart; (4) 85/85 unit test + build sạch 2 ngôn ngữ + CI xanh. Điểm cần cải thiện (trung thực): chưa có load test HTTP, integration test chưa tự động hoá, độ phủ OCR chưa đo bằng ground-truth biển độc lập."},
    {"type": "page_break"},

    # ================= CHƯƠNG 6 =================
    {"type": "heading", "text": "CHƯƠNG 6. TRIỂN KHAI VÀ VẬN HÀNH", "level": 1},

    {"type": "heading", "text": "6.1. Production architecture", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Kiến trúc sản xuất giống kiến trúc triển khai chương 3.10 cộng reverse proxy Nginx/Caddy terminates TLS trước web-dashboard; tường lửa chỉ mở 443; PostgreSQL/MinIO chỉ truy cập trong mạng Docker nội bộ. Mọi secret qua .env. Như thể hiện trong Hình 6.1."},
    {"type": "image", "path": DIAG + "/deployment.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 6.1. Kiến trúc triển khai sản xuất (trùng Hình 3.6 — reverse proxy TLS ở trước web)"},

    {"type": "heading", "text": "6.2. Deployment", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Các bước chính (chi tiết từng lệnh trong docs/deployment/production-guide.md): sinh secret bằng openssl → viết .env → ./start.sh → verify checklist 6 bước (container healthy, /api/health, login đúng mật khẩu mới, node register, kẻ vạch bật violations, idempotent ingest) → cấu hình crontab backup. Rollback: git checkout phiên bản trước + ./start.sh (rebuild) + restore DB từ backup."},

    {"type": "heading", "text": "6.3. Docker", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "5 container, restart unless-stopped toàn bộ, healthcheck postgres/central (10s interval), web phụ thuộc central healthy (tránh 500 lúc boot — Spring khởi động ~30 giây), edge nhận GPU qua device reservations. Volume bền vững 4 loại dữ liệu (postgres, minio, outbox, hls). Sau kiểm định 9/2026: toàn bộ credential hạ tầng (postgres, MinIO) đọc từ biến môi trường với default chỉ dùng dev."},

    {"type": "heading", "text": "6.4. Environment", "level": 2},
    {"type": "table", "header": ["Biến", "Tầng", "Bắt buộc production", "Ý nghĩa"], "rows": [
        ["JWT_SECRET", "central", "Có (≥32 ký tự)", "chữ ký access token — fail-fast nếu thiếu"],
        ["JWT_TTL_SECONDS", "central", "Không (mặc định 43200)", "tuổi JWT"],
        ["JWT_REFRESH_TTL_SECONDS", "central", "Không (7 ngày)", "tuổi refresh token"],
        ["INGEST_TOKEN", "central + edge", "Có", "token node đẩy hồ sơ (X-Ingest-Token)"],
        ["ADMIN_USERNAME / ADMIN_PASSWORD", "central", "Có (đổi default)", "tài khoản admin seed lần đầu"],
        ["DB_PASS / DB_HOST / DB_NAME / DB_USER", "central + postgres", "Có", "PostgreSQL"],
        ["MINIO_*", "central + minio", "Có (đổi default)", "object storage ảnh bằng chứng"],
        ["GEMINI_API_KEY", "central", "Có (nếu dùng AI)", "Google AI Studio"],
        ["EDGE_API_TOKEN", "edge", "Có", "guard POST /action/* (X-Edge-Token)"],
        ["EDGE_REQUIRE_TOKEN", "edge", "Có (=true)", "bật kiểm tra token bắt buộc"],
        ["EDGE_ALLOWED_ORIGINS", "edge", "Có", "CORS whitelist control-plane"],
        ["NODE_ID", "edge", "Không (edge-node-01)", "định danh node + event_id"],
        ["VIDEO_INPUT / YOLO_MODEL_PATH", "edge", "Có", "nguồn video + weights"],
        ["OUTBOX_BATCH_SIZE / FLUSH_INTERVAL", "edge", "Không (20 / 5s)", "cấu hình gửi batch"],
        ["CENTRAL_SERVER_URL", "edge + web", "Có", "địa chỉ trung tâm"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 4.2. Biến môi trường cấu hình hệ thống (đầy đủ trong production-guide)"},

    {"type": "heading", "text": "6.5. Reverse proxy", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Nginx cấu hình khuyến nghị có trong production-guide: proxy_pass về web :3000, bật http2, X-Forwarded-Proto https, tắt proxy_buffering + tăng read_timeout 300s cho luồng HLS (/edge-api), redirect 80→443. Sau khi có domain cần cập nhật EDGE_ALLOWED_ORIGINS cho khớp origin mới."},

    {"type": "heading", "text": "6.6. HTTPS", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "TLS terminate tại reverse proxy bằng Let's Encrypt (certbot renew tự động). Trong mạng nội bộ Docker các service gọi nhau bằng http://service-name — chấp nhận được vì mạng bridge cô lập; nếu triển khai nhiều máy phải bật TLS nội bộ (mục 7.4 hướng phát triển)."},

    {"type": "heading", "text": "6.7. Monitoring", "level": 2},
    {"type": "bullet_list", "items": [
        "Docker healthcheck: postgres (pg_isready), central (wget /api/health), minio (mc ready) — interval 10s, web đợi central healthy mới khởi động.",
        "Edge /health tự chế 503 khi pipeline stale >10 giây, trả metrics in-process: frames_processed, violations_detected, fps.",
        "Log chuẩn SLF4J/loguru theo mức INFO/WARN/ERROR tiếng Việt — ./start.sh logs để xem toàn bộ.",
        "Chưa có Prometheus/Grafana + alert — ghi nhận như hạn chế mức MEDIUM trong checklist readiness.",
    ]},

    {"type": "heading", "text": "6.8. Logging", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Mỗi tầng log theo cấu trúc riêng: edge — logging Python theo module (outbox, sender, pipeline) kèm event_id truy vết; central — SLF4J có context (node, event_id, username) cho audit đăng nhập/duyệt; web — console của Next (proxy lỗi hiển thị rõ ECONNREFUSED). Mọi timestamp ISO múi giờ Việt Nam."},

    {"type": "heading", "text": "6.9. Backup", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "scripts/backup_central.sh (thêm trong kiểm định 9/2026): pg_dump custom format (nén, restore chọn bảng) + mirror bucket MinIO violations; output ./backups/backup_YYYYmmdd_HHMMSS; tự xoá giữ 14 bản gần nhất; crontab đề xuất 2 giờ sáng hằng ngày. Header script có lệnh restore cho từng loại dữ liệu."},

    {"type": "heading", "text": "6.10. Recovery", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Kịch bản khôi phục: (1) dừng stack; (2) thay volume postgres_data bằng bản restore từ pg_dump; (3) sync lại bucket MinIO từ backup; (4) ./start.sh --no-build; (5) verify counts + media blob. Điểm chờ kiểm chứng: chưa chạy restore test thực tế đầy đủ — nêu trong readiness (HIGH #2)."},

    {"type": "heading", "text": "6.11. Security", "level": 2},
    {"type": "bullet_list", "items": [
        "Đổi toàn bộ secret mặc định (mục 6.4) — fail-fast hỗ trợ: JWT yếu từ chối khởi động.",
        "Tường lửa chỉ mở 443 (+3000 nếu nội bộ); postgres/minio không expose ra internet.",
        "RBAC + JWT 12h + refresh rotation + reuse-detection; log cảnh báo replay token để điều tra.",
        "Edge control-plane: token + rate limit + CORS whitelist + security headers.",
        "Kiểm định 9/2026: quét secret hard-code = 0; credential compose đã parameterize.",
        "Còn mở (HIGH): central chưa rate-limit /api/auth/login — kẻ tấn công có thể dò mật khẩu; cần bucket4j hoặc filter tương tự edge.",
    ]},

    {"type": "heading", "text": "6.12. Production readiness checklist", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Điểm tổng kết readiness 2026-09-06: 78/100. Phân loại vấn đề còn lại: CRITICAL 0 (đã dọn trong đợt kiểm định), HIGH 2 (rate-limit login, restore-test + resource limits), MEDIUM 5 (Flyway, monitoring tập trung, integration test tự động, component test web, lockout account), LOW 3 (ruff edge_node stylistic, image 10,4GB, container root). Kết luận: sẵn sàng vận hành nội bộ/demo và bảo vệ đồ án; CHƯA kết luận production-ready cho internet công cộng cho tới khi xử lý 2 HIGH. Chi tiết 22 mục checklist trong docs/production-readiness.md."},
    {"type": "table", "header": ["Nhóm mục checklist", "Điểm", "Trạng thái"], "rows": [
        ["Build, env, compose, health, restart, volumes, CI/CD, rollback", "đạt tuyệt đại đa số", "✅"],
        ["Backup (đã có script)", "chưa restore-test", "⚠ HIGH"],
        ["Rate limiting login trung tâm", "chưa có", "⚠ HIGH"],
        ["Migration Flyway", "ddl-auto=update", "⚠ MEDIUM"],
        ["Monitoring tập trung + error tracking", "healthcheck + log", "⚠ MEDIUM"],
        ["Resource limits container", "chưa đặt", "⚠ HIGH (kèm restore-test)"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 6.1. Checklist readiness production — tóm tắt theo nhóm (chi tiết 22 mục trong docs/production-readiness.md)"},
    {"type": "page_break"},

    # ================= CHƯƠNG 7 =================
    {"type": "heading", "text": "CHƯƠNG 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", "level": 1},

    {"type": "heading", "text": "7.1. Kết quả đạt được", "level": 2},
    {"type": "bullet_list", "items": [
        "Hệ thống ba tầng hoàn chỉnh vận hành end-to-end bằng một lệnh Docker Compose trên GPU phổ thông; quy mô mã nguồn thực đo: 5.846 dòng Python (node biên) + 3.582 dòng Java (trung tâm) + 6.591 dòng TS/TSX (web).",
        "Ba mô hình fine-tune đạt chỉ số thực: phương tiện mAP50 0,944 @73 FPS (so 0,010 của COCO gốc — bằng chứng fine-tune bắt buộc); đèn top-1 100% val + 99,0% thực tế sau fusion; biển số mAP50 0,993.",
        "Độ trễ chuyển đèn tối ưu 0,66 s @6fps (từ 1,00 s) và 0,67 s trên camera 3fps (từ 2,3 s) nhờ scale ngưỡng theo FPS nguồn.",
        "Giao hàng bền vững: durable outbox + batch idempotent, kiểm chứng mất mạng/restart không mất hồ sơ, không trùng lặp.",
        "Bảo mật nhiều lớp: JWT + RBAC 3 vai trò + refresh rotation + reuse detection; ingest token constant-time; edge token + rate limit + CORS + security headers; AI guard SELECT-only; quét secret 0 kết quả.",
        "Trợ lý AI Text-to-SQL tiếng Việt — câu hỏi tự nhiên → SQL an toàn → bảng/biểu đồ ngay trên dashboard.",
        "Hiệu chuẩn từ xa hoàn chỉnh (vạch dừng, hướng, vùng đèn) trực tiếp trên video live, hiệu lực tức thì không restart.",
        "Kiểm định trước sản xuất 19 giai đoạn hoàn tất: các lỗi HIGH (credential compose, truy vấn quét toàn bảng, thiếu backup, status tuỳ ý, shutdown đột ngột) đã sửa có giải thích; readiness 78/100.",
        "Bộ tài liệu đầy đủ: README cập nhật, tài liệu API, hướng dẫn production, test report, changelog, 10 diagram Mermaid, 21 diagram UML tổng cộng.",
    ]},

    {"type": "heading", "text": "7.2. Hạn chế", "level": 2},
    {"type": "bullet_list", "items": [
        "Mới kiểm thử 1 node biên + video tệp phát vòng lặp; chưa chứng minh nhiều node/camera RTSP thực địa đồng thời.",
        "OCR chạy CPU (onnxruntime-gpu cần CUDA 13, máy 12.4) — ~21 ms/ảnh, phụ thuộc GPU single-node.",
        "Vùng đèn cố định sau hiệu chuẩn; camera lệch góc (gió, va chạm) cần vẽ lại — chưa tự phát hiện lệch.",
        "Chưa kiểm chứng ánh sáng xấu (mưa đêm, ngược sáng) quy mô lớn; độ phủ OCR chưa đo bằng ground-truth độc lập.",
        "AI phụ thuộc Gemini bên ngoài (key, chi phí, độ trễ mạng) — chưa có fallback offline.",
        "Quản lý user chưa có CRUD UI (seed qua env/DB); JWT chưa có revoke-list (access token tự chấm dứt 12h; refresh có revoke).",
        "Hai HIGH còn mở cho internet công cộng: rate-limit login trung tâm, restore-test + resource limits.",
        "Load test HTTP chưa thực hiện — chưa ước lượng được băng tải/độ trễ theo số node (chỉ ước tính logic ở 7.5).",
    ]},

    {"type": "heading", "text": "7.3. Bài học", "level": 2},
    {"type": "bullet_list", "items": [
        "Domain-shift là quy tắc chứ không ngoại lệ: mô hình 100% trên tập val vẫn chỉ 69% trên video thật — phải đo trên dữ liệu mục tiêu và thiết kế fusion (đèn: YOLO + HSV + vị trí bóng vật lý).",
        "Mô hình fine-tune đổi không gian lớp — mọi lọc theo id COCO cũ im lặng sai hoàn toàn (mất 100% ô tô); phải verify class-map sau mỗi lần đổi model.",
        "Ghi dữ liệu trước khi gọi mạng (outbox pattern) biến \"hệ thống có thể mất dữ liệu\" thành \"hệ thống chỉ có thể trễ\" — thay đổi tư duy thiết kế đáng giá nhất của đồ án.",
        "Chi tiết giao thức nhỏ có thể phá cả luồng: HTTP/2 upgrade h2c làm uvicorn rơi POST body → 422 → 502 ở proxy; phải ép HTTP/1.1.",
        "Danh sách đường dẫn ingest tập trung MỘT nguồn sự thật: hai filter từng lệch nhau làm mất 263 ảnh — bài học về single source of truth.",
        "Kiểm định phải chạy THẬT: 85 test + build Docker + quét secret nói nhiều hơn mọi cam kết giấy.",
    ]},

    {"type": "heading", "text": "7.4. Hướng phát triển", "level": 2},
    {"type": "bullet_list", "items": [
        "Đa node + TLS nội bộ giữa các máy; cân bằng tải ingest; giám sát tập trung Prometheus + Grafana + alert Telegram.",
        "Camera RTSP thực địa: auto-recovery, phát hiện lệch khung hình nhắc hiệu chuẩn lại; chế độ drop-frame realtime đã có sẵn.",
        "Mở rộng loại vi phạm trên cùng pipeline: vượt tốc độ (hai điểm đo), lấn làn, đi ngược chiều — tận dụng tracker + tripwire hiện có.",
        "OCR: nâng onnxruntime-gpu tương thích CUDA hoặc TensorRT; huấn luyện OCR trên tập biển Việt Nam thực tế.",
        "Kiểm thử: tự động hoá integration (JUnit + Testcontainers), component test web (Vitest), load test k6; Flyway migration; rate-limit + lockout login.",
        "Nghiệp vụ: xuất biên bản PDF từ hồ sơ đã duyệt, API đối soát hệ thống xử phạt; explainable AI (truy hồi ảnh + metadata theo event).",
    ]},

    {"type": "heading", "text": "7.5. Khả năng mở rộng", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Phân tích khả năng mở rộng theo số node biên (dựa trên thiết kế — chưa có benchmark tải, các con số dưới là ước lượng kiến trúc có ghi chú): mỗi node gửi ~20 hồ sơ JSON/lần flush 5 giây + 1 ảnh JPEG (~100–300 KB) mỗi hồ sơ — băng thông ingest tuyến tính theo node và thấp (node vi phạm trung bình vài hồ sơ/phút, không phải vài trăm). Trung tâm stateless JWT + pool 10 connection PostgreSQL + batch idempotent → có thể chạy nhiều instance central sau reverse proxy khi cần; điểm nghẽn thực tế sẽ là PostgreSQL ghi violations (giải pháp: partition theo tháng, read replica cho thống kê) và băng thông đọc/ghi ảnh MinIO. Về phát hiện: mỗi node tự xử lý GPU riêng — mở rộng node không tăng tải tính toán trung tâm. Ước lượng thận trọng (chưa benchmark): một central instance + PostgreSQL mặc định đủ cho hàng chục node với tần suất vi phạm thực tế; để chứng minh con số chính xác cần k6 benchmark trên stack thật — đã ghi nhận là việc cần làm (7.4)."},
    {"type": "page_break"},

    # ================= TÀI LIỆU THAM KHẢO =================
    {"type": "heading", "text": "TÀI LIỆU THAM KHẢO", "level": 1},
    {"type": "numbered_list", "items": [
        "Zhang, Y., Sun, P., Jiang, Y., et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.",
        "Khanam, R. & Hussain, M. (2024). YOLOv11: An Overview of the Key Architectural Enhancements. arXiv:2410.17725.",
        "Jocher, G., Qiu, J. (2024). Ultralytics YOLO (v8.x/26). Ultralytics Solutions. https://docs.ultralytics.com",
        "Møgelmose, A., Trivedi, M. M., Moeslund, T. B. (2012). Vision-based Traffic Sign Detection and Analysis for Intelligent Driver Assistance Systems. IEEE TPAMI — LISA dataset.",
        "Silva, C. et al. (2023). fast-plate-ocr: License Plate OCR library. https://github.com/ankandela/fast-plate-ocr",
        "AhmadYahya97 (2023). Fully-Automated-red-light-Violation-Detection. GitHub repository.",
        "Bộ Giao thông Vận tải (2023). Thông tư 24/2023/TT-BGTVT quy định về cấp, thu hồi đăng ký, biển số xe của xe cơ giới.",
        "Fields, J. et al. (2024). Spring Boot Reference Documentation (3.3.x). VMware Tanzu. https://docs.spring.io",
        "Vercel (2025). Next.js Documentation (App Router, v16). https://nextjs.org/docs",
        "MinIO (2025). MinIO Object Storage Documentation. https://min.io/docs",
        "FFmpeg (2025). FFmpeg HLS muxer documentation. https://ffmpeg.org",
        "Patterson, D. et al. (2022). Docker and Kubernetes Container Orchestration fundamentals — Docker Compose documentation. https://docs.docker.com",
        "Richardson, C. (2018). Microservices Patterns: pattern Outbox / Transactional Inbox. Manning.",
        "OWASP Foundation (2025). Cheat Sheet Series — Authentication, Docker Security. https://cheatsheetseries.owasp.org",
    ]},

    # ================= PHỤ LỤC =================
    {"type": "heading", "text": "PHỤ LỤC", "level": 1},
    {"type": "heading", "text": "Phụ lục A — Tài liệu API (trích)", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Tài liệu API đầy đủ (40+ endpoint, body/response chi tiết, bảng mã lỗi) trong docs/api/README.md của repository. Trích các endpoint lõi:"},
    {"type": "bullet_list", "items": [
        "POST /api/violations/batch — body {violations: [ViolationRequest...]}; header X-Node-ID + X-Ingest-Token; response 201 {accepted, duplicates, failed, acceptedEventIds, duplicateEventIds}.",
        "POST /api/v1/violations/{eventId}/media — multipart file; response 201 {object_name, url: /api/v1/violations/{id}/media/blob}.",
        "GET /api/violations/page?page=0&size=20&status=pending,approved — response {content, page, size, total_elements, total_pages, first, last}.",
        "PATCH /api/violations/{id}/status — body {status: approved|rejected|confirmed}; RBAC ADMIN/OFFICER.",
        "POST /api/ai/query — body {question} (≤500 ký tự); response {question, sql, columns, rows, count, chartType, error?}.",
        "POST /action/stop-line (edge) — body {x1,y1,x2,y2,direction}; header X-Edge-Token; rate limit 30 req/phút/IP.",
    ]},
    {"type": "heading", "text": "Phụ lục B — Schema CSDL (trích)", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Xem chương 3.14 (Bảng 3.6–3.8) và ERD Hình 3.12. File đầy đủ: docs/diagrams/erd.mmd trong repository."},
    {"type": "heading", "text": "Phụ lục C — Cấu hình (biến môi trường)", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Xem Bảng 4.2 chương 6.4 + file central_server/.env.example trong repository."},
    {"type": "heading", "text": "Phụ lục D — Test cases", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Xem chương 5.9 (Bảng 5.1) + docs/test-report.md trong repository (85 unit test + 78 ca tích hợp/bảo quy tắc đã ghi nhận)."},
    {"type": "heading", "text": "Phụ lục E — Diagram", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "10 diagram Mermaid mới (docs/diagrams/): system-architecture, three-tier, deployment, api-surface, erd, use-case, edge-lifecycle, sequence-edge-data, sequence-auth, sequence-calibration + 11 diagram UML PlantUML Việt hoá (docs/report-assets/vn/): use-case, use-case-officer, use-case-operator, activity-detection, activity-calibration, activity-review, sequence-violation, sequence-calibration, sequence-ai, sequence-outbox, class-diagram, component, data-flow, state-violation, state-light, security-layers, erd, deployment."},
    {"type": "heading", "text": "Phụ lục F — Screenshot", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "9 screenshot thật từ stack đang chạy (docs/report-assets/shots/): login, dashboard, violations, review, nodes, cameras, node-calibration + 2 ảnh sau đăng nhập. Các hình chèn ở chương 4 đều từ thư mục này."},
]
