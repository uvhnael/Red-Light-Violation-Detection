# -*- coding: utf-8 -*-
"""Nội dung báo cáo đồ án RLVD (bản pre-production) — phần 1: bìa → Chương 3.

Mọi số liệu trong file này đều取 từ nguồn thực: codebase (LOC), pytest 85 pass,
benchmark_results.csv, audit 2026-09-06. Không có số liệu bịa.
"""

ASSETS = "/home/uvhnael/projects/Red-Light-Violation-Detection/docs/report-assets"
DIAG = "/home/uvhnael/projects/Red-Light-Violation-Detection/docs/diagrams"

BLOCKS_1 = [
    # ================= TRANG BÌA =================
    {"type": "paragraph", "style": "TitleBig", "text": "BÁO CÁO ĐỒ ÁN KỸ SƯ PHẦN MỀM"},
    {"type": "paragraph", "style": "TitleSub", "text": "HỆ THỐNG PHÁT HIỆN VI PHẠM VƯỢT ĐÈN ĐỎ"},
    {"type": "paragraph", "style": "TitleSub2", "text": "(Red-Light Violation Detection — RLVD)"},
    {"type": "paragraph", "style": "TitleSub2", "text": "Kiến trúc ba tầng Edge Node — Central Server — Web Dashboard"},
    {"type": "paragraph", "style": "TitleMeta", "text": "Bản kiểm định trước sản xuất (Pre-Production Audit)"},
    {"type": "paragraph", "style": "TitleMeta", "text": "Tháng 9, năm 2026"},
    {"type": "page_break"},

    # ================= LỜI CẢM ƠN =================
    {"type": "heading", "text": "LỜI CẢM ƠN", "level": 1},
    {"type": "paragraph", "style": "Body", "text": "Nhóm thực hiện đề tài xin gửi lời cảm ơn đến giảng viên hướng dẫn đã định hướng kỹ thuật, góp ý về kiến trúc hệ thống và các yêu cầu nghiệp vụ trong suốt quá trình thực hiện đồ án; cảm ơn cộng đồng mã nguồn mở Ultralytics, supervision, fast-plate-ocr, Spring Boot, Next.js và MinIO — các công cụ nền tảng mà đồ án xây dựng dựa trên; cùng các bạn đã hỗ trợ thu thập dữ liệu video thực tế tại các nút giao thông để huấn luyện và kiểm thử mô hình."},
    {"type": "page_break"},

    # ================= TÓM TẮT =================
    {"type": "heading", "text": "TÓM TẮT", "level": 1},
    {"type": "paragraph", "style": "Body", "text": "Đồ án xây dựng hệ thống phát hiện vi phạm vượt đèn đỏ (Red-Light Violation Detection — RLVD) theo kiến trúc ba tầng: node biên (edge node) xử lý thị giác máy tính thời gian thực ngay tại camera, máy chủ trung tâm (central server) lưu trữ hồ sơ và cung cấp API, và bảng điều khiển web (web dashboard) phục vụ giám sát — hiệu chuẩn — duyệt hồ sơ."},
    {"type": "paragraph", "style": "Body", "text": "Tại node biên, pipeline dùng mô hình YOLO26m fine-tune (mAP50 = 0,944) phát hiện phương tiện 4 lớp, ByteTrack theo dõi đa đối tượng, bộ phân loại đèn YOLO26n-cls kết hợp fusion HSV + vị trí bóng đèn (99,0% chính xác trên video thực tế có domain-shift), và OCR biển số Việt Nam có kiểm tra cấu trúc theo Thông tư 24/2023/TT-BGTVT. Hồ sơ vi phạm được ghi vào durable outbox SQLite trước mọi thao tác mạng, đẩy theo batch idempotent — hệ thống không mất hồ sơ khi mất mạng hoặc khởi động lại."},
    {"type": "paragraph", "style": "Body", "text": "Máy chủ trung tâm Spring Boot (Java 17) lưu PostgreSQL + MinIO, bảo vệ bằng JWT phân vai ba cấp (ADMIN/OPERATOR/OFFICER) và ingest token riêng cho node biên; tích hợp trợ lý AI Text-to-SQL cho phép hỏi dữ liệu bằng tiếng Việt với lớp chặn SQL nguy hiểm. Bảng điều khiển Next.js cung cấp thống kê, duyệt human-in-the-loop, camera live HLS và hiệu chuẩn vạch dừng — vùng đèn — hướng giám sát trực tiếp trên video, có hiệu lực tức thời."},
    {"type": "paragraph", "style": "Body", "text": "Kiểm định trước sản xuất (9/2026) thực hiện quét toàn bộ 3 tầng: 85/85 unit test đạt, biên dịch Java/TypeScript sạch, quét security không phát hiện secret hard-code; các vấn đề phát hiện (credential hard-code trong Docker Compose, truy vấn quét toàn bảng, thiếu backup) đã được sửa. Điểm readiness production đạt 78/100 — đủ cho vận hành nội bộ và bảo vệ đồ án, còn 2 vấn đề mức HIGH cần xử lý trước khi public internet (rate-limit đăng nhập, kiểm thử khôi phục dữ liệu)."},
    {"type": "paragraph", "style": "Body", "text": "Từ khoá: thị giác máy tính, YOLO, ByteTrack, OCR biển số, edge computing, kiến trúc ba tầng, Spring Boot, Next.js, Docker, PostgreSQL, MinIO, JWT, outbox pattern."},
    {"type": "page_break"},

    # ================= ABSTRACT =================
    {"type": "heading", "text": "ABSTRACT", "level": 1},
    {"type": "paragraph", "style": "Body", "text": "This capstone project builds a red-light violation detection system (RLVD) with a strict three-tier architecture: an edge node running real-time computer vision at the camera, a central server storing violation records and exposing APIs, and a web dashboard for monitoring, remote calibration and human-in-the-loop review."},
    {"type": "paragraph", "style": "Body", "text": "The edge pipeline uses a fine-tuned YOLO26m detector (mAP50 = 0.944) for 4 vehicle classes, ByteTrack multi-object tracking, a YOLO26n-cls traffic-light classifier fused with HSV evidence and lamp position (99.0% accuracy on a domain-shifted real-world video), and Vietnamese license-plate OCR validated against Circular 24/2023/TT-BGTVT. Violations are written to a durable SQLite outbox before any network call and delivered in idempotent batches — no record is lost across outages or restarts."},
    {"type": "paragraph", "style": "Body", "text": "The Spring Boot central server (Java 17) persists to PostgreSQL and MinIO, protected by role-based JWT authentication (ADMIN/OPERATOR/OFFICER) plus a dedicated ingest token for edge nodes, and integrates a Vietnamese Text-to-SQL AI assistant with a dangerous-SQL guard. The Next.js dashboard provides statistics, review workflow, live HLS camera view, and stop-line/light-ROI/direction calibration drawn directly on live video with immediate effect."},
    {"type": "paragraph", "style": "Body", "text": "The pre-production audit (Sept 2026) scanned all three tiers: 85/85 unit tests pass, Java/TypeScript builds are clean, and a security scan found zero hard-coded secrets; discovered issues (hard-coded infrastructure credentials in Docker Compose, a full-table-scan query, missing backups) were fixed. Production readiness scores 78/100 — sufficient for internal operation and thesis defense, with 2 HIGH items (login rate limiting, restore testing) remaining before public internet exposure."},
    {"type": "paragraph", "style": "Body", "text": "Keywords: computer vision, YOLO, ByteTrack, license plate OCR, edge computing, three-tier architecture, Spring Boot, Next.js, Docker, PostgreSQL, MinIO, JWT, outbox pattern."},
    {"type": "page_break"},

    # ================= MỤC LỤC =================
    {"type": "heading", "text": "MỤC LỤC", "level": 1},
    {"type": "toc"},
    {"type": "page_break"},

    # ================= DANH SÁCH HÌNH =================
    {"type": "heading", "text": "DANH SÁCH HÌNH", "level": 1},
    {"type": "numbered_list", "items": [
        "Hình 3.1 — Sơ đồ use case tổng thể hệ thống",
        "Hình 3.2 — Kiến trúc tổng thể hệ thống (System Architecture)",
        "Hình 3.3 — Kiến trúc ba tầng (Three-Tier Architecture)",
        "Hình 3.4 — Kiến trúc Central Server",
        "Hình 3.5 — Vòng đời node biên (Edge Node lifecycle)",
        "Hình 3.6 — Sơ đồ triển khai (Deployment Diagram)",
        "Hình 3.7 — Sơ đồ thành phần (Component Diagram)",
        "Hình 3.8 — Sơ đồ sequence gửi dữ liệu Edge → Central (outbox)",
        "Hình 3.9 — Sơ đồ sequence xác thực (login, refresh rotation)",
        "Hình 3.10 — Sơ đồ sequence hiệu chuẩn từ xa",
        "Hình 3.11 — Sơ đồ luồng dữ liệu (Data Flow)",
        "Hình 3.12 — Sơ đồ thực thể dữ liệu (ERD)",
        "Hình 4.1 — Trang đăng nhập web dashboard",
        "Hình 4.2 — Trang thống kê tổng quan",
        "Hình 4.3 — Trang tra cứu vi phạm",
        "Hình 4.4 — Trang duyệt hồ sơ",
        "Hình 4.5 — Trang quản lý node biên",
        "Hình 4.6 — Trang xem camera trực tuyến",
        "Hình 4.7 — Hiệu chuẩn trên video trực tuyến",
        "Hình 4.8 — Bề mặt API và phân nhóm bảo vệ",
        "Hình 5.1 — Kết quả chạy 85 unit test pytest",
        "Hình 6.1 — Sơ đồ kiến trúc sản xuất và reverse proxy",
    ]},
    {"type": "page_break"},

    # ================= DANH SÁCH BẢNG =================
    {"type": "heading", "text": "DANH SÁCH BẢNG", "level": 1},
    {"type": "numbered_list", "items": [
        "Bảng 1.1 — Cấu trúc báo cáo",
        "Bảng 2.1 — Công nghệ thực tế sử dụng trong đồ án",
        "Bảng 3.1 — Yêu cầu chức năng",
        "Bảng 3.2 — Yêu cầu phi chức năng",
        "Bảng 3.3 — Đặc tả use case UC-01 Phát hiện xe vượt đèn đỏ",
        "Bảng 3.4 — Đặc tả use case UC-02 Hiệu chuẩn từ xa",
        "Bảng 3.5 — Đặc tả use case UC-03 Duyệt hồ sơ vi phạm",
        "Bảng 3.6 — Thiết kế bảng dữ liệu violations",
        "Bảng 3.7 — Thiết kế bảng dữ liệu edge_nodes",
        "Bảng 3.8 — Thiết kế bảng users và refresh_tokens",
        "Bảng 4.1 — Danh sách API chính",
        "Bảng 4.2 — Biến môi trường cấu hình hệ thống",
        "Bảng 5.1 — Các ca kiểm thử chính",
        "Bảng 5.2 — Kết quả tổng hợp kiểm thử",
        "Bảng 5.3 — Kết quả benchmark mô hình deep learning",
        "Bảng 6.1 — Checklist readiness production và chấm điểm",
    ]},
    {"type": "page_break"},

    # ================= DANH SÁCH TỪ VIẾT TẮT =================
    {"type": "heading", "text": "DANH SÁCH TỪ VIẾT TẮT", "level": 1},
    {"type": "table", "header": ["Từ viết tắt", "Nghĩa đầy đủ"], "rows": [
        ["RLVD", "Red-Light Violation Detection — Phát hiện vi phạm vượt đèn đỏ"],
        ["YOLO", "You Only Look Once — họ mô hình phát hiện đối tượng thời gian thực"],
        ["mAP", "mean Average Precision — chỉ số đánh giá mô hình phát hiện"],
        ["OCR", "Optical Character Recognition — nhận dạng ký tự quang học"],
        ["MOT", "Multi-Object Tracking — theo dõi đa đối tượng"],
        ["IoU", "Intersection over Union — tỷ lệ giao nhau của hai hộp giới hạn"],
        ["HLS", "HTTP Live Streaming — giao thức phát video streaming phân đoạn"],
        ["API", "Application Programming Interface"],
        ["REST", "Representational State Transfer — phong cách thiết kế API"],
        ["JWT", "JSON Web Token — token xác thực có chữ ký"],
        ["RBAC", "Role-Based Access Control — phân quyền theo vai trò"],
        ["JPA", "Jakarta Persistence API — chuẩn ánh xạ đối tượng — cơ sở dữ liệu của Java"],
        ["DTO", "Data Transfer Object — đối tượng truyền dữ liệu giữa các tầng"],
        ["DAO", "Data Access Object — mẫu thiết kế truy cập dữ liệu"],
        ["ERD", "Entity Relationship Diagram — sơ đồ thực thể — quan hệ"],
        ["CSDL", "Cơ sở dữ liệu"],
        ["NFR", "Non-Functional Requirement — yêu cầu phi chức năng"],
        ["CI/CD", "Continuous Integration / Continuous Deployment"],
        ["OCR gate", "Cổng kiểm tra cấu trúc biển số sau OCR trước khi ghi nhận"],
        ["Outbox", "Kho SQLite trung gian đảm bảo giao hàng bền vững (outbox pattern)"],
        ["TT 24/2023", "Thông tư 24/2023/TT-BGTVT về đăng ký, biển số xe"],
    ], "style": "Table Grid"},
    {"type": "page_break"},

    # ================= CHƯƠNG 1 =================
    {"type": "heading", "text": "CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI", "level": 1},

    {"type": "heading", "text": "1.1. Lý do chọn đề tài", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Vi phạm đèn đỏ là một trong những hành vi vi phạm giao thông phổ biến và nguy hiểm nhất tại Việt Nam, trực tiếp gây tai nạn nghiêm trọng tại nút giao. Công tác phát hiện, xử lý hiện phụ thuộc lớn vào lực lượng trực tiếp tại hiện trường hoặc xem lại video thủ công — tốn nhân lực, chỉ phủ được phần nhỏ vi phạm thực tế và khó đảm bảo tính minh bạch khi lập hồ sơ xử phạt."},
    {"type": "paragraph", "style": "Body", "text": "Thị giác máy tính với các mô hình deep learning thế hệ YOLO đã đạt độ chính xác và tốc độ đủ để vận hành thời gian thực trên phần cứng phổ thông. Kiến trúc tính toán biên (edge computing) cho phép phân tích video ngay tại camera, giảm băng thông và giúp mở rộng theo số lượng nút giao linh hoạt. Đây là cơ sở kỹ thuật để tự động hoá giám sát vượt đèn đỏ một cách bền vững."},
    {"type": "paragraph", "style": "Body", "text": "Một hệ thống thực thụ không dừng ở nhận diện: cần OCR biển số theo chuẩn Việt Nam, truyền hồ sơ đáng tin khi mạng không ổn định, quy trình duyệt có con người (human-in-the-loop) bảo đảm tính pháp lý, giao diện quản trị điều phối nhiều camera. Việc tích hợp các thành phần đó thành hệ thống phân tán hoàn chỉnh là bài toán có giá trị ứng dụng và học thuật cao."},

    {"type": "heading", "text": "1.2. Bối cảnh", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Hệ thống được thiết kế cho bài toán giám sát nút giao thông tại Việt Nam: mật độ xe máy cao (chiếm đa số phương tiện), biển số theo quy định TT 24/2023/TT-BGTVT, điều kiện ánh sáng đa dạng. Camera quan sát cố định một hướng/làn; hệ thống cần chạy ổn định trên hạ tầng Docker với GPU NVIDIA phổ thông (đồ án dùng RTX 2060)."},

    {"type": "heading", "text": "1.3. Vấn đề cần giải quyết", "level": 2},
    {"type": "bullet_list", "items": [
        "Phát hiện chính xác phương tiện (kể cả xe máy cỡ nhỏ, độ tin cậy thấp) trong video thời gian thực tại node biên.",
        "Nhận biết trạng thái đèn tín hiệu đủ tin cậy trên dữ liệu thực tế khác biệt với tập huấn luyện (domain-shift).",
        "Xét vi phạm đúng: chỉ tính xe cắt vạch dừng khi đèn đỏ đã ổn định và xe đi đúng hướng giám sát; tránh vi phạm giả.",
        "Đọc và chuẩn hoá biển số Việt Nam với các lỗi OCR thường gặp (O/0, I/1, S/5).",
        "Truyền hồ sơ về trung tâm không mất dữ liệu khi mất mạng hoặc node khởi động lại; không trùng lặp hồ sơ.",
        "Cung cấp nghiệp vụ tra cứu, thống kê, duyệt hồ sơ và hiệu chuẩn từ xa cho vận hành viên với phân quyền chặt.",
    ]},

    {"type": "heading", "text": "1.4. Mục tiêu", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Xây dựng hệ thống phần mềm phân tán tự động phát hiện phương tiện vượt đèn đỏ từ luồng video, tạo hồ sơ vi phạm kèm ảnh bằng chứng và biển số xe, phục vụ tra cứu, thống kê và duyệt xử lý. Mục tiêu cụ thể:"},
    {"type": "bullet_list", "items": [
        "Pipeline thị giác thời gian thực tại node biên: YOLO26m fine-tune (4 lớp), ByteTrack, phân loại đèn YOLO26n-cls + fusion HSV + vị trí bóng đèn, logic vi phạm theo vạch dừng + hướng.",
        "Huấn luyện và đánh giá ba mô hình chuyên biệt: phương tiện, màu đèn, biển số; benchmark với mô hình gốc COCO.",
        "OCR biển số Việt Nam (fast-plate-ocr) với bộ kiểm tra cấu trúc theo TT 24/2023.",
        "Durable outbox SQLite + batch sender idempotent: chịu mất mạng/restart, chống trùng lặp theo event_id.",
        "Backend Spring Boot (REST API, PostgreSQL, MinIO) + dashboard Next.js: thống kê, duyệt, camera live HLS, hiệu chuẩn từ xa.",
        "Trợ lý AI Text-to-SQL tiếng Việt, chỉ cho phép SELECT.",
        "Kiểm thử toàn diện + vận hành full stack bằng Docker Compose một lệnh.",
    ]},

    {"type": "heading", "text": "1.5. Phạm vi", "level": 2},
    {"type": "bullet_list", "items": [
        "Trong phạm vi: camera cố định giám sát một hướng; nguồn vào là tệp video MP4 (mô phỏng camera, phát HLS) hoặc luồng RTSP; node biên chạy trên một máy GPU; một trung tâm, nhiều node về mặt thiết kế.",
        "Ngoài phạm vi: xử phạt hành chính tự động (chỉ tạo hồ sơ chờ duyệt); luồng nhiều hướng từ một camera; điều kiện ánh sáng cực đoan chưa kiểm chứng quy mô lớn; giám sát đêm có đèn vin không có trong dữ liệu huấn luyện.",
    ]},

    {"type": "heading", "text": "1.6. Đối tượng nghiên cứu", "level": 2},
    {"type": "bullet_list", "items": [
        "Phương tiện đường bộ 4 lớp: ô tô con, xe mô tô/gắn máy, xe van/khách, xe tải (mô hình fine-tune yolo26m_vehicle).",
        "Tín hiệu đèn 3 màu đỏ/vàng/xanh tại nút giao có camera cố định.",
        "Biển số xe cơ giới Việt Nam dạng một dãy NN-XXXXXXX (mã tỉnh 11–99, seri 2 chữ/chữ+số/1 chữ, 4–5 số).",
        "Luồng video tệp MP4/RTSP; dữ liệu thực tế ghi tại các nút giao Việt Nam.",
    ]},

    {"type": "heading", "text": "1.7. Phương pháp thực hiện", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Đồ án theo phương pháp phát triển lặp (iterative): mỗi chức năng được cài đặt, đo lường bằng số liệu thực (benchmark mô hình, đo độ trễ, đếm vi phạm trên video có ground-truth) rồi tinh chỉnh; mọi thay đổi quan trọng có kiểm thử hồi quy (85 unit test pytest + CI GitHub Actions 3 job); kiểm định trước sản xuất được thực hiện toàn diện theo quy trình 19 giai đoạn (phân tích source → audit kiến trúc → audit bảo mật → sửa lỗi nghiêm trọng → kiểm thử → báo cáo)."}, 

    {"type": "heading", "text": "1.8. Kết quả đạt được", "level": 2},
    {"type": "bullet_list", "items": [
        "Hệ thống 3 tầng vận hành end-to-end bằng một lệnh Docker Compose (5 container), tổng quy mô mã nguồn: node biên 5.846 dòng Python, trung tâm 3.582 dòng Java, web 6.591 dòng TypeScript/TSX.",
        "Ba mô hình deep learning đạt chỉ số thực đo: phương tiện mAP50 = 0,944 @ 73 FPS FP16 (YOLO26m gốc COCO chỉ 0,010 — chứng minh fine-tune bắt buộc); đèn top-1 100% trên tập val, 99,0% trên video domain-shift sau fusion; biển số mAP50 = 0,993, ~21 ms/ảnh.",
        "Độ trễ chuyển trạng thái đèn 0,66 giây @6fps (tối ưu từ 1,00 giây); 0,67 giây trên camera 3fps (từ 2,3 giây).",
        "85/85 unit test đạt; biên dịch Java 17 + TypeScript sạch; CI 3 job chạy xanh.",
        "Kiểm định trước sản xuất 2026-09: không còn secret hard-code trong source; các lỗi nghiêm trọng (credential compose, truy vấn quét toàn bảng, thiếu backup) đã sửa; readiness 78/100.",
    ]},

    {"type": "heading", "text": "1.9. Cấu trúc báo cáo", "level": 2},
    {"type": "table", "header": ["Chương", "Nội dung"], "rows": [
        ["Chương 1", "Tổng quan đề tài: lý do, bối cảnh, vấn đề, mục tiêu, phạm vi, phương pháp, kết quả"],
        ["Chương 2", "Cơ sở lý thuyết: kiến trúc ba tầng, client-server, edge computing, REST, xác thực, CSDL, Docker, công nghệ sử dụng"],
        ["Chương 3", "Phân tích và thiết kế: yêu cầu, actor, use case, kiến trúc, sequence, ERD, thiết kế CSDL"],
        ["Chương 4", "Xây dựng hệ thống: môi trường, frontend, backend, CSDL, API, xác thực, đồng bộ, Docker, triển khai"],
        ["Chương 5", "Kiểm thử và đánh giá: chiến lược, unit/integration/API/e2e/security/performance, kết quả"],
        ["Chương 6", "Triển khai và vận hành: kiến trúc sản xuất, Docker, môi trường, reverse proxy, HTTPS, monitoring, backup, security, readiness"],
        ["Chương 7", "Kết luận: kết quả, hạn chế, bài học, hướng phát triển, khả năng mở rộng"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 1.1. Cấu trúc báo cáo"},
    {"type": "page_break"},

    # ================= CHƯƠNG 2 =================
    {"type": "heading", "text": "CHƯƠNG 2. CƠ SỞ LÝ THUYẾT", "level": 1},

    {"type": "heading", "text": "2.1. Kiến trúc Three-Tier", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Kiến trúc ba tầng tách bạch thành: (1) Presentation — giao diện người dùng; (2) Business Logic — xử lý nghiệp vụ; (3) Data Access — lưu trữ và truy vấn dữ liệu. Lợi ích: mỗi tầng thay đổi độc lập, kiểm thử riêng, tái sử dụng logic; ranh giới rõ giúp tránh việc giao diện truy cập thẳng dữ liệu — lỗi phổ biến làm hệ thống khó bảo trì và thiếu an toàn. Trong RLVD, tầng trình bày là dashboard Next.js; tầng nghiệp vụ gồm pipeline node biên + Spring Boot service layer; tầng dữ liệu gồm Spring Data JPA repository, PostgreSQL và MinIO."},

    {"type": "heading", "text": "2.2. Client-Server", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Mô hình client-server phân tách vai trò: client khởi tạo yêu cầu, server đáp ứng. Web dashboard là client của central server; node biên là một dạng client đặc biệt (machine client) vừa cung cấp API điều khiển vừa gửi dữ liệu lên server. Giao tiếp qua HTTP/REST với JSON."},

    {"type": "heading", "text": "2.3. Centralized Server", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Máy chủ trung tâm tập trung lưu trữ, xử lý và cung cấp API cho mọi node biên và client web. Ưu điểm: điểm duy nhất tích hợp nghiệp vụ (thống kê, phân quyền, AI), dữ liệu nhất quán, bảo mật tập trung. Rủi ro: điểm hỏng đơn lẻ — RLVD giảm thiểu bằng healthcheck, restart policy, outbox chịu mất trung tâm ở node biên."},

    {"type": "heading", "text": "2.4. Edge Computing / Edge Node", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Tính toán biên đưa xử lý ra gần nguồn dữ liệu (camera) thay vì đẩy toàn bộ video về trung tâm. Với RLVD: node biên chạy YOLO + tracking + OCR tại chỗ, chỉ đẩy hồ sơ vi phạm (vài KB JSON + 1 ảnh JPEG) — tiết kiệm băng thông đáng kể so với stream video liên tục; giảm độ trễ phát hiện; tận dụng GPU tại chỗ. Node biên vẫn cần trung tâm cho lưu trữ tập trung, hiệu chuẩn từ xa và thống kê."},

    {"type": "heading", "text": "2.5. REST API", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "REST là phong cách kiến trúc API dùng verb HTTP (GET/POST/PUT/PATCH/DELETE) trên tài nguyên định danh bằng URI, stateless. RLVD áp dụng REST cho toàn bộ giao tiếp: node biên POST hồ sơ theo batch; web GET dữ liệu phân trang; hiệu chuẩn POST qua proxy; mã trạng thái chuẩn (200/201/400/401/403/404/409/429/502) như tài liệu API chương 4."},

    {"type": "heading", "text": "2.6. Authentication / Authorization", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Xác thực (authentication) trả lời \"ai đang gọi\", phân quyền (authorization) trả lời \"được làm gì\". RLVD dùng JWT HS256 (access token 12 giờ) cho người dùng web — stateless, dễ scale; refresh token opaque lưu SHA-256 hash trong DB, có rotation và phát hiện reuse (revoke toàn bộ token của user); mật khẩu băm BCrypt. Node biên — không phải người dùng — dùng static ingest token (X-Ingest-Token) so sánh constant-time; control-plane edge dùng EDGE_API_TOKEN riêng. Phân quyền RBAC ba vai trò ADMIN > OPERATOR > OFFICER áp tại cả backend (Spring Security requestMatchers) lẫn frontend."},

    {"type": "heading", "text": "2.7. Database", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Hệ thống dùng PostgreSQL 16 (quan hệ, ACID, index B-tree, JSON support) làm kho chính: violations, edge_nodes, users, refresh_tokens. Ảnh bằng chứng lưu MinIO — object storage tương thích S3 — vì ảnh lớn, không quan hệ, cần URL stream. Node biên dùng SQLite cho outbox cục bộ: file đơn, ghi bền vững, đủ cho quy mô một node. Kỹ thuật chống trùng lặp: unique constraint event_id ở cả hai phía."},

    {"type": "heading", "text": "2.8. Docker / Containerization", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Docker đóng gói ứng dụng cùng dependency thành image chạy nhất quán mọi máy. RLVD dùng Docker Compose dựng full stack 5 dịch vụ trên một mạng bridge nội bộ (rlvd-net): postgres, minio, central-server (multi-stage Maven build), web-dashboard (standalone Next.js, user non-root), edge-pipeline (Python 3.11 + torch cu124, GPU qua nvidia-container-toolkit). Volume bền: postgres_data, minio_data, outbox_data, hls_data. Healthcheck + depends_on đảm bảo thứ tự khởi động."},

    {"type": "heading", "text": "2.9. Các công nghệ thực tế được sử dụng", "level": 2},
    {"type": "table", "header": ["Thành phần", "Công nghệ", "Vai trò trong RLVD"], "rows": [
        ["Node biên", "Python 3.11, FastAPI, OpenCV, ultralytics 8.4, supervision 0.28", "Pipeline YOLO + ByteTrack + API điều khiển :8080"],
        ["Phát hiện phương tiện", "YOLO26m fine-tune 4 lớp (car/bike/van-bus/truck)", "mAP50 0,944, 73 FPS FP16"],
        ["Phân loại đèn", "YOLO26n-cls + fusion HSV + vị trí bóng", "top-1 val 100%, thực tế 99,0%"],
        ["OCR biển số", "fast-plate-ocr (ONNX) + validator TT 24/2023", "mAP50 detect 0,993, ~21 ms/ảnh"],
        ["Trung tâm", "Java 17, Spring Boot 3.3.2, JPA/Hibernate, jjwt", "REST API, RBAC, ingest, MinIO SDK"],
        ["Lưu trữ", "PostgreSQL 16, MinIO", "Hồ sơ + ảnh bằng chứng"],
        ["Web", "Next.js 16, React 19, TypeScript, Tailwind 4, recharts, hls.js, motion", "Dashboard, camera live, hiệu chuẩn"],
        ["Streaming", "FFmpeg (HLS 4s segments)", "Camera live qua HTTP"],
        ["AI", "Gemini 2.5 Flash (REST, java.net.http)", "Text-to-SQL tiếng Việt, SELECT-only"],
        ["Hạ tầng", "Docker Compose, nvidia-container-toolkit", "5 dịch vụ, GPU trong container"],
        ["CI", "GitHub Actions (3 job)", "pytest, maven compile, next build"],
        ["Vận hành", "start.sh, backup_central.sh", "Dựng/chạy stack, backup hàng ngày"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 2.1. Công nghệ thực tế sử dụng trong đồ án"},
    {"type": "page_break"},

    # ================= CHƯƠNG 3 =================
    {"type": "heading", "text": "CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG", "level": 1},

    {"type": "heading", "text": "3.1. Yêu cầu chức năng", "level": 2},
    {"type": "table", "header": ["Mã", "Yêu cầu chức năng"], "rows": [
        ["FR-01", "Phát hiện phương tiện (4 lớp) mỗi frame tại node biên"],
        ["FR-02", "Nhận biết trạng thái đèn tín hiệu (đỏ/vàng/xanh) có debounce chống nhiễu"],
        ["FR-03", "Hiệu chuẩn từ xa: kẻ vạch dừng, chọn hướng giám sát, vùng đèn — áp dụng tức thời"],
        ["FR-04", "Xét vi phạm: xe cắt vạch khi đèn đỏ ổn định, đúng hướng giám sát, chống trùng"],
        ["FR-05", "Tạo hồ sơ vi phạm kèm ảnh bằng chứng (toàn cảnh + biển số) và metadata"],
        ["FR-06", "Đọc biển số Việt Nam theo chuẩn TT 24/2023, sửa lỗi OCR theo vùng"],
        ["FR-07", "Gửi hồ sơ về trung tâm qua durable outbox — không mất khi mất mạng/restart"],
        ["FR-08", "Quản lý node biên: đăng ký, heartbeat, trạng thái online/offline"],
        ["FR-09", "Tra cứu, thống kê vi phạm (phân trang server-side, filter theo trạng thái/node/biển)"],
        ["FR-10", "Duyệt hồ sơ human-in-the-loop: xác nhận / từ chối, kèm vai trò"],
        ["FR-11", "Xem camera trực tuyến HLS + ảnh snapshot"],
        ["FR-12", "Trợ lý AI hỏi dữ liệu bằng tiếng Việt (chỉ SELECT)"],
        ["FR-13", "Đăng nhập JWT + phân quyền 3 vai trò + refresh token rotation"],
        ["FR-14", "Trạng thái đèn realtime hiển thị trên dashboard"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.1. Yêu cầu chức năng"},

    {"type": "heading", "text": "3.2. Yêu cầu phi chức năng", "level": 2},
    {"type": "table", "header": ["Mã", "Yêu cầu phi chức năng", "Chỉ tiêu / cách đo"], "rows": [
        ["NFR-01", "Thời gian thực node biên", "FPS xử lý ≥ FPS nguồn (đo 73 FPS @ RTX 2060 FP16; video 3–10 fps)"],
        ["NFR-02", "Độ trễ chuyển đèn", "≤ 0,8 giây (đo 0,66 s @6fps, 0,67 s @3fps sau tối ưu)"],
        ["NFR-03", "Không mất hồ sơ", "Outbox SQLite bền vững; kiểm thử offline→online e2e"],
        ["NFR-04", "Không trùng hồ sơ", "event_id UNIQUE 2 phía; gửi lại batch → duplicates được skip"],
        ["NFR-05", "Bảo mật", "JWT HS256 ≥32 ký tự, BCrypt, RBAC, ingest token, rate-limit, CORS whitelist"],
        ["NFR-06", "Khả năng mở rộng node", "Đăng ký node theo node_id; batch ingest idempotent; pool 10 connection"],
        ["NFR-07", "Vận hành 1 lệnh", "Docker Compose full stack; healthcheck toàn dịch vụ"],
        ["NFR-08", "Giám sát", "/health + metrics in-process (frames, violations, fps, stale→503)"],
        ["NFR-09", "Khả phục hồi", "Restart policy + backup pg_dump + MinIO mirror hàng ngày"],
        ["NFR-10", "Múi giờ nhất quán", "Asia/Ho_Chi_Minh xuyên suốt 3 tầng (kiểm tra checklist timezone)"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.2. Yêu cầu phi chức năng"},

    {"type": "heading", "text": "3.3. Actor", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Hệ thống có 4 actor thực tế (dựa trên RBAC trong code và luồng vận hành):"},
    {"type": "bullet_list", "items": [
        "Administrator (primary) — toàn quyền: quản trị người dùng, node, duyệt, hiệu chuẩn, xoá hồ sơ.",
        "Operator (primary) — kỹ thuật vận hành: hiệu chuẩn node/camera, xem dữ liệu, không duyệt hồ sơ.",
        "Officer (primary) — cán bộ nghiệp vụ: tra cứu, duyệt/từ chối hồ sơ, hỏi AI (dùng nhiều nhất).",
        "Edge Node (secondary/primary theo luồng) — actor máy: đăng ký, heartbeat, gửi batch vi phạm + ảnh.",
        "Central Server (secondary) — hệ thống: lưu trữ, cung cấp API, proxy hiệu chuẩn xuống edge.",
    ]},

    {"type": "heading", "text": "3.4. Use Case Diagram", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Như thể hiện trong Hình 3.1, Administrator thực hiện toàn bộ use case của Operator và Officer; Operator chuyên về hiệu chuẩn và giám sát node; Officer chuyên duyệt hồ sơ và hỏi AI; Edge Node là actor máy gửi dữ liệu vào hệ thống. Use case \"Duyệt hồ sơ\" include \"Xem chi tiết vi phạm\"; \"Hiệu chuẩn\" include \"Xem camera live\"; \"Gửi hồ sơ\" include \"Đăng ký node\"."},
    {"type": "image", "path": ASSETS + "/vn/use-case.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.1. Sơ đồ use case tổng thể hệ thống"},

    {"type": "heading", "text": "3.5. Đặc tả Use Case quan trọng", "level": 2},
    {"type": "heading", "text": "3.5.1. UC-01: Phát hiện xe vượt đèn đỏ", "level": 3},
    {"type": "table", "header": ["Trường", "Nội dung"], "rows": [
        ["Actor", "Edge Node (primary), Administrator/Officer (xem kết quả)"],
        ["Tiền điều kiện", "Camera/video nguồn hoạt động; mô hình đã nạp; vạch dừng + vùng đèn đã hiệu chuẩn"],
        ["Luồng chính", "1. Đọc frame → 2. Phân loại đèn trong ROI (YOLO26n-cls + fusion) → 3. Debounce trạng thái đèn → 4. YOLO26m phát hiện phương tiện → 5. ByteTrack gán track ID → 6. Nếu đèn ĐỎ ổn định: kiểm tra từng track có cắt vạch dừng theo hướng giám sát → 7. Tạo hồ sơ vi phạm (event_id unique theo run) → 8. OCR biển số nếu phát hiện biển → 9. Ghi outbox SQLite → 10. Vẽ ảnh bằng chứng"],
        ["Luồng thay thế", "Chưa hiệu chuẩn vạch → bỏ qua xét vi phạm (detection vẫn chạy); frame lỗi → log warning, sang frame kế; OCR không hợp lệ → plate = null, hồ sơ vẫn gửi"],
        ["Hậu điều kiện", "Hồ sơ nằm trong outbox, trạng thái pending, đợi sender đẩy batch"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.3. Đặc tả UC-01 Phát hiện xe vượt đèn đỏ"},
    {"type": "heading", "text": "3.5.2. UC-02: Hiệu chuẩn từ xa", "level": 3},
    {"type": "table", "header": ["Trường", "Nội dung"], "rows": [
        ["Actor", "Operator / Administrator (primary), Edge Node (secondary)"],
        ["Tiền điều kiện", "Node đã đăng ký với trung tâm; người dùng đã đăng nhập vai trò đủ quyền"],
        ["Luồng chính", "1. Mở trang /nodes/{id} → 2. Bật chế độ vẽ trên video live → 3. Kéo 2 điểm tạo vạch dừng → 4. (Tuỳ chọn) vẽ mũi tên hướng giám sát / chọn dropdown hướng → 5. POST qua Next proxy → central EdgeProxyService → edge POST /action/stop-line (kèm X-Edge-Token) → 6. Pipeline cập nhật active tripwire NGAY frame kế tiếp — không restart → 7. Verify GET /api/calibration"],
        ["Luồng thay thế", "Vẽ vùng đèn thay vạch (chế độ box); chỉ đổi hướng không vẽ lại vạch"],
        ["Hậu điều kiện", "Violations từ DISABLED → ENABLED nếu trước đó chưa có vạch"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.4. Đặc tả UC-02 Hiệu chuẩn từ xa"},
    {"type": "heading", "text": "3.5.3. UC-03: Duyệt hồ sơ vi phạm", "level": 3},
    {"type": "table", "header": ["Trường", "Nội dung"], "rows": [
        ["Actor", "Officer / Administrator (primary)"],
        ["Tiền điều kiện", "Hồ sơ có trạng thái pending; đã đăng nhập vai trò OFFICER trở lên"],
        ["Luồng chính", "1. Vào trang Review (tải pending theo batch 50, prefetch) → 2. Xem ảnh bằng chứng + biển số + metadata → 3. Bấm Xác nhận / Từ chối → 4. PATCH /api/violations/{id}/status (RBAC kiểm tra vai trò) → 5. Chuyển hồ sơ kế (animation AnimatePresence)"],
        ["Luồng thay thế", "ADMIN có thể xoá hồ sơ; filter theo node/trạng thái"],
        ["Hậu điều kiện", "Hồ sơ chuyển approved/rejected; badge số pending giảm"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.5. Đặc tả UC-03 Duyệt hồ sơ vi phạm"},

    {"type": "heading", "text": "3.6. System Architecture", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Như thể hiện trong Hình 3.2, luồng chính: Camera/Video → Edge Node (YOLO + OCR + outbox) → Central Server (Spring Boot) → PostgreSQL + MinIO; Web Dashboard vừa gọi central qua rewrite /api (kèm JWT) vừa gọi trực tiếp edge qua /edge-api (camera HLS, hiệu chuẩn). Central cũng gọi NGƯỢC xuống edge (EdgeProxyService) cho hiệu chuẩn — đường duy nhất, có token. Điểm neo độ tin cậy: outbox SQLite nằm giữa detection và network."},
    {"type": "image", "path": ASSETS + "/vn/architecture.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.2. Kiến trúc tổng thể hệ thống"},

    {"type": "heading", "text": "3.7. Three-Tier Architecture", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Như thể hiện trong Hình 3.3, ba tầng được ánh xạ cụ thể: Presentation = web/app + web/components (Next.js) — không import thư viện DB/AI nào; Business Logic = central_server/service (business rules, RBAC, ingest, proxy) + edge_node/core (pipeline nghiệp vụ thị giác); Data Access = Spring Data JPA repository + PostgreSQL + MinIO. Kiểm định thực tế (9/2026): web/package.json không chứa pg hay @google/genai; service không import controller; không có SQL trong controller — 3 tiêu chí tách tầng đều đạt."},
    {"type": "image", "path": DIAG + "/three-tier.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.3. Kiến trúc ba tầng"},

    {"type": "heading", "text": "3.8. Central Server Architecture", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Central Server theo mẫu layer chuẩn Spring: Controller (REST, validation đầu vào) → Service (nghiệp vụ) → Repository (JPA) → Entity ↔ DTO chuyển đổi. Hai luồng bảo mật đi song song trước controller: JwtAuthFilter (người dùng) và IngestTokenFilter (node biên), danh sách đường dẫn ingest tập trung một chỗ duy nhất (IngestPaths) để hai filter không thể lệch nhau — lỗi từng xảy ra trước khi tách (mất 263 ảnh bằng chứng). ViolationStatsService tách khỏi ViolationService để thống kê dùng truy vấn COUNT/GROUP BY thay vì tải bảng. AiQueryService gọi Gemini REST sinh SQL, lọc an toàn, chạy JdbcTemplate. EdgeProxyService là đường gọi ngược central→edge duy nhất, cố định HTTP/1.1 (tránh bug h2c làm rơi body POST), timeout 5–60 giây."},
    {"type": "image", "path": DIAG + "/api-surface.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.4. Bề mặt API trung tâm — phân nhóm bảo vệ (ingest token, JWT + RBAC, public health)"},

    {"type": "heading", "text": "3.9. Edge Node Architecture", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Như thể hiện trong Hình 3.5, node biên có vòng đời: BOOT → INITIALIZE (nạp weights) → REGISTER (đăng ký trung tâm, tiếp tục chạy kể cả offline) → RUNNING (vòng lặp mỗi frame: đọc frame → phân loại đèn → debounce → detect → track → OCR → xét vi phạm → ghi outbox) → khi mất mạng chuyển OFFLINE (buffer local + retry backoff 2^n tối đa 60 giây) → mạng lại → flush toàn bộ pending. Shutdown graceful: đợi sender rút outbox trong flush_interval + 5 giây trước khi thoát. Nếu chưa có vạch dừng, violation check bị vô hiệu hoá nhưng detection/tracking vẫn chạy — hiển thị rõ trong log."},
    {"type": "image", "path": DIAG + "/edge-lifecycle.png", "width_mm": 130},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.5. Vòng đời node biên — trạng thái và offline buffer"},

    {"type": "heading", "text": "3.10. Deployment Architecture", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Như thể hiện trong Hình 3.6, sản phẩm chạy trên một máy chủ Docker: 5 container trong mạng bridge rlvd-net. Reverse proxy Nginx/Caddy (khuyến nghị khi public) terminates TLS rồi proxy về web :3000. Cổng host: web 3000, central 8002 (tránh đụng portainer 8000), edge 8082 (qbittorrent giữ 8080), MinIO 9010/9011, PostgreSQL 5432. GPU NVIDIA cấp cho edge-pipeline qua nvidia-container-toolkit; volume models mount read-only; outbox + hls nằm volume riêng."},
    {"type": "image", "path": DIAG + "/deployment.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.6. Sơ đồ triển khai (Deployment Diagram)"},

    {"type": "heading", "text": "3.11. Component Diagram", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Hình 3.7 thể hiện các thành phần chính và quan hệ: node biên gồm pipeline, outbox, sender, camera stream, control API; trung tâm gồm auth, violation, node registry, media, AI, proxy; web gồm pages, API client, auth session, video player, AI sidebar. Giao tiếp đều qua HTTP/REST trừ luồng HLS là HTTP m3u8/ts."},
    {"type": "image", "path": ASSETS + "/vn/component.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.7. Sơ đồ thành phần (Component Diagram)"},

    {"type": "heading", "text": "3.12. Sequence Diagram", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Ba luồng quan trọng nhất: (1) Gửi dữ liệu Edge → Central (Hình 3.8) — điểm nhấn là ghi outbox TRƯỚC khi gọi mạng, batch idempotent theo event_id, upload ảnh sau, backoff khi lỗi; (2) Xác thực người dùng (Hình 3.9) — JWT 12 giờ, refresh token hash SHA-256 + rotation + phát hiện reuse thu hồi toàn bộ; (3) Hiệu chuẩn từ xa (Hình 3.10) — web → central → edge, hiệu lực tức thì không restart. Như thể hiện trong các hình 3.8–3.10."},
    {"type": "image", "path": DIAG + "/sequence-edge-data.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.8. Sequence gửi dữ liệu Edge → Central qua durable outbox"},
    {"type": "image", "path": DIAG + "/sequence-auth.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.9. Sequence xác thực: login, JWT, refresh rotation, reuse detection"},

    {"type": "heading", "text": "3.13. Data Flow", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Luồng dữ liệu chính: video frame → detections → tracks → (đèn đỏ + cắt vạch) → violation event + ảnh → outbox SQLite → HTTP batch JSON → PostgreSQL + ảnh JPEG → MinIO (object key) → web hiển thị qua proxy blob URL. Luồng hiệu chuẩn: thao tác chuột web → tọa độ pixel video native → JSON → central → edge memory (active tripwire) → ảnh hưởng frame kế tiếp. Luồng AI: câu hỏi tiếng Việt → Gemini (kèm schema) → SQL SELECT → PostgreSQL → bảng/biểu đồ web."},
    {"type": "image", "path": ASSETS + "/vn/data-flow.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.11. Sơ đồ luồng dữ liệu"},

    {"type": "heading", "text": "3.14. Database Design", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Cơ sở dữ liệu rlvd_central gồm 4 bảng. Bảng violations (Bảng 3.6) là bảng chính, có index trên event_id (unique), node_id, status, created_at phục vụ truy vấn lọc + phân trang + thống kê theo giờ. Bảng edge_nodes (Bảng 3.7) là registry node. Bảng users + refresh_tokens (Bảng 3.8) phục vụ xác thực — refresh token chỉ lưu hash nên DB leak không replay được. Không dùng khoá ngoại cứng giữa violations.node_id và edge_nodes.node_id (quan hệ logic, node_id là chuỗi) — chấp nhận được vì ingest idempotent và node_id do trung tâm kiểm soát qua đăng ký."},
    {"type": "table", "header": ["Cột", "Kiểu", "Ràng buộc / ý nghĩa"], "rows": [
        ["id", "BIGINT", "PK auto-increment"],
        ["event_id", "VARCHAR", "UNIQUE (idx_event_id) — idempotent ingest, định danh rlv-{run}-{frame}-{track}"],
        ["node_id", "VARCHAR", "idx_node_id — node gửi hồ sơ"],
        ["track_id, frame_index, timestamp_ms", "INT/DOUBLE", "vị trí phát hiện trong video"],
        ["crossing/previous point, bbox xyxy", "DOUBLE", "toạ độ hình học lúc cắt vạch"],
        ["light_state, light_confidence", "VARCHAR/DOUBLE", "trạng thái đèn + độ tin cậy lúc vi phạm"],
        ["previous_side, current_side", "INT", "chuyển phía qua vạch (bằng chứng hướng)"],
        ["plate_text, plate_confidence", "VARCHAR/DOUBLE", "OCR đã validate theo TT 24/2023"],
        ["status", "VARCHAR", "pending/approved/rejected (idx_status) — human-in-the-loop"],
        ["media_url", "VARCHAR(1024)", "object key MinIO violations/{eventId}/{uuid}.jpg"],
        ["metadata", "TEXT", "JSON mở rộng (hướng giám sát, nguồn...)"],
        ["created_at, updated_at", "DATETIME", "idx_created_at — audit + thống kê giờ, múi giờ VN"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.6. Thiết kế bảng violations"},
    {"type": "table", "header": ["Cột", "Kiểu", "Ràng buộc / ý nghĩa"], "rows": [
        ["id", "UUID", "PK"],
        ["node_id", "VARCHAR", "UNIQUE — định danh node, upsert theo đây"],
        ["name", "VARCHAR", "tên hiển thị (mặc định = node_id)"],
        ["ip_address", "VARCHAR", "địa chỉ gọi ngược từ central (đã strip scheme/port)"],
        ["status", "VARCHAR", "online/offline/degraded/maintenance (idx)"],
        ["last_ping", "DATETIME", "idx — online nếu ping < 2 phút"],
        ["settings_json", "TEXT", "toàn bộ cấu hình edge đăng ký (kể cả api_port, api_token cho proxy)"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.7. Thiết kế bảng edge_nodes"},
    {"type": "table", "header": ["Bảng", "Cột chính", "Ràng buộc"], "rows": [
        ["users", "id PK, username UNIQUE(64), password_hash BCrypt(128), full_name, role (ADMIN/OPERATOR/OFFICER), enabled, created/updated_at", "username unique không phân biệt hoa thường khi tra cứu"],
        ["refresh_tokens", "id PK, user_id (idx), token_hash UNIQUE(64) SHA-256, expires_at, revoked_at, created_at, user_agent(255)", "chỉ lưu hash; revoke_all_by_user khi phát hiện reuse"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "CaptionV", "text": "Bảng 3.8. Thiết kế bảng users và refresh_tokens"},

    {"type": "heading", "text": "3.15. ERD", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Như thể hiện trong Hình 3.12: users 1—n refresh_tokens (một user nhiều phiên); edge_nodes 1—n violations theo node_id (quan hệ logic); violations là bảng trung tâm với 4 index phục vụ các truy vấn chính. Việc duy nhất nhất quán (event_id) bảo đảm node gửi lại batch sau lỗi mạng không tạo bản trùng."},
    {"type": "image", "path": DIAG + "/erd.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.12. Sơ đồ thực thể dữ liệu (ERD)"},
    {"type": "page_break"},
]
