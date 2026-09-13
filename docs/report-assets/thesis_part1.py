# -*- coding: utf-8 -*-
"""Đồ án RLVD — nội dung phần 1: bìa → Chương 3 (Phân tích hệ thống).

Nguyên tắc số liệu: mọi con số trong file này đều lấy từ nguồn kiểm chứng được
trong repository tại thời điểm hoàn thiện đồ án (12/9/2026):
  * LOC đếm bằng cloc/wc trên edge_node/, central_server/, web/, tests/
  * kết quả pytest chạy thật: 124 passed in 1.09s
  * scripts/benchmark_results.csv (đo trên RTX 2060, FP16)
  * /tmp/bench_ocr2.json (đo OCR + plate detector, 12/9/2026)
  * nhật ký chạy end-to-end /tmp/baseline_run.log và /tmp/ev_2022.json
Những hạng mục chưa đo được ghi rõ là chưa đo, không suy diễn.
"""

ROOT = "/home/uvhnael/projects/Red-Light-Violation-Detection"
T = ROOT + "/docs/report-assets/thesis"   # diagram vẽ mới cho đồ án
V = ROOT + "/docs/report-assets/vn"       # diagram UML tiếng Việt
S = ROOT + "/docs/report-assets/shots"    # screenshot dashboard

P = lambda t, s="Body": {"type": "paragraph", "style": s, "text": t}
H1 = lambda t: {"type": "heading", "text": t, "level": 1}
H2 = lambda t: {"type": "heading", "text": t, "level": 2}
H3 = lambda t: {"type": "heading", "text": t, "level": 3}
UL = lambda items: {"type": "bullet_list", "items": items}
PL = lambda items, **kw: {"type": "plain_list", "items": items, **kw}
TB = lambda header, rows, caption: {"type": "table", "header": header,
                                    "rows": rows, "caption": caption}
IMG = lambda path, w, caption: {"type": "image", "path": path,
                                "width_mm": w, "caption": caption}
PB = {"type": "page_break"}

BLOCKS_1 = [
    # ============================================================== BÌA
    {"type": "paragraph", "style": "TitleBig", "align": "center",
     "text": "ĐỒ ÁN TỐT NGHIỆP"},
    {"type": "paragraph", "style": "TitleSub", "align": "center",
     "text": "XÂY DỰNG HỆ THỐNG PHÁT HIỆN VI PHẠM VƯỢT ĐÈN ĐỎ"},
    {"type": "paragraph", "style": "TitleSub2", "align": "center",
     "text": "Red-Light Violation Detection (RLVD)"},
    {"type": "paragraph", "style": "TitleSub2", "align": "center",
     "text": "Kiến trúc ba tầng: Node biên — Máy chủ trung tâm — Bảng điều khiển web"},
    {"type": "paragraph", "style": "TitleMeta", "align": "center",
     "text": "Ngành: Kỹ thuật phần mềm"},
    {"type": "paragraph", "style": "TitleMeta", "align": "center",
     "text": "Tháng 9 năm 2026"},
    PB,

    # ======================================================== LỜI CẢM ƠN
    H1("LỜI CẢM ƠN"),
    P("Nhóm thực hiện xin gửi lời cảm ơn chân thành đến giảng viên hướng dẫn đã định "
      "hướng đề tài, góp ý về kiến trúc hệ thống và kiên nhẫn theo sát từng giai đoạn "
      "của đồ án. Những nhận xét phản biện của thầy/cô ở các buổi nghiệm thu trung gian "
      "đã giúp nhóm điều chỉnh nhiều quyết định thiết kế quan trọng, trong đó có việc "
      "chuyển từ mô hình hàng đợi trung gian sang mẫu outbox bền vững tại node biên."),
    P("Nhóm cũng xin cảm ơn cộng đồng mã nguồn mở đã cung cấp các công cụ nền tảng mà "
      "đồ án kế thừa: Ultralytics (YOLO), supervision (ByteTrack), fast-plate-ocr, "
      "Spring Boot, Next.js, PostgreSQL, MinIO và FFmpeg. Việc được xây dựng trên những "
      "thư viện trưởng thành cho phép nhóm tập trung vào phần đóng góp riêng: logic xét "
      "vi phạm, cơ chế giao hàng không mất dữ liệu và quy trình duyệt hồ sơ có sự tham "
      "gia của con người."),
    P("Cuối cùng, nhóm cảm ơn các bạn đã hỗ trợ thu thập video tại một số nút giao thông "
      "để phục vụ huấn luyện và kiểm thử, cùng gia đình đã tạo điều kiện về thời gian và "
      "thiết bị trong suốt quá trình thực hiện."),
    PB,

    # ============================================================ TÓM TẮT
    H1("TÓM TẮT"),
    P("Đồ án nghiên cứu và xây dựng hệ thống phần mềm phát hiện hành vi vượt đèn đỏ của "
      "phương tiện giao thông từ luồng video giám sát tại nút giao, theo kiến trúc phân "
      "tán ba tầng: node biên thực hiện toàn bộ phân tích thị giác máy tính ngay tại vị "
      "trí camera; máy chủ trung tâm lưu trữ hồ sơ vi phạm, quản lý người dùng và cung "
      "cấp giao diện lập trình ứng dụng; bảng điều khiển web phục vụ giám sát, hiệu "
      "chuẩn từ xa và duyệt hồ sơ."),
    P("Tại node biên, đồ án xây dựng một pipeline thị giác hoàn chỉnh: mô hình YOLO26m "
      "được huấn luyện lại trên dữ liệu phương tiện Việt Nam đạt mAP50 = 0,944 với tốc "
      "độ 73,4 khung hình/giây ở độ chính xác nửa (FP16) trên GPU RTX 2060; ByteTrack "
      "được tinh chỉnh tham số để giữ liên tục vết của xe máy ở tốc độ khung hình thấp; "
      "bộ phân loại đèn tín hiệu kết hợp mô hình học sâu với bằng chứng màu HSV và vị "
      "trí vật lý của bóng đèn; bộ đọc biển số tự động kèm cổng kiểm tra cấu trúc theo "
      "Thông tư 24/2023/TT-BGTVT. Việc xét vi phạm dựa trên vạch dừng ảo có hướng, "
      "trạng thái đèn đã ổn định nhờ cơ chế debounce có trễ (hysteresis) và điểm neo "
      "bottom-center của phương tiện."),
    P("Đóng góp đáng chú ý nhất về mặt kiến trúc là cơ chế giao hàng bền vững: mỗi hồ sơ "
      "vi phạm cùng ảnh bằng chứng được ghi vào kho trung gian SQLite ngay tại thời điểm "
      "phát hiện, trước mọi thao tác mạng, và được một luồng nền đẩy lên trung tâm theo "
      "lô có tính idempotent. Nhờ đó hệ thống chỉ có thể trễ chứ không thể mất dữ liệu "
      "khi mất kết nối hoặc khởi động lại. Máy chủ trung tâm Spring Boot bảo vệ bằng "
      "JWT phân quyền ba vai trò, kèm token riêng cho node biên và cơ chế xoay refresh "
      "token có phát hiện việc dùng lại."),
    P("Kết quả kiểm thử: 124 kiểm thử đơn vị đạt toàn bộ trong 1,09 giây; biên dịch Java "
      "và TypeScript sạch; kiểm thử end-to-end trên video có nhãn cho kết quả đúng với "
      "ground-truth (một vi phạm thật được phát hiện trên video aziz1.MP4, không có vi "
      "phạm giả nào trên video đối chứng 384 khung hình). Toàn bộ hệ thống vận hành được "
      "bằng một lệnh Docker Compose với năm dịch vụ. Đồ án cũng ghi nhận trung thực các "
      "hạn chế: chưa kiểm thử tải HTTP, chưa có bộ kiểm thử tích hợp tự động cho tầng "
      "Java, và chưa có tập video biển số Việt Nam đủ lớn để đo độ chính xác OCR độc lập."),
    P("Từ khoá: thị giác máy tính, YOLO, ByteTrack, nhận dạng biển số, tính toán biên, "
      "kiến trúc ba tầng, Spring Boot, Next.js, Docker, PostgreSQL, MinIO, JWT, outbox.",
      "BodyJ"),
    PB,

    # =========================================================== ABSTRACT
    H1("ABSTRACT"),
    P("This thesis designs and implements a software system that detects red-light running "
      "violations from surveillance video at road intersections, organised as a distributed "
      "three-tier architecture: an edge node performing all computer-vision analysis at the "
      "camera site, a central server storing violation records and exposing APIs, and a web "
      "dashboard for monitoring, remote calibration and human-in-the-loop review.", "BodyJ"),
    P("The edge pipeline combines a YOLO26m detector fine-tuned on Vietnamese traffic data "
      "(mAP50 = 0.944 at 73.4 FPS in FP16 on an RTX 2060), a parameter-tuned ByteTrack "
      "tracker that keeps motorbike tracks continuous at low frame rates, a traffic-light "
      "classifier fusing a deep model with HSV colour evidence and physical lamp position, "
      "and an automatic plate reader gated by a structural validator conforming to Circular "
      "24/2023/TT-BGTVT. Violation decisions rely on a directional virtual stop line, a "
      "hysteresis-debounced light state and the bottom-center anchor of each vehicle.", "BodyJ"),
    P("The main architectural contribution is durable delivery: every violation record and "
      "its evidence image are written to a local SQLite outbox at detection time, before any "
      "network operation, and are then pushed to the centre in idempotent batches by a "
      "background thread. The system can therefore lag but cannot lose data across outages or "
      "restarts. The Spring Boot central server is protected by role-based JWT "
      "authentication, a separate ingest token for edge nodes, and refresh-token rotation "
      "with reuse detection.", "BodyJ"),
    P("Testing results: all 124 unit tests pass in 1.09 s; Java and TypeScript builds are "
      "clean; end-to-end runs on labelled video match ground truth (one true violation "
      "detected in aziz1.MP4, zero false positives across a 384-frame control video). The "
      "complete stack runs from a single Docker Compose command with five services. The "
      "thesis also reports its limitations honestly: no HTTP load testing, no automated "
      "integration test suite for the Java tier, and no sufficiently large Vietnamese "
      "plate-video corpus for an independent OCR accuracy measurement.", "BodyJ"),
    P("Keywords: computer vision, YOLO, ByteTrack, license plate recognition, edge "
      "computing, three-tier architecture, Spring Boot, Next.js, Docker, PostgreSQL, MinIO, "
      "JWT, outbox pattern.", "BodyJ"),
    PB,

    # =========================================================== MỤC LỤC
    H1("MỤC LỤC"),
    {"type": "toc", "max_level": 3},
    PB,

    # =================================================== DANH SÁCH HÌNH
    H1("DANH SÁCH HÌNH"),
    PL([
        'Hình 2.1 — Quy trình xử lý vi phạm vượt đèn đỏ hiện nay (as-is)',
        'Hình 2.2 — Quy trình đề xuất khi có hệ thống RLVD (to-be)',
        'Hình 3.1 — Quan hệ actor, vai trò và quyền hạn trong hệ thống',
        'Hình 3.2 — Sơ đồ use case tổng thể',
        'Hình 3.3 — Sơ đồ use case của Administrator',
        'Hình 3.4 — Sơ đồ use case của Officer',
        'Hình 3.5 — Sơ đồ use case của Operator',
        'Hình 3.6 — Sơ đồ hoạt động xét vi phạm tại node biên',
        'Hình 3.7 — Sơ đồ hoạt động duyệt hồ sơ vi phạm',
        'Hình 3.8 — Sơ đồ hoạt động hiệu chuẩn từ xa',
        'Hình 3.9 — Máy trạng thái tín hiệu đèn có debounce',
        'Hình 3.10 — Máy trạng thái vòng đời hồ sơ vi phạm',
        'Hình 3.11 — Sơ đồ sequence phân phối hồ sơ bền vững',
        'Hình 3.12 — Sơ đồ sequence xác thực và làm mới token',
        'Hình 3.13 — Sơ đồ sequence hiệu chuẩn vạch dừng từ xa',
        'Hình 3.14 — Sơ đồ sequence trợ lý hỏi dữ liệu bằng tiếng Việt',
        'Hình 4.1 — Kiến trúc hệ thống ba tầng',
        'Hình 4.2 — Sơ đồ thành phần và giao diện giữa các tầng',
        'Hình 4.3 — Sơ đồ triển khai bằng Docker Compose với năm dịch vụ',
        'Hình 4.4 — Sơ đồ lớp của hệ thống',
        'Hình 4.5 — Mô hình thực thể quan hệ của cơ sở dữ liệu rlvd_central',
        'Hình 4.6 — Luồng dữ liệu hồ sơ vi phạm qua ba tầng',
        'Hình 4.7 — Các lớp bảo vệ của hệ thống theo nhóm yêu cầu',
        'Hình 4.8 — Kiến trúc thông tin của bảng điều khiển web',
        'Hình 4.9 — Wireflow nghiệp vụ duyệt hồ sơ của cán bộ xử lý vi phạm',
        'Hình 5.1 — Trang đăng nhập',
        'Hình 5.2 — Trang bảng điều khiển tổng quan sau khi đăng nhập',
        'Hình 5.3 — Trang tra cứu hồ sơ vi phạm có phân trang và lọc',
        'Hình 5.4 — Trang chi tiết hồ sơ với ảnh bằng chứng và nút quyết định xử lý',
        'Hình 5.5 — Trang duyệt hồ sơ hàng loạt',
        'Hình 5.6 — Trang xem camera trực tiếp với các bố cục lưới 1, 4, 9 và 16',
        'Hình 5.7 — Trang danh sách node biên',
        'Hình 5.8 — Trang chi tiết node với luồng video trực tiếp',
        'Hình 5.9 — Công cụ hiệu chuẩn vạch dừng trên video trực tiếp',
        'Hình 5.10 — Trợ lý hỏi dữ liệu bằng tiếng Việt',
        'Hình 5.11 — Trang cài đặt giao diện',
    ]),
    PB,

    # =================================================== DANH SÁCH BẢNG
    H1("DANH SÁCH BẢNG"),
    PL([
        'Bảng 0.1 — Danh mục từ viết tắt sử dụng trong đồ án',
        'Bảng 1.1 — Các giai đoạn thực hiện đồ án',
        'Bảng 2.1 — Hạn chế của quy trình xử lý thủ công',
        'Bảng 2.2 — So sánh hệ thống RLVD với các giải pháp liên quan',
        'Bảng 2.3 — Yêu cầu chức năng',
        'Bảng 2.4 — Yêu cầu phi chức năng',
        'Bảng 3.1 — Danh sách actor và quyền hạn',
        'Bảng 3.2 — Đặc tả UC-01 Phát hiện xe vượt đèn đỏ',
        'Bảng 3.3 — Đặc tả UC-02 Gửi hồ sơ vi phạm về trung tâm',
        'Bảng 3.4 — Đặc tả UC-03 Hiệu chuẩn từ xa',
        'Bảng 3.5 — Đặc tả UC-04 Duyệt hồ sơ vi phạm',
        'Bảng 3.6 — Đặc tả UC-05 Hỏi dữ liệu bằng tiếng Việt',
        'Bảng 4.1 — Thiết kế bảng violations',
        'Bảng 4.2 — Thiết kế bảng edge_nodes',
        'Bảng 4.3 — Thiết kế bảng users và refresh_tokens',
        'Bảng 4.4 — Lược đồ kho outbox tại node biên (SQLite)',
        'Bảng 4.5 — Quy ước thiết kế API',
        'Bảng 4.6 — Danh sách nhóm API của máy chủ trung tâm',
        'Bảng 4.7 — Danh sách API điều khiển tại node biên',
        'Bảng 4.8 — Mã trạng thái HTTP và ý nghĩa trong hệ thống',
        'Bảng 4.9 — Nguyên tắc thiết kế giao diện và cách thể hiện',
        'Bảng 5.1 — Quy mô mã nguồn theo tầng',
        'Bảng 5.2 — Các trang của bảng điều khiển web',
        'Bảng 5.3 — Các mô-đun chính của tầng web',
        'Bảng 5.4 — Các module chính của node biên',
        'Bảng 5.5 — Các lớp chính của máy chủ trung tâm',
        'Bảng 5.6 — Ánh xạ cổng dịch vụ khi triển khai bằng Docker Compose',
        'Bảng 5.7 — Biến môi trường cấu hình hệ thống',
        'Bảng 6.1 — Danh sách ca kiểm thử',
        'Bảng 6.2 — Kết quả kiểm thử đơn vị theo tệp',
        'Bảng 6.3 — Kết quả đánh giá mô hình phát hiện phương tiện',
        'Bảng 6.4 — Kết quả đánh giá mô hình phân loại đèn và mô hình phát hiện biển số',
        'Bảng 6.5 — Kết quả đo thời gian xử lý khâu biển số',
        'Bảng 6.6 — Kết quả đo khâu nhận dạng biển số',
        'Bảng 6.7 — Kết quả kiểm thử hồi quy trên video có nhãn',
        'Bảng 6.8 — Tổng hợp kết quả kiểm thử',
        'Bảng 7.1 — Đối chiếu mục tiêu và kết quả đạt được',
        'Bảng 7.2 — Hạn chế của đồ án theo mức độ ảnh hưởng',
    ]),
    PB,

    # ================================================ DANH SÁCH TỪ VIẾT TẮT
    H1("DANH SÁCH TỪ VIẾT TẮT"),
    TB(["Từ viết tắt", "Nghĩa đầy đủ"], [
        ["RLVD", "Red-Light Violation Detection — phát hiện vi phạm vượt đèn đỏ"],
        ["ANPR", "Automatic Number Plate Recognition — nhận dạng biển số tự động"],
        ["YOLO", "You Only Look Once — họ mô hình phát hiện đối tượng thời gian thực"],
        ["mAP", "mean Average Precision — độ chính xác trung bình, chỉ số đánh giá mô hình phát hiện"],
        ["mAP50", "mAP tính tại ngưỡng IoU = 0,50"],
        ["IoU", "Intersection over Union — tỉ lệ giao trên hợp của hai khung giới hạn"],
        ["MOT", "Multi-Object Tracking — theo dõi đa đối tượng"],
        ["OCR", "Optical Character Recognition — nhận dạng ký tự quang học"],
        ["FP16", "Half Precision Floating Point — số thực dấu phẩy động 16 bit"],
        ["FPS", "Frames Per Second — số khung hình xử lý mỗi giây"],
        ["ROI", "Region of Interest — vùng quan tâm"],
        ["HSV", "Hue–Saturation–Value — không gian màu sắc độ, độ bão hoà, độ sáng"],
        ["EMA", "Exponential Moving Average — trung bình trượt theo hàm mũ"],
        ["API", "Application Programming Interface — giao diện lập trình ứng dụng"],
        ["REST", "Representational State Transfer — phong cách thiết kế API"],
        ["DTO", "Data Transfer Object — đối tượng truyền dữ liệu giữa các tầng"],
        ["ERD", "Entity Relationship Diagram — sơ đồ thực thể quan hệ"],
        ["CSDL", "Cơ sở dữ liệu"],
        ["JPA", "Jakarta Persistence API — chuẩn ánh xạ đối tượng – CSDL của Java"],
        ["JWT", "JSON Web Token — token xác thực có chữ ký số"],
        ["RBAC", "Role-Based Access Control — phân quyền theo vai trò"],
        ["BCrypt", "Hàm băm mật khẩu thích ứng dựa trên Blowfish"],
        ["HLS", "HTTP Live Streaming — giao thức phát video phân đoạn qua HTTP"],
        ["RTSP", "Real-Time Streaming Protocol — giao thức truyền video thời gian thực"],
        ["FR", "Functional Requirement — yêu cầu chức năng"],
        ["NFR", "Non-Functional Requirement — yêu cầu phi chức năng"],
        ["UC", "Use Case — ca sử dụng"],
        ["CI", "Continuous Integration — tích hợp liên tục"],
        ["Outbox", "Kho trung gian bền vững đặt phía node sản sinh dữ liệu (mẫu outbox)"],
        ["TT 24/2023", "Thông tư 24/2023/TT-BGTVT về cấp, thu hồi đăng ký và biển số xe cơ giới"],
    ], "Bảng 0.1. Danh mục từ viết tắt sử dụng trong đồ án"),
    PB,

    # ======================================================= CHƯƠNG 1
    H1("CHƯƠNG 1. GIỚI THIỆU"),

    H2("1.1. Lý do chọn đề tài"),
    P("Không chấp hành tín hiệu đèn giao thông là một trong những hành vi vi phạm phổ "
      "biến và nguy hiểm nhất tại Việt Nam, bởi xung đột xảy ra đúng ở vị trí các luồng "
      "giao cắt có tốc độ khác nhau và hầu như không có cơ hội tránh né. Hậu quả thường "
      "là va chạm ngang hông với mức độ thương tích cao. Trong khi đó, việc phát hiện "
      "hành vi này hiện vẫn dựa chủ yếu vào hai hình thức: cảnh sát giao thông trực "
      "tiếp tại chốt, hoặc xem lại video ghi hình một cách thủ công."),
    P("Cả hai hình thức đều có giới hạn cố hữu. Giám sát trực tiếp chỉ phủ được số lượng "
      "nút giao bằng số tổ công tác có thể bố trí, phụ thuộc thời tiết, ca kíp và sự "
      "hiện diện của lực lượng chức năng — nghĩa là hành vi vi phạm dễ tái diễn ngay khi "
      "vắng bóng người kiểm soát. Xem lại video thủ công tuy bao phủ rộng hơn nhưng tốn "
      "rất nhiều nhân lực: một cán bộ phải quan sát hàng giờ footage để tìm vài chục "
      "giây có vi phạm, và kết quả phụ thuộc vào sự tập trung của người xem. Ngoài ra, "
      "cả hai hình thức đều tạo ra bằng chứng ở dạng mô tả, khó chuẩn hoá và khó đối "
      "chiếu tự động với cơ sở dữ liệu đăng ký xe."),
    P("Về mặt kỹ thuật, hai điều kiện để tự động hoá bài toán này đã chín muồi. Thứ "
      "nhất, các mô hình phát hiện đối tượng họ YOLO đã đạt đồng thời độ chính xác và "
      "tốc độ đủ để xử lý video thời gian thực trên phần cứng phổ thông, không còn yêu "
      "cầu hạ tầng chuyên dụng. Thứ hai, mô hình tính toán biên cho phép đặt năng lực xử "
      "lý ngay tại vị trí camera: chỉ hồ sơ vi phạm vài kilobyte kèm một ảnh bằng chứng "
      "cần truyền về trung tâm, thay vì toàn bộ luồng video liên tục vốn đòi hỏi băng "
      "thông lớn và không mở rộng được theo số nút giao."),
    P("Tuy nhiên, một hệ thống hoàn chỉnh không dừng lại ở việc nhận diện được phương "
      "tiện và màu đèn. Những khó khăn thực sự nằm ở phần tích hợp: đọc biển số đúng "
      "chuẩn Việt Nam và loại được kết quả OCR sai; xác định đúng thời điểm vi phạm để "
      "không bỏ sót nhưng cũng không tạo hồ sơ giả; truyền hồ sơ về trung tâm một cách "
      "tin cậy trong điều kiện mạng không ổn định; xây dựng quy trình duyệt có sự tham "
      "gia của con người để bảo đảm giá trị pháp lý; và cung cấp công cụ hiệu chuẩn từ "
      "xa để một kỹ thuật viên có thể cấu hình vạch dừng cho nhiều nút giao mà không cần "
      "đến tận nơi. Chính yêu cầu tích hợp đa tầng này là lý do nhóm lựa chọn đề tài: "
      "bài toán đủ hẹp để hoàn thiện trọn vẹn trong khuôn khổ đồ án, nhưng đủ phức tạp "
      "để vận dụng kiến thức của nhiều học phần — thị giác máy tính, kiến trúc phần mềm, "
      "cơ sở dữ liệu, an toàn thông tin và phát triển giao diện người dùng."),

    H2("1.2. Mục tiêu"),
    P("Mục tiêu tổng quát của đồ án là xây dựng một hệ thống phần mềm phân tán có khả "
      "năng tự động phát hiện phương tiện vượt đèn đỏ từ luồng video giám sát, lập hồ sơ "
      "vi phạm kèm ảnh bằng chứng và biển số xe, sau đó phục vụ tra cứu, thống kê và "
      "duyệt xử lý thông qua giao diện web."),
    P("Từ mục tiêu tổng quát, đồ án xác định các mục tiêu cụ thể sau:"),
    UL([
        "Xây dựng pipeline thị giác máy tính chạy thời gian thực tại node biên, gồm bốn "
        "khâu: phát hiện phương tiện, theo dõi đa đối tượng, nhận biết trạng thái đèn "
        "tín hiệu và xét vi phạm theo vạch dừng ảo có hướng.",
        "Huấn luyện lại và đánh giá định lượng các mô hình học sâu trên dữ liệu giao "
        "thông Việt Nam, có đối chiếu với mô hình gốc huấn luyện trên tập COCO để xác "
        "định mức độ cần thiết của việc huấn luyện lại.",
        "Thiết kế bộ đọc biển số tự động kèm cổng kiểm tra cấu trúc biển Việt Nam theo "
        "Thông tư 24/2023/TT-BGTVT, có khả năng sửa các lỗi OCR thường gặp.",
        "Thiết kế cơ chế giao hàng bền vững bảo đảm hai tính chất: không mất hồ sơ khi "
        "mất kết nối hoặc khởi động lại, và không tạo hồ sơ trùng khi gửi lại.",
        "Xây dựng máy chủ trung tâm cung cấp REST API có xác thực và phân quyền theo ba "
        "vai trò, lưu hồ sơ trong PostgreSQL và ảnh bằng chứng trong MinIO.",
        "Xây dựng bảng điều khiển web phục vụ thống kê, tra cứu, duyệt hồ sơ, xem camera "
        "trực tiếp và hiệu chuẩn vạch dừng từ xa với hiệu lực tức thời.",
        "Tích hợp trợ lý cho phép đặt câu hỏi bằng tiếng Việt và nhận kết quả dưới dạng "
        "bảng hoặc biểu đồ, với cơ chế chặn câu lệnh SQL nguy hiểm.",
        "Kiểm thử hệ thống ở nhiều mức và đóng gói toàn bộ để vận hành bằng một lệnh.",
    ]),
    P("Mục tiêu được coi là đạt khi hệ thống chạy thông suốt từ đầu vào video đến hồ sơ "
      "hiển thị trên giao diện web, các chỉ tiêu hiệu năng đo được nằm trong ngưỡng chấp "
      "nhận, và toàn bộ kiểm thử đã viết đều cho kết quả đạt."),

    H2("1.3. Đối tượng nghiên cứu"),
    P("Đối tượng nghiên cứu của đồ án gồm hai nhóm: nhóm đối tượng nghiệp vụ mà hệ "
      "thống nhận biết, và nhóm đối tượng kỹ thuật mà đồ án tập trung giải quyết."),
    P("Về nghiệp vụ, hệ thống làm việc với bốn loại đối tượng sau:"),
    UL([
        "Phương tiện đường bộ thuộc bốn lớp: ô tô con, xe mô tô và xe gắn máy, xe "
        "van/xe khách, xe tải. Trong đó xe mô tô là lớp khó nhất vì kích thước nhỏ trong "
        "khung hình, dễ bị che khuất và chiếm tỉ trọng lớn nhất trong dữ liệu thực tế.",
        "Tín hiệu đèn giao thông ba trạng thái đỏ, vàng, xanh, quan sát qua một vùng "
        "quan tâm cố định trong khung hình do người vận hành hiệu chuẩn.",
        "Biển số xe cơ giới Việt Nam một dòng, cấu trúc gồm hai chữ số mã tỉnh từ 11 đến "
        "99, tiếp theo là ký tự seri và bốn đến năm chữ số, theo quy định tại Thông tư "
        "24/2023/TT-BGTVT.",
        "Hành vi vi phạm, được định nghĩa hình học là việc phương tiện cắt qua vạch dừng "
        "theo hướng giám sát trong khi trạng thái đèn đỏ đã ổn định.",
    ]),
    P("Về kỹ thuật, đồ án tập trung nghiên cứu các vấn đề sau: kiến trúc phân tán theo "
      "mô hình tính toán biên; mẫu thiết kế outbox cho giao hàng bền vững trong điều "
      "kiện mạng không tin cậy; cơ chế debounce có trễ để ổn định tín hiệu nhiễu; bài "
      "toán domain-shift khi mô hình đạt độ chính xác cao trên tập kiểm định nhưng giảm "
      "đáng kể trên dữ liệu thực tế; và mô hình xác thực không trạng thái kết hợp phân "
      "quyền theo vai trò cho cả người dùng lẫn tác nhân máy."),

    H2("1.4. Phạm vi"),
    P("Phạm vi thực hiện:"),
    UL([
        "Nguồn video đầu vào là tệp MP4 được phát lặp để mô phỏng camera, hoặc luồng "
        "RTSP; camera cố định, giám sát một hướng của một nút giao.",
        "Node biên chạy trên một máy trạm có GPU NVIDIA, đóng gói bằng Docker.",
        "Một máy chủ trung tâm duy nhất; kiến trúc được thiết kế để tiếp nhận nhiều node "
        "biên nhưng quá trình kiểm thử chỉ thực hiện với một node.",
        "Ba vai trò người dùng: Administrator, Operator và Officer.",
        "Loại vi phạm được xét là vượt đèn đỏ; hồ sơ tạo ra ở trạng thái chờ duyệt và "
        "do con người ra quyết định cuối cùng.",
        "Giao diện web hỗ trợ tra cứu, thống kê, duyệt, xem camera trực tiếp, hiệu chuẩn "
        "và hỏi dữ liệu bằng tiếng Việt.",
    ]),
    P("Ngoài phạm vi thực hiện:"),
    UL([
        "Không tự động ra quyết định xử phạt hành chính hay sinh biên bản có giá trị "
        "pháp lý; hệ thống chỉ lập hồ sơ và cung cấp bằng chứng.",
        "Không xử lý luồng giao thông nhiều hướng từ cùng một camera; mỗi node biên "
        "giám sát một hướng.",
        "Không tích hợp với cơ sở dữ liệu đăng ký xe quốc gia để xác định chủ phương "
        "tiện; đây là bước đối soát thuộc hệ thống nghiệp vụ khác.",
        "Chưa kiểm chứng ở quy mô lớn trong điều kiện thời tiết xấu, ban đêm không có "
        "chiếu sáng bổ trợ, hoặc ngược sáng mạnh.",
        "Chưa đo tải HTTP với nhiều người dùng và nhiều node đồng thời.",
    ]),

    H2("1.5. Phương pháp thực hiện"),
    P("Đồ án áp dụng phương pháp phát triển lặp và tăng dần: mỗi chức năng được cài đặt "
      "ở mức tối thiểu, đo lường bằng số liệu thực, rồi tinh chỉnh dựa trên kết quả đo. "
      "Nguyên tắc xuyên suốt là mọi khẳng định về chất lượng đều phải có bằng chứng đo "
      "lường hoặc kiểm thử đi kèm; những hạng mục chưa đo được sẽ được ghi nhận là chưa "
      "đo thay vì ước lượng."),
    P("Quy trình thực hiện được chia thành các giai đoạn sau:"),
    TB(["Giai đoạn", "Nội dung công việc", "Sản phẩm"], [
        ["1. Khảo sát",
         "Tìm hiểu quy trình xử lý vi phạm hiện nay; khảo sát các hệ thống camera giám "
         "sát đã triển khai tại Việt Nam; nghiên cứu tài liệu học thuật về phát hiện đối "
         "tượng, theo dõi đa đối tượng và nhận dạng biển số.",
         "Phân tích hiện trạng, bảng yêu cầu chức năng và phi chức năng"],
        ["2. Phân tích",
         "Xác định actor và quyền hạn; xây dựng sơ đồ use case; đặc tả các use case "
         "quan trọng; mô hình hoá luồng xử lý bằng sơ đồ hoạt động và máy trạng thái.",
         "Bộ sơ đồ use case, activity, state machine"],
        ["3. Thiết kế",
         "Thiết kế kiến trúc ba tầng; xây dựng sơ đồ lớp; thiết kế mô hình thực thể quan "
         "hệ và lược đồ CSDL; thiết kế bề mặt API; thiết kế kiến trúc thông tin và "
         "wireflow cho giao diện web.",
         "Tài liệu thiết kế, ERD, đặc tả API, thiết kế UI/UX"],
        ["4. Huấn luyện mô hình",
         "Chuẩn bị và gán nhãn dữ liệu; huấn luyện lại YOLO26m cho bốn lớp phương tiện, "
         "YOLO26n-cls cho phân loại đèn, mô hình phát hiện biển số; đánh giá bằng "
         "precision, recall, mAP50, mAP50-95 và đối chiếu với mô hình gốc.",
         "Ba tệp trọng số và bảng số liệu benchmark"],
        ["5. Cài đặt",
         "Lập trình node biên bằng Python; máy chủ trung tâm bằng Java với Spring Boot; "
         "bảng điều khiển web bằng TypeScript với Next.js; đóng gói toàn bộ bằng Docker "
         "Compose.",
         "Mã nguồn ba tầng, tập tin cấu hình triển khai"],
        ["6. Kiểm thử",
         "Viết kiểm thử đơn vị cho logic nghiệp vụ; chạy end-to-end trên video có nhãn và "
         "video đối chứng; kiểm thử tích hợp và phân quyền trên stack đang chạy; đo hiệu "
         "năng.",
         "Bộ kiểm thử, báo cáo kết quả"],
        ["7. Hoàn thiện",
         "Chụp ảnh minh hoạ giao diện; dựng lại các sơ đồ cho khớp mã nguồn hiện hành; "
         "viết báo cáo.",
         "Báo cáo đồ án, bộ hình minh hoạ"],
    ], "Bảng 1.1. Các giai đoạn thực hiện đồ án"),
    P("Về công cụ, đồ án sử dụng Git để quản lý phiên bản theo quy ước Conventional "
      "Commits; GitHub Actions để chạy kiểm thử và biên dịch tự động ở ba tầng mỗi khi "
      "có thay đổi; pytest cho kiểm thử đơn vị phần Python; PlantUML và Mermaid để dựng "
      "sơ đồ; Docker và Docker Compose để đóng gói và vận hành."),
    P("Cấu trúc của báo cáo gồm bảy chương. Chương 1 trình bày lý do chọn đề tài, mục "
      "tiêu, đối tượng, phạm vi và phương pháp thực hiện. Chương 2 khảo sát hiện trạng, "
      "các hệ thống liên quan và xác định yêu cầu. Chương 3 phân tích hệ thống thông qua "
      "actor, use case, sơ đồ hoạt động và sơ đồ sequence. Chương 4 trình bày thiết kế "
      "kiến trúc, sơ đồ lớp, ERD, thiết kế CSDL, thiết kế API và thiết kế giao diện. "
      "Chương 5 mô tả quá trình cài đặt và trình diễn hệ thống. Chương 6 trình bày các "
      "ca kiểm thử và kết quả đo được. Chương 7 kết luận về kết quả đạt được, hạn chế "
      "và hướng phát triển."),
    PB,

    # ======================================================= CHƯƠNG 2
    H1("CHƯƠNG 2. KHẢO SÁT VÀ PHÂN TÍCH"),

    H2("2.1. Khảo sát hiện trạng"),
    H3("2.1.1. Quy trình xử lý vi phạm hiện nay"),
    P("Quy trình xử lý hành vi vượt đèn đỏ tại Việt Nam hiện diễn ra theo hai nhánh song "
      "song. Nhánh thứ nhất là xử lý trực tiếp: cảnh sát giao thông đứng tại nút giao, "
      "quan sát bằng mắt, ra hiệu lệnh dừng phương tiện, kiểm tra giấy tờ và lập biên "
      "bản tại chỗ. Nhánh thứ hai là xử lý qua ghi hình, thường gọi là phạt nguội: hình "
      "ảnh được thu thập từ camera giám sát hoặc thiết bị ghi hình của lực lượng chức "
      "năng, sau đó được xem lại để trích xuất các trường hợp vi phạm, đối chiếu biển số "
      "với cơ sở dữ liệu đăng ký xe, xác định chủ phương tiện và gửi thông báo."),
    IMG(T + "/asis-process.png", 145,
        "Hình 2.1. Quy trình xử lý vi phạm vượt đèn đỏ hiện nay (as-is)"),
    P("Điểm nghẽn của cả hai nhánh nằm ở khâu nhận biết và trích xuất. Ở nhánh trực "
      "tiếp, năng lực phát hiện bị giới hạn bởi số lượng cán bộ có thể bố trí và tầm "
      "quan sát tại một thời điểm; một nút giao rộng có nhiều hướng xung đột thì một tổ "
      "công tác khó bao quát hết. Ở nhánh phạt nguội, khối lượng video cần xem lại rất "
      "lớn trong khi tỉ lệ khung hình thực sự chứa vi phạm lại nhỏ, khiến công sức bỏ ra "
      "không tỉ lệ với số hồ sơ lập được."),

    H3("2.1.2. Hạn chế của quy trình thủ công"),
    TB(["Nhóm hạn chế", "Biểu hiện cụ thể", "Hệ quả"], [
        ["Độ bao phủ",
         "Chỉ giám sát được nơi có cán bộ trực hoặc nơi có camera được xem lại; phụ "
         "thuộc ca kíp, thời tiết và quân số.",
         "Tỉ lệ vi phạm được phát hiện thấp hơn nhiều so với thực tế; hành vi dễ tái "
         "diễn khi vắng lực lượng chức năng."],
        ["Năng suất lao động",
         "Một cán bộ phải xem hàng giờ video để tìm vài chục giây có vi phạm.",
         "Chi phí nhân lực lớn, không mở rộng được khi số nút giao tăng."],
        ["Tính khách quan",
         "Kết quả phụ thuộc sự tập trung và kinh nghiệm của người xem; dễ bỏ sót ở thời "
         "điểm mất chú ý.",
         "Thiếu nhất quán giữa các ca trực; khó bảo đảm công bằng."],
        ["Chuẩn hoá bằng chứng",
         "Bằng chứng ở dạng mô tả hoặc ảnh chụp tuỳ ý, không có cấu trúc trường dữ liệu "
         "thống nhất.",
         "Khó đối soát tự động, khó tổng hợp thống kê, khó truy vết."],
        ["Thời gian phản hồi",
         "Từ khi vi phạm xảy ra đến khi chủ phương tiện nhận thông báo có thể kéo dài "
         "nhiều ngày.",
         "Giảm tác dụng răn đe của biện pháp xử phạt."],
        ["Khả năng tổng hợp",
         "Số liệu nằm rải rác trong các biên bản giấy hoặc tệp rời rạc.",
         "Không phân tích được xu hướng theo giờ, theo nút giao để phục vụ tổ chức giao "
         "thông."],
    ], "Bảng 2.1. Hạn chế của quy trình xử lý thủ công"),
    P("Từ khảo sát trên, đồ án xác định bài toán cần giải quyết là tự động hoá khâu nhận "
      "biết và lập hồ sơ, đồng thời chuẩn hoá bằng chứng ở dạng có cấu trúc, để con "
      "người chỉ cần thực hiện phần việc thực sự cần phán đoán nghiệp vụ là duyệt hồ sơ. "
      "Hình 2.2 mô tả quy trình đề xuất: khâu nhận biết và lập hồ sơ được chuyển cho hệ "
      "thống tự động thực hiện liên tục, còn con người tập trung vào khâu duyệt và ra "
      "quyết định xử lý trên bộ hồ sơ đã chuẩn hoá."),
    IMG(T + "/tobe-process.png", 138,
        "Hình 2.2. Quy trình đề xuất khi có hệ thống RLVD (to-be)"),

    H2("2.2. Các hệ thống liên quan"),
    H3("2.2.1. Hệ thống camera ứng dụng trí tuệ nhân tạo của lực lượng cảnh sát giao thông"),
    P("Trong những năm gần đây, Cục Cảnh sát giao thông đã triển khai hệ thống camera "
      "tích hợp trí tuệ nhân tạo tại một số tuyến đường và nút giao. Theo thông tin công "
      "bố trên báo chí, hệ thống này có khả năng tự động nhận diện hơn hai mươi hành vi "
      "vi phạm khác nhau, từ không chấp hành tín hiệu đèn, chạy quá tốc độ, đi ngược "
      "chiều, lấn làn, dừng đỗ trái quy định, đến các hành vi khó quan sát hơn như "
      "không đội mũ bảo hiểm, không thắt dây an toàn hay sử dụng điện thoại khi lái xe. "
      "Về hạ tầng, hệ thống sử dụng camera độ phân giải cao, có ống kính zoom quang học "
      "và cảm biến hồng ngoại để ghi hình trong điều kiện thiếu sáng; toàn bộ hình ảnh "
      "được truyền về Trung tâm thông tin chỉ huy để phân tích. Sau khi ghi nhận vi "
      "phạm, hệ thống tự động trích xuất ảnh hoặc đoạn video liên quan kèm tuyến đường, "
      "thời gian và hành vi, đối chiếu với cơ sở dữ liệu đăng ký xe để xác định chủ "
      "phương tiện, rồi gửi thông báo qua ứng dụng dành cho người dân trong khoảng hai "
      "giờ."),
    P("Một số liệu đáng chú ý từ đợt thí điểm trên phố Phạm Văn Bạch, Hà Nội: chỉ trong "
      "khoảng thời gian từ 0 giờ đến 12 giờ của một ngày, hệ thống đã phát hiện 1.798 "
      "trường hợp vi phạm. Con số này cho thấy quy mô của nhu cầu tự động hoá — không "
      "lực lượng nào có thể lập biên bản thủ công với khối lượng như vậy. Tại Thành phố "
      "Hồ Chí Minh, hệ thống camera giám sát cũng đã được lắp đặt trên nhiều tuyến đường "
      "trọng điểm, kết nối về Trung tâm điều hành giao thông thông minh, sử dụng công "
      "nghệ nhận dạng biển số tự động để phát hiện vượt đèn đỏ, đi sai làn và dừng đỗ "
      "sai quy định."),
    P("Đánh giá dưới góc độ đồ án: đây là các hệ thống quy mô lớn, có hạ tầng camera "
      "chuyên dụng và được tích hợp sâu với cơ sở dữ liệu quốc gia về đăng ký xe. Chúng "
      "xác nhận tính đúng đắn của hướng tiếp cận tự động hoá, đồng thời cho thấy khoảng "
      "trống mà một hệ thống cỡ nhỏ có thể lấp đầy: chi phí triển khai. Các hệ thống "
      "thương mại đòi hỏi camera chuyên dụng có hồng ngoại và zoom quang học, trong khi "
      "đồ án này hướng tới việc chạy được trên camera giám sát thông thường và phần cứng "
      "phổ thông."),

    H3("2.2.2. Các giải pháp thương mại và xu hướng kỹ thuật"),
    P("Khảo sát thị trường thiết bị giám sát giao thông tại Việt Nam cho thấy các thông "
      "số triển khai phổ biến là camera độ phân giải hai đến bốn megapixel, cảm biến "
      "kích thước 1/2,8 đến 1/1,8 inch, tốc độ 25 đến 30 khung hình mỗi giây, có đèn "
      "hồng ngoại tầm 30 đến 100 mét, hỗ trợ dải tương phản động rộng để xử lý ngược "
      "sáng, đạt chuẩn bảo vệ IP66/IP67. Đáng chú ý là xu hướng xử lý ngay tại thiết bị: "
      "nhiều dòng camera đã tích hợp bộ xử lý chuyên dụng để chạy mô hình ANPR và phân "
      "loại phương tiện ngay trên camera, chỉ gửi sự kiện về trung tâm. Xu hướng quốc tế "
      "là kiến trúc lai giữa biên và đám mây: xử lý thời gian thực tại biên, còn huấn "
      "luyện và cập nhật mô hình thực hiện ở phía máy chủ."),
    P("Hướng tiếp cận này trùng khớp với lựa chọn kiến trúc của đồ án. Điểm khác biệt "
      "là các giải pháp thương mại xử lý tại biên thường nằm trong firmware của camera "
      "nên khó tuỳ biến logic nghiệp vụ, trong khi đồ án đặt node biên trên một máy tính "
      "riêng, cho phép toàn quyền thay đổi quy tắc xét vi phạm và bổ sung loại vi phạm "
      "mới mà không phụ thuộc nhà sản xuất thiết bị."),

    H3("2.2.3. Các công trình và dự án mã nguồn mở liên quan"),
    P("Về nền tảng học thuật, đồ án kế thừa ba nhóm công trình. Thứ nhất là họ mô hình "
      "phát hiện đối tượng thời gian thực YOLO do Ultralytics phát triển, với ưu điểm "
      "cân bằng được độ chính xác và tốc độ suy luận trên GPU phổ thông. Thứ hai là "
      "ByteTrack, phương pháp theo dõi đa đối tượng liên kết mọi khung giới hạn phát hiện "
      "được thay vì chỉ những khung có độ tin cậy cao; đặc điểm này rất phù hợp với "
      "bối cảnh xe máy kích thước nhỏ và thường xuyên bị che khuất. Thứ ba là các thư "
      "viện nhận dạng biển số mã nguồn mở, trong đó fast-plate-ocr cung cấp mô hình "
      "nhẹ chạy được trên cả CPU."),
    P("Về dự án mã nguồn mở, đã có một số repository công khai giải quyết bài toán phát "
      "hiện vượt đèn đỏ, song phần lớn dừng ở mức minh hoạ thuật toán trên video ngắn: "
      "phát hiện phương tiện, xác định màu đèn và in kết quả ra màn hình. Những khía "
      "cạnh làm nên một hệ thống dùng được trong thực tế — giao hàng tin cậy khi mạng "
      "không ổn định, chống trùng lặp hồ sơ, hiệu chuẩn từ xa, phân quyền người dùng, "
      "quy trình duyệt có con người, lưu trữ ảnh bằng chứng — hầu như không được đề cập. "
      "Đây chính là phần đồ án tập trung đóng góp."),

    H3("2.2.4. Vị trí của hệ thống RLVD"),
    TB(["Tiêu chí", "Xử lý thủ công", "Hệ thống thương mại quy mô lớn", "RLVD (đồ án)"], [
        ["Khâu nhận biết vi phạm", "Con người quan sát trực tiếp hoặc xem lại video",
         "Tự động bằng camera AI chuyên dụng", "Tự động bằng node biên chạy trên máy tính phổ thông có GPU"],
        ["Yêu cầu thiết bị ghi hình", "Camera bất kỳ", "Camera chuyên dụng 4K, zoom quang, hồng ngoại",
         "Camera giám sát thông thường hoặc tệp video"],
        ["Vị trí xử lý", "Tại trung tâm, bởi con người", "Tại camera (firmware) và trung tâm",
         "Tại node biên riêng, logic nghiệp vụ mở"],
        ["Khả năng tuỳ biến quy tắc", "Hoàn toàn linh hoạt nhưng chậm", "Hạn chế, phụ thuộc nhà cung cấp",
         "Toàn quyền, thay đổi được trong mã nguồn"],
        ["Giao hàng khi mất mạng", "Không áp dụng", "Tuỳ giải pháp",
         "Kho outbox SQLite tại node, gửi lại theo lô idempotent"],
        ["Quyết định xử lý", "Con người", "Tự động kết hợp đối soát CSDL đăng ký xe",
         "Hệ thống lập hồ sơ, con người duyệt (human-in-the-loop)"],
        ["Quy mô triển khai", "Theo ca kíp", "Hàng nghìn camera",
         "Một nút giao mỗi node, mở rộng bằng cách thêm node"],
        ["Chi phí", "Nhân lực cao, thiết bị thấp", "Đầu tư thiết bị và hạ tầng rất lớn",
         "Thiết bị thấp, chạy được trên GPU phổ thông"],
    ], "Bảng 2.2. So sánh hệ thống RLVD với các giải pháp liên quan"),
    P("Kết luận của phần khảo sát: nhu cầu tự động hoá là có thật và đã được khẳng định "
      "bằng các triển khai thực tế; tuy nhiên vẫn tồn tại khoảng trống cho một hệ thống "
      "mã nguồn mở, chạy trên phần cứng phổ thông, có logic nghiệp vụ mở và cơ chế giao "
      "hàng tin cậy. Đồ án định vị sản phẩm của mình vào khoảng trống đó."),

    H2("2.3. Yêu cầu chức năng"),
    P("Từ khảo sát hiện trạng và phân tích khoảng trống, đồ án xác định các yêu cầu chức "
      "năng sau. Mỗi yêu cầu được đánh mã để truy vết đến use case ở Chương 3, thiết kế "
      "ở Chương 4 và ca kiểm thử ở Chương 6."),
    TB(["Mã", "Yêu cầu chức năng", "Mô tả"], [
        ["FR-01", "Phát hiện phương tiện",
         "Nhận diện phương tiện thuộc bốn lớp trong từng khung hình tại node biên, kèm "
         "khung giới hạn và độ tin cậy."],
        ["FR-02", "Theo dõi đa đối tượng",
         "Gán và duy trì mã theo dõi ổn định cho từng phương tiện qua các khung hình, kể "
         "cả khi bị che khuất ngắn; cung cấp thông tin chuyển động gồm vận tốc và hướng."],
        ["FR-03", "Nhận biết trạng thái đèn",
         "Xác định trạng thái đỏ, vàng hoặc xanh trong vùng quan tâm đã hiệu chuẩn; ổn "
         "định hoá kết quả để tránh dao động do nhiễu từng khung hình."],
        ["FR-04", "Hiệu chuẩn từ xa",
         "Cho phép người vận hành kẻ vạch dừng, xác định vùng quan tâm của đèn và chọn "
         "hướng giám sát thông qua giao diện web, áp dụng ngay vào pipeline đang chạy mà "
         "không cần khởi động lại node."],
        ["FR-05", "Xét vi phạm",
         "Kết luận một phương tiện vi phạm khi điểm neo của nó cắt qua vạch dừng theo "
         "đúng hướng giám sát trong lúc trạng thái đèn đỏ đã ổn định; mỗi phương tiện chỉ "
         "sinh một hồ sơ cho một lần vượt."],
        ["FR-06", "Đọc và chuẩn hoá biển số",
         "Phát hiện biển số, gắn biển với phương tiện tương ứng, đọc nội dung bằng OCR và "
         "kiểm tra cấu trúc theo quy định biển số Việt Nam, có sửa các lỗi nhận dạng "
         "thường gặp."],
        ["FR-07", "Lập hồ sơ kèm bằng chứng",
         "Tạo hồ sơ vi phạm gồm mã định danh duy nhất, ảnh bằng chứng toàn cảnh đã chú "
         "thích, biển số, trạng thái đèn, độ tin cậy, toạ độ cắt vạch và thời điểm."],
        ["FR-08", "Giao hàng bền vững",
         "Lưu hồ sơ vào kho trung gian tại node trước mọi thao tác mạng; đẩy lên trung "
         "tâm theo lô; gửi lại khi thất bại; không mất hồ sơ khi mất kết nối hoặc khởi "
         "động lại; không tạo hồ sơ trùng."],
        ["FR-09", "Quản lý node biên",
         "Node tự đăng ký với trung tâm kèm thông tin cấu hình; gửi heartbeat định kỳ; "
         "trung tâm suy ra trạng thái hoạt động dựa trên thời điểm heartbeat gần nhất."],
        ["FR-10", "Tra cứu và thống kê",
         "Tra cứu hồ sơ có phân trang phía máy chủ, lọc theo trạng thái, node và biển số; "
         "thống kê theo trạng thái, theo giờ, theo node và theo trạng thái đèn."],
        ["FR-11", "Duyệt hồ sơ",
         "Cho phép người có thẩm quyền phê duyệt hoặc từ chối hồ sơ; thay đổi trạng thái "
         "được ghi nhận kèm thời điểm; hồ sơ đã duyệt có thể đặt lại trạng thái chờ."],
        ["FR-12", "Xem camera trực tiếp",
         "Phát luồng video từ node biên qua giao thức HLS trên giao diện web, hỗ trợ xem "
         "nhiều camera đồng thời theo lưới; cung cấp ảnh chụp nhanh từng khung hình."],
        ["FR-13", "Hỏi dữ liệu bằng tiếng Việt",
         "Tiếp nhận câu hỏi tự nhiên bằng tiếng Việt, sinh câu lệnh SQL, thực thi và trả "
         "kết quả dạng bảng hoặc biểu đồ; chỉ cho phép truy vấn đọc."],
        ["FR-14", "Xác thực và phân quyền",
         "Đăng nhập bằng tên người dùng và mật khẩu; cấp JWT; phân quyền theo ba vai trò; "
         "hỗ trợ làm mới phiên và đăng xuất."],
        ["FR-15", "Hiển thị trạng thái đèn trực tiếp",
         "Công bố trạng thái đèn đã ổn định hoá để giao diện web hiển thị theo thời gian "
         "thực mà không phải chạy lại bộ phân loại."],
    ], "Bảng 2.3. Yêu cầu chức năng"),

    H2("2.4. Yêu cầu phi chức năng"),
    P("Các yêu cầu phi chức năng được phát biểu kèm chỉ tiêu đo được hoặc cách đo, để có "
      "thể kiểm chứng ở Chương 6."),
    TB(["Mã", "Yêu cầu", "Chỉ tiêu và cách kiểm chứng"], [
        ["NFR-01", "Tính thời gian thực",
         "Tốc độ xử lý của node biên không thấp hơn tốc độ khung hình của nguồn; với "
         "nguồn tốc độ thấp hơn thì phải bám được thời gian thực nhờ cơ chế bỏ khung hình "
         "cũ. Đo bằng FPS suy luận của mô hình và số khung hình đã xử lý trên tổng thời "
         "gian chạy."],
        ["NFR-02", "Độ trễ ổn định tín hiệu đèn",
         "Thời gian từ khi đèn đổi màu đến khi trạng thái ổn định được công bố không quá "
         "0,8 giây. Đo bằng số khung hình debounce nhân với chu kỳ khung hình của nguồn."],
        ["NFR-03", "Độ tin cậy giao hàng",
         "Không mất hồ sơ trong các kịch bản mất kết nối và khởi động lại node. Kiểm "
         "chứng bằng cách ngắt máy chủ trung tâm, cho node tiếp tục phát hiện, sau đó "
         "khôi phục kết nối và đối chiếu số hồ sơ."],
        ["NFR-04", "Tính idempotent",
         "Gửi lại cùng một lô hồ sơ không tạo bản ghi trùng. Kiểm chứng bằng cách gửi hai "
         "lần và đối chiếu số lượng chấp nhận và số lượng trùng ở phản hồi."],
        ["NFR-05", "Bảo mật",
         "Mật khẩu được băm bằng BCrypt; JWT ký bằng HS256 với khoá bí mật tối thiểu 32 "
         "ký tự và từ chối khởi động nếu khoá yếu; token node biên được so sánh theo thời "
         "gian không đổi; API ghi của node biên yêu cầu token và giới hạn tần suất; "
         "nguồn gốc truy cập bị kiểm soát bằng danh sách cho phép; không có thông tin bí "
         "mật nào nằm cứng trong mã nguồn."],
        ["NFR-06", "Khả năng mở rộng",
         "Thêm node biên mới không đòi hỏi thay đổi mã nguồn ở trung tâm; hồ sơ được định "
         "danh theo node và theo lần chạy để không xung đột; thống kê được tính bằng truy "
         "vấn tổng hợp phía CSDL thay vì tải toàn bộ dữ liệu vào bộ nhớ."],
        ["NFR-07", "Khả năng vận hành",
         "Toàn bộ hệ thống khởi động bằng một lệnh; mỗi dịch vụ có kiểm tra sức khoẻ; "
         "dịch vụ phụ thuộc chỉ khởi động sau khi dịch vụ nền đã sẵn sàng; chính sách tự "
         "khởi động lại khi lỗi."],
        ["NFR-08", "Khả năng quan sát",
         "Node biên công bố số khung hình đã xử lý, số vi phạm đã phát hiện, FPS và tự "
         "báo trạng thái không sẵn sàng khi pipeline ngừng tiến triển; mỗi tầng ghi nhật "
         "ký có ngữ cảnh để truy vết theo mã hồ sơ."],
        ["NFR-09", "Khả năng khôi phục",
         "Dữ liệu CSDL và ảnh bằng chứng nằm trong volume bền vững; có kịch bản sao lưu "
         "và khôi phục; hồ sơ chưa gửi được giữ lại qua các lần khởi động lại container."],
        ["NFR-10", "Nhất quán thời gian",
         "Mọi dấu thời gian trong hệ thống dùng cùng múi giờ Việt Nam để số liệu thống kê "
         "theo giờ không bị lệch."],
        ["NFR-11", "Khả năng bảo trì",
         "Tách bạch trách nhiệm giữa các tầng; tham số vận hành đọc từ biến môi trường "
         "thay vì nằm cứng; các quy tắc nghiệp vụ phức tạp có kiểm thử đơn vị kèm theo; "
         "danh sách đường dẫn cần token node chỉ khai báo ở một nơi duy nhất."],
        ["NFR-12", "Tính sử dụng",
         "Giao diện bằng tiếng Việt; thông báo lỗi diễn đạt theo ngôn ngữ nghiệp vụ thay "
         "vì mã lỗi kỹ thuật; thao tác hiệu chuẩn thực hiện trực tiếp trên khung hình "
         "video; có nhiều chủ đề giao diện phù hợp điều kiện ánh sáng khác nhau."],
    ], "Bảng 2.4. Yêu cầu phi chức năng"),
    P("Hai yêu cầu NFR-03 và NFR-04 có vai trò đặc biệt quan trọng vì chúng quyết định "
      "kiến trúc của tầng giao tiếp giữa node biên và trung tâm, được trình bày chi tiết "
      "ở mục 4.1 và kiểm chứng ở Chương 6."),
    PB,

    # ======================================================= CHƯƠNG 3
    H1("CHƯƠNG 3. PHÂN TÍCH HỆ THỐNG"),

    H2("3.1. Actors"),
    P("Hệ thống có ba actor là con người, tương ứng với ba vai trò được cài đặt trong cơ "
      "chế phân quyền, và hai actor là hệ thống. Việc tách bạch vai trò xuất phát từ "
      "nguyên tắc phân nhiệm: người cấu hình thiết bị không nên đồng thời là người ra "
      "quyết định xử lý hồ sơ, và cả hai nhóm này đều không nên có quyền quản trị người "
      "dùng."),
    IMG(V + "/actor-rbac.png", 150,
        "Hình 3.1. Quan hệ actor, vai trò và quyền hạn trong hệ thống"),
    TB(["Actor", "Loại", "Vai trò hệ thống", "Quyền hạn chính"], [
        ["Administrator", "Con người, actor chính",
         "Người quản trị hệ thống, chịu trách nhiệm về tài khoản và dữ liệu.",
         "Toàn bộ quyền của Operator và Officer; quản lý tài khoản người dùng; xoá hồ sơ "
         "vi phạm; xem toàn bộ dữ liệu và thống kê."],
        ["Operator", "Con người, actor chính",
         "Kỹ thuật viên vận hành thiết bị tại các nút giao.",
         "Xem và cập nhật cấu hình node biên; hiệu chuẩn vạch dừng, vùng đèn và hướng "
         "giám sát; xem camera trực tiếp và ảnh chụp nhanh; tra cứu hồ sơ và thống kê. "
         "Không có quyền duyệt hồ sơ."],
        ["Officer", "Con người, actor chính",
         "Cán bộ nghiệp vụ ra quyết định xử lý hồ sơ.",
         "Tra cứu hồ sơ; xem chi tiết và bằng chứng; phê duyệt hoặc từ chối hồ sơ; xem "
         "thống kê; đặt câu hỏi bằng tiếng Việt. Không có quyền hiệu chuẩn thiết bị và "
         "không có quyền xoá hồ sơ."],
        ["Edge Node", "Hệ thống, actor máy",
         "Tác nhân tự động đặt tại nút giao, vừa là nguồn dữ liệu vừa là nơi thi hành "
         "lệnh hiệu chuẩn.",
         "Đăng ký với trung tâm; gửi heartbeat; phát hiện vi phạm; đọc biển số; đẩy hồ sơ "
         "theo lô; tải ảnh bằng chứng; tiếp nhận lệnh hiệu chuẩn; phục vụ luồng HLS và "
         "ảnh chụp nhanh."],
        ["Central Server", "Hệ thống, actor phụ",
         "Máy chủ trung tâm điều phối dữ liệu và làm cầu nối giữa giao diện web với node "
         "biên.",
         "Lưu hồ sơ và ảnh; cung cấp REST API; xác thực và phân quyền; tính thống kê; "
         "chuyển tiếp lệnh hiệu chuẩn xuống node; đại diện gọi mô hình ngôn ngữ để sinh "
         "SQL."],
    ], "Bảng 3.1. Danh sách actor và quyền hạn"),
    P("Một điểm cần lưu ý về mô hình actor: Edge Node tham gia hệ thống theo hai chiều. "
      "Ở chiều chủ động, node tự khởi xướng việc đăng ký, heartbeat và đẩy hồ sơ, dùng "
      "token dành riêng cho node chứ không dùng JWT của người dùng. Ở chiều bị động, node "
      "tiếp nhận lệnh hiệu chuẩn do trung tâm chuyển tiếp xuống, và lệnh này được bảo vệ "
      "bằng một token khác. Hai chiều dùng hai cơ chế bảo vệ riêng biệt vì mức độ rủi ro "
      "khác nhau: chiều đẩy dữ liệu chỉ có thể tạo hồ sơ, còn chiều tiếp nhận lệnh có "
      "thể thay đổi hành vi xét vi phạm của cả một nút giao."),

    H2("3.2. Use Case Diagram"),
    P("Sơ đồ use case tổng thể được trình bày ở Hình 3.2. Hệ thống gồm mười một use case "
      "chính, chia thành bốn nhóm: nhóm vận hành node (đăng ký và heartbeat, quản lý "
      "node, hiệu chuẩn, xem camera), nhóm xử lý tại biên (phát hiện vi phạm, đọc biển "
      "số, gửi hồ sơ, tải ảnh bằng chứng), nhóm nghiệp vụ hồ sơ (tra cứu, thống kê, "
      "duyệt) và nhóm hỗ trợ (hỏi dữ liệu bằng tiếng Việt)."),
    IMG(T + "/use-case-full.png", 150, "Hình 3.2. Sơ đồ use case tổng thể"),
    P("Các quan hệ giữa use case phản ánh đúng thứ tự phụ thuộc nghiệp vụ. Use case phát "
      "hiện vi phạm bao gồm use case đọc biển số và use case gửi hồ sơ, vì một hồ sơ hoàn "
      "chỉnh cần cả biển số lẫn đường truyền về trung tâm. Use case gửi hồ sơ bao gồm use "
      "case tải ảnh bằng chứng. Use case hiệu chuẩn có quan hệ kích hoạt với use case "
      "phát hiện vi phạm, bởi khi chưa có vạch dừng thì bộ xét vi phạm ở trạng thái ngừng "
      "hoạt động — pipeline vẫn chạy, vẫn phát hiện và theo dõi phương tiện, nhưng không "
      "sinh hồ sơ nào. Use case đăng ký node là điều kiện tiên quyết của use case hiệu "
      "chuẩn, vì trung tâm cần biết địa chỉ của node mới có thể chuyển tiếp lệnh xuống."),
    P("Ba sơ đồ chi tiết theo từng vai trò dưới đây làm rõ ranh giới quyền hạn. Đây cũng "
      "là cơ sở để xây dựng các kiểm thử phân quyền ở Chương 6."),
    IMG(T + "/use-case-admin.png", 140, "Hình 3.3. Sơ đồ use case của Administrator"),
    P("Administrator bao trùm toàn bộ use case của hai vai trò còn lại, đồng thời giữ hai "
      "quyền riêng là quản lý tài khoản người dùng và xoá hồ sơ vi phạm. Quyền xoá được "
      "tách riêng khỏi quyền duyệt để tránh việc một hồ sơ bị loại khỏi hệ thống mà "
      "không để lại dấu vết trạng thái."),
    IMG(V + "/use-case-officer.png", 150, "Hình 3.4. Sơ đồ use case của Officer"),
    P("Officer là vai trò sử dụng hệ thống thường xuyên nhất. Luồng làm việc điển hình là "
      "mở danh sách hồ sơ chờ duyệt, xem chi tiết từng hồ sơ cùng ảnh bằng chứng, rồi phê "
      "duyệt hoặc từ chối. Hai use case tra cứu và thống kê phục vụ việc đối chiếu khi "
      "cần thêm căn cứ; use case hỏi dữ liệu bằng tiếng Việt cho phép đặt những câu hỏi "
      "tổng hợp mà giao diện thống kê cố định chưa đáp ứng."),
    IMG(V + "/use-case-operator.png", 150, "Hình 3.5. Sơ đồ use case của Operator"),
    P("Operator tập trung vào khâu thiết bị: kiểm tra trạng thái node, xem camera trực "
      "tiếp để đánh giá góc quay, và thực hiện hiệu chuẩn. Điểm đáng chú ý là Operator "
      "không có quyền duyệt hồ sơ, còn Officer không có quyền hiệu chuẩn — đây là ràng "
      "buộc phân nhiệm được kiểm chứng bằng kiểm thử phân quyền ở Chương 6."),

    H2("3.3. Use Case Specification"),
    P("Năm use case quan trọng nhất được đặc tả chi tiết theo mẫu thống nhất, gồm actor "
      "tham gia, tiền điều kiện, luồng sự kiện chính, các luồng thay thế và hậu điều kiện."),

    H3("3.3.1. UC-01: Phát hiện xe vượt đèn đỏ"),
    TB(["Trường", "Nội dung"], [
        ["Mã use case", "UC-01"],
        ["Tên", "Phát hiện xe vượt đèn đỏ"],
        ["Actor", "Edge Node (actor chính); Administrator và Officer tiếp nhận kết quả "
                "một cách gián tiếp"],
        ["Mô tả tóm tắt", "Node biên phân tích từng khung hình để xác định phương tiện "
                          "cắt qua vạch dừng trong khi đèn đỏ đã ổn định, từ đó lập hồ sơ "
                          "vi phạm kèm ảnh bằng chứng."],
        ["Tiền điều kiện", "Node đã khởi động và nạp xong ba tệp trọng số; nguồn video "
                           "đọc được; vạch dừng và vùng quan tâm của đèn đã được hiệu "
                           "chuẩn; kho outbox mở được."],
        ["Luồng sự kiện chính",
         "1. Node đọc một khung hình từ nguồn kèm chỉ số khung hình và dấu thời gian.\n"
         "2. Bộ phân loại đèn xác định trạng thái trong vùng quan tâm, kết hợp mô hình "
         "học sâu với bằng chứng màu HSV và vị trí bóng đèn.\n"
         "3. Bộ ổn định tín hiệu cập nhật trạng thái ổn định theo cơ chế debounce có trễ; "
         "kết quả được công bố cho API điều khiển.\n"
         "4. Mô hình phát hiện phương tiện trả về danh sách khung giới hạn kèm nhãn và độ "
         "tin cậy.\n"
         "5. Bộ theo dõi gán và duy trì mã vết cho từng phương tiện, đồng thời ước lượng "
         "vận tốc, hướng di chuyển và quỹ đạo.\n"
         "6. Với mỗi vết đã đủ số lần xác nhận, hệ thống so sánh điểm neo ở khung hình "
         "trước và khung hình hiện tại với vạch dừng.\n"
         "7. Nếu trạng thái đèn đỏ đã ổn định, vết chuyển từ phía này sang phía kia của "
         "vạch theo đúng hướng giám sát, và đoạn di chuyển thực sự cắt vạch, hệ thống "
         "kết luận vi phạm.\n"
         "8. Hồ sơ được gán mã định danh duy nhất cấu thành từ tiền tố, mã lần chạy, chỉ "
         "số khung hình và mã vết.\n"
         "9. Nếu biển số đã được đọc cho vết này, nội dung biển đã chuẩn hoá được đính "
         "kèm hồ sơ.\n"
         "10. Ảnh bằng chứng được vẽ chú thích và nén lại.\n"
         "11. Hồ sơ cùng ảnh được ghi vào kho outbox."],
        ["Luồng thay thế",
         "2a. Vùng quan tâm chưa hiệu chuẩn: bộ phân loại dùng vùng suy đoán, kết quả có "
         "thể kém tin cậy.\n"
         "3a. Trạng thái đèn không đạt độ tin cậy tối thiểu: bộ ổn định giữ nguyên trạng "
         "thái hiện tại, tăng bộ đếm không xác định; vượt ngưỡng dung sai thì quay về "
         "trạng thái không xác định và không xét vi phạm.\n"
         "6a. Vạch dừng chưa được hiệu chuẩn: bỏ qua toàn bộ bước xét vi phạm, pipeline "
         "tiếp tục phát hiện và theo dõi bình thường.\n"
         "6b. Vết đi ngược hướng giám sát: không kết luận vi phạm.\n"
         "7a. Vết đã từng được kết luận vi phạm trong lần vượt này: bỏ qua để không sinh "
         "hồ sơ trùng.\n"
         "9a. Biển số đọc được nhưng không khớp cấu trúc biển Việt Nam và không sửa được: "
         "trường biển số để trống, hồ sơ vẫn được lập.\n"
         "11a. Mã định danh đã tồn tại trong outbox: bỏ qua bản ghi mới, không báo lỗi.\n"
         "1b. Khung hình đọc lỗi hoặc mô hình suy luận lỗi: ghi nhật ký cảnh báo và "
         "chuyển sang khung hình kế tiếp, không làm gián đoạn pipeline."],
        ["Hậu điều kiện", "Hồ sơ vi phạm nằm trong kho outbox ở trạng thái chờ gửi; bộ đếm "
                          "vi phạm trong metrics được tăng; ảnh bằng chứng đã được lưu "
                          "kèm bản ghi."],
        ["Ràng buộc nghiệp vụ", "Mỗi vết chỉ sinh tối đa một hồ sơ cho một lần vượt vạch. "
                                 "Không kết luận vi phạm khi tín hiệu đèn chưa ổn định."],
    ], "Bảng 3.2. Đặc tả UC-01 Phát hiện xe vượt đèn đỏ"),

    H3("3.3.2. UC-02: Gửi hồ sơ vi phạm về trung tâm"),
    TB(["Trường", "Nội dung"], [
        ["Mã use case", "UC-02"],
        ["Tên", "Gửi hồ sơ vi phạm về trung tâm"],
        ["Actor", "Edge Node (actor chính); Central Server (actor phụ tiếp nhận)"],
        ["Mô tả tóm tắt", "Luồng nền tại node biên rút hồ sơ từ kho outbox, đẩy lên trung "
                          "tâm theo lô có tính idempotent, sau đó tải ảnh bằng chứng cho "
                          "các hồ sơ đã được chấp nhận."],
        ["Tiền điều kiện", "Kho outbox đã được khởi tạo; địa chỉ máy chủ trung tâm và "
                           "token node đã được cấu hình; luồng gửi đang chạy."],
        ["Luồng sự kiện chính",
         "1. Luồng nền thức dậy theo chu kỳ cấu hình, mặc định năm giây.\n"
         "2. Rút tối đa hai mươi hồ sơ ở trạng thái chờ gửi, chỉ đọc phần dữ liệu JSON, "
         "không đọc ảnh để tiết kiệm bộ nhớ.\n"
         "3. Đóng gói thành một lô và gửi bằng một yêu cầu POST kèm token node.\n"
         "4. Trung tâm kiểm tra từng mã định danh; bản ghi mới được lưu, bản ghi đã tồn "
         "tại được đếm là trùng và bỏ qua.\n"
         "5. Trung tâm trả về số lượng chấp nhận, số lượng trùng, số lượng lỗi kèm danh "
         "sách mã tương ứng.\n"
         "6. Node đánh dấu các hồ sơ đã được chấp nhận là đã gửi.\n"
         "7. Node rút tiếp các hồ sơ đã gửi nhưng chưa tải ảnh, tải ảnh lên trung tâm "
         "bằng yêu cầu multipart.\n"
         "8. Trung tâm lưu ảnh vào MinIO theo khoá đối tượng suy ra từ mã hồ sơ và ghi "
         "đường dẫn vào bản ghi.\n"
         "9. Node đánh dấu ảnh đã tải xong."],
        ["Luồng thay thế",
         "3a. Không thể kết nối hoặc trung tâm trả lỗi: tăng bộ đếm số lần thử của cả lô, "
         "lui lại theo cấp số nhân với trần sáu mươi giây, hồ sơ vẫn ở trạng thái chờ "
         "gửi.\n"
         "3b. Gửi theo lô thất bại lặp lại: thử gửi từng hồ sơ một để cô lập bản ghi "
         "hỏng, các bản ghi tốt vẫn được chấp nhận.\n"
         "7a. Tải ảnh thất bại: hồ sơ vẫn giữ trạng thái đã gửi, chỉ cờ ảnh chưa tải "
         "được giữ nguyên để thử lại ở chu kỳ sau.\n"
         "2a. Không có hồ sơ nào chờ gửi: luồng ngủ tiếp, không phát sinh yêu cầu mạng."],
        ["Hậu điều kiện", "Hồ sơ đã có mặt trong CSDL trung tâm với trạng thái chờ duyệt; "
                          "ảnh bằng chứng đã có trong MinIO và đường dẫn đã được ghi; kho "
                          "outbox tại node không còn bản ghi chờ gửi tương ứng."],
        ["Ràng buộc nghiệp vụ", "Thứ tự hai bước bắt buộc: dữ liệu JSON phải được chấp "
                                 "nhận trước, ảnh tải sau. Điều này bảo đảm không bao giờ "
                                 "có ảnh mồ côi trong MinIO mà thiếu bản ghi trong CSDL."],
    ], "Bảng 3.3. Đặc tả UC-02 Gửi hồ sơ vi phạm về trung tâm"),

    H3("3.3.3. UC-03: Hiệu chuẩn từ xa"),
    TB(["Trường", "Nội dung"], [
        ["Mã use case", "UC-03"],
        ["Tên", "Hiệu chuẩn vạch dừng, vùng đèn và hướng giám sát từ xa"],
        ["Actor", "Operator hoặc Administrator (actor chính); Central Server và Edge Node "
                "(actor phụ)"],
        ["Mô tả tóm tắt", "Người vận hành vẽ vạch dừng trực tiếp trên khung hình video "
                          "trực tiếp; lệnh được chuyển tiếp qua trung tâm xuống node và "
                          "có hiệu lực từ khung hình kế tiếp."],
        ["Tiền điều kiện", "Người dùng đã đăng nhập với vai trò có quyền hiệu chuẩn; node "
                           "đã đăng ký với trung tâm kèm địa chỉ và token điều khiển; node "
                           "đang phục vụ luồng video."],
        ["Luồng sự kiện chính",
         "1. Người dùng mở trang chi tiết node, chọn chế độ vẽ vạch dừng.\n"
         "2. Giao diện hiển thị khung hình trực tiếp lấy từ node qua đường proxy.\n"
         "3. Người dùng kéo hai điểm trên khung hình để xác định vạch dừng.\n"
         "4. Giao diện quy đổi toạ độ hiển thị về toạ độ gốc của khung hình video.\n"
         "5. Người dùng chọn hướng giám sát.\n"
         "6. Giao diện gửi yêu cầu POST tới trung tâm kèm JWT.\n"
         "7. Trung tâm kiểm tra vai trò, tra cứu node trong sổ đăng ký để lấy địa chỉ và "
         "token điều khiển.\n"
         "8. Trung tâm chuyển tiếp yêu cầu xuống node bằng giao thức HTTP phiên bản 1.1, "
         "kèm token điều khiển của node.\n"
         "9. Node kiểm tra token theo thời gian không đổi, ghi nhận vạch dừng mới vào "
         "biến trạng thái dùng chung có khoá bảo vệ.\n"
         "10. Ở khung hình kế tiếp, bộ xét vi phạm đọc lại vạch dừng đang hoạt động và "
         "áp dụng ngay.\n"
         "11. Giao diện truy vấn lại trạng thái hiệu chuẩn để xác nhận và vẽ vạch lên "
         "khung hình."],
        ["Luồng thay thế",
         "3a. Người dùng chọn chế độ vẽ vùng đèn: thao tác kéo thả tạo hình chữ nhật "
         "thay vì đoạn thẳng, gửi tới endpoint tương ứng.\n"
         "5a. Người dùng chỉ đổi hướng giám sát mà không vẽ lại vạch.\n"
         "8a. Node không phản hồi hoặc từ chối token: trung tâm trả lỗi cổng upstream, "
         "giao diện hiển thị thông báo và không thay đổi trạng thái.\n"
         "7a. Node chưa có trong sổ đăng ký hoặc địa chỉ không gọi được: trung tâm báo "
         "lỗi, giao diện gợi ý kiểm tra kết nối node."],
        ["Hậu điều kiện", "Vạch dừng mới có hiệu lực tức thời; nếu trước đó bộ xét vi "
                          "phạm ở trạng thái ngừng do chưa hiệu chuẩn thì nay được kích "
                          "hoạt; trạng thái hiệu chuẩn truy vấn được qua API."],
        ["Ràng buộc kỹ thuật", "Việc gọi ngược từ trung tâm xuống node phải dùng HTTP "
                               "phiên bản 1.1. Nguyên nhân là cơ chế nâng cấp giao thức "
                               "h2c từng làm rơi phần thân yêu cầu POST khi đi qua máy "
                               "chủ ứng dụng Python, dẫn đến lỗi cú pháp ở node."],
    ], "Bảng 3.4. Đặc tả UC-03 Hiệu chuẩn từ xa"),

    H3("3.3.4. UC-04: Duyệt hồ sơ vi phạm"),
    TB(["Trường", "Nội dung"], [
        ["Mã use case", "UC-04"],
        ["Tên", "Duyệt hồ sơ vi phạm"],
        ["Actor", "Officer hoặc Administrator (actor chính)"],
        ["Mô tả tóm tắt", "Cán bộ nghiệp vụ xem bằng chứng của từng hồ sơ chờ duyệt và ra "
                          "quyết định phê duyệt hoặc từ chối."],
        ["Tiền điều kiện", "Người dùng đã đăng nhập với vai trò Officer hoặc Administrator; "
                           "tồn tại ít nhất một hồ sơ ở trạng thái chờ duyệt."],
        ["Luồng sự kiện chính",
         "1. Cán bộ mở trang duyệt hồ sơ.\n"
         "2. Giao diện tải danh sách hồ sơ chờ duyệt theo lô năm mươi bản ghi, đồng thời "
         "tải trước lô kế tiếp.\n"
         "3. Cán bộ chọn một hồ sơ để xem chi tiết.\n"
         "4. Giao diện hiển thị ảnh bằng chứng, biển số nếu đọc được, trạng thái đèn kèm "
         "độ tin cậy, toạ độ cắt vạch, mã vết, node nguồn và thời điểm ghi nhận.\n"
         "5. Cán bộ đối chiếu bằng chứng trực quan với các trường dữ liệu.\n"
         "6. Cán bộ chọn phê duyệt hoặc từ chối.\n"
         "7. Giao diện gửi yêu cầu cập nhật trạng thái kèm JWT.\n"
         "8. Trung tâm kiểm tra vai trò, cập nhật trạng thái và thời điểm cập nhật.\n"
         "9. Giao diện làm mới bộ đếm hồ sơ chờ duyệt và chuyển sang hồ sơ kế tiếp."],
        ["Luồng thay thế",
         "4a. Ảnh bằng chứng chưa tải được: cán bộ căn cứ vào các trường dữ liệu còn lại "
         "để quyết định.\n"
         "6a. Hồ sơ đã ở trạng thái được chọn: nút tương ứng bị vô hiệu hoá.\n"
         "6b. Cán bộ muốn đưa hồ sơ đã xử lý về trạng thái chờ: dùng chức năng đặt lại "
         "trạng thái, chỉ khả dụng với vai trò phù hợp.\n"
         "8a. Người dùng không đủ quyền: trung tâm trả lỗi cấm truy cập, giao diện hiển "
         "thị thông báo và giữ nguyên hồ sơ.\n"
         "2a. Không còn hồ sơ chờ duyệt: giao diện hiển thị trạng thái rỗng."],
        ["Hậu điều kiện", "Hồ sơ chuyển sang trạng thái đã duyệt hoặc đã từ chối; thời "
                          "điểm cập nhật được ghi nhận; số liệu thống kê và bộ đếm trên "
                          "giao diện phản ánh trạng thái mới."],
        ["Ràng buộc nghiệp vụ", "Hệ thống không tự động ra quyết định xử phạt. Trạng thái "
                                 "duyệt chỉ xác nhận hồ sơ có đủ căn cứ, việc lập biên bản "
                                 "thuộc quy trình nghiệp vụ bên ngoài."],
    ], "Bảng 3.5. Đặc tả UC-04 Duyệt hồ sơ vi phạm"),

    H3("3.3.5. UC-05: Hỏi dữ liệu bằng tiếng Việt"),
    TB(["Trường", "Nội dung"], [
        ["Mã use case", "UC-05"],
        ["Tên", "Hỏi dữ liệu bằng tiếng Việt"],
        ["Actor", "Officer, Operator hoặc Administrator (actor chính); Central Server và "
                "dịch vụ mô hình ngôn ngữ bên ngoài (actor phụ)"],
        ["Mô tả tóm tắt", "Người dùng đặt câu hỏi tự nhiên bằng tiếng Việt; hệ thống sinh "
                          "câu lệnh SQL, kiểm tra an toàn, thực thi và trả kết quả kèm "
                          "gợi ý loại biểu đồ."],
        ["Tiền điều kiện", "Người dùng đã đăng nhập; khoá API của dịch vụ mô hình ngôn "
                           "ngữ đã được cấu hình; CSDL có dữ liệu."],
        ["Luồng sự kiện chính",
         "1. Người dùng mở bảng trợ lý và nhập câu hỏi, tối đa năm trăm ký tự.\n"
         "2. Giao diện gửi câu hỏi tới bộ định tuyến phía máy chủ của ứng dụng web.\n"
         "3. Bộ định tuyến chuyển tiếp tới trung tâm kèm JWT.\n"
         "4. Trung tâm dựng lời nhắc gồm mô tả lược đồ bốn bảng và yêu cầu chỉ viết câu "
         "lệnh SELECT.\n"
         "5. Trung tâm gọi dịch vụ mô hình ngôn ngữ qua giao thức REST.\n"
         "6. Dịch vụ trả về câu lệnh SQL và gợi ý loại biểu đồ.\n"
         "7. Trung tâm kiểm tra an toàn: chỉ chấp nhận SELECT, chặn các từ khoá nguy "
         "hiểm, tự bổ sung giới hạn năm mươi dòng.\n"
         "8. Trung tâm thực thi truy vấn chỉ đọc và trả về tên cột, các dòng dữ liệu, số "
         "lượng và loại biểu đồ.\n"
         "9. Giao diện hiển thị câu lệnh SQL đã sinh cùng bảng kết quả hoặc biểu đồ cột."],
        ["Luồng thay thế",
         "7a. Câu lệnh chứa từ khoá bị chặn: trung tâm trả thông báo lỗi, không thực thi, "
         "giao diện hiển thị lý do.\n"
         "5a. Dịch vụ mô hình ngôn ngữ không phản hồi hoặc khoá API sai: trả lỗi, giao "
         "diện gợi ý kiểm tra cấu hình.\n"
         "8a. Câu lệnh hợp lệ nhưng sai ngữ pháp hoặc tham chiếu bảng không tồn tại: trả "
         "lỗi truy vấn cho người dùng, không làm lộ chi tiết lược đồ.\n"
         "1a. Câu hỏi vượt giới hạn độ dài: giao diện chặn ngay phía client."],
        ["Hậu điều kiện", "Người dùng nhận được câu trả lời dưới dạng dữ liệu; không có "
                          "thay đổi nào trong CSDL."],
        ["Ràng buộc an toàn", "Toàn bộ câu lệnh sinh ra chỉ được phép đọc. Danh sách từ "
                              "khoá bị chặn gồm DROP, DELETE, INSERT, UPDATE, ALTER, "
                              "TRUNCATE, GRANT và REVOKE. Câu lệnh nhiều lệnh bị từ chối."],
    ], "Bảng 3.6. Đặc tả UC-05 Hỏi dữ liệu bằng tiếng Việt"),

    H2("3.4. Activity Diagram"),
    H3("3.4.1. Hoạt động xét vi phạm tại node biên"),
    P("Sơ đồ hoạt động ở Hình 3.6 mô tả vòng lặp xử lý mỗi khung hình. Điểm thiết kế đáng "
      "chú ý là trình tự các khâu: trạng thái đèn được xác định và ổn định hoá trước khi "
      "xét đến phương tiện. Cách sắp xếp này cho phép bỏ qua sớm toàn bộ phần logic vi "
      "phạm khi đèn không ở trạng thái đỏ ổn định, giảm chi phí tính toán không cần thiết "
      "ở phần lớn khung hình."),
    IMG(T + "/activity-detection-v2.png", 72,
        "Hình 3.6. Sơ đồ hoạt động xét vi phạm tại node biên"),
    P("Nhánh xử lý lỗi được đặt ở cấp toàn khối: bất kỳ khâu nào trong khung hình hiện "
      "tại gặp lỗi đều dẫn tới việc ghi nhật ký cảnh báo và chuyển sang khung hình kế "
      "tiếp. Đây là yêu cầu thực tế bắt buộc, vì một hệ thống giám sát chạy nhiều giờ "
      "liên tục không được phép dừng chỉ vì một khung hình hỏng hoặc một lần suy luận "
      "lỗi. Bộ theo dõi và bộ ổn định tín hiệu đều tự dung sai được khoảng trống ngắn "
      "trong dữ liệu."),

    H3("3.4.2. Hoạt động duyệt hồ sơ"),
    P("Sơ đồ ở Hình 3.7 mô tả luồng duyệt của Officer. Nhánh rẽ quan trọng nhất là trường "
      "hợp ảnh bằng chứng không tải được: thay vì chặn toàn bộ quy trình, hệ thống cho "
      "phép cán bộ căn cứ vào các trường dữ liệu còn lại. Quyết định này xuất phát từ "
      "thực tế vận hành — việc chờ ảnh có thể làm tồn đọng hồ sơ, trong khi các trường "
      "trạng thái đèn, độ tin cậy và toạ độ vẫn cung cấp căn cứ nhất định."),
    IMG(V + "/activity-review.png", 110,
        "Hình 3.7. Sơ đồ hoạt động duyệt hồ sơ vi phạm"),

    H3("3.4.3. Hoạt động hiệu chuẩn từ xa"),
    P("Sơ đồ ở Hình 3.8 mô tả luồng hiệu chuẩn, trong đó phần xử lý tại node được tách "
      "biệt với phần tương tác trên giao diện. Vòng lặp kiểm tra xác nhận ở cuối luồng là "
      "cần thiết vì lệnh hiệu chuẩn đi qua hai bước chuyển tiếp, và người vận hành cần "
      "biết chắc chắn node đã áp dụng giá trị mới."),
    IMG(V + "/activity-calibration.png", 66,
        "Hình 3.8. Sơ đồ hoạt động hiệu chuẩn từ xa"),

    H3("3.4.4. Máy trạng thái tín hiệu đèn"),
    P("Trạng thái đèn tín hiệu không được dùng trực tiếp từ kết quả phân loại từng khung "
      "hình mà phải đi qua một máy trạng thái có trễ. Lý do là kết quả phân loại trên dữ "
      "liệu thực tế luôn có nhiễu: một vài khung hình bị chói nắng, bị che khuất hoặc "
      "chuyển tiếp giữa hai màu có thể cho kết quả sai. Nếu dùng trực tiếp, một khung "
      "hình nhiễu duy nhất cũng đủ tạo ra hồ sơ vi phạm giả."),
    IMG(V + "/state-light.png", 135,
        "Hình 3.9. Máy trạng thái tín hiệu đèn có debounce"),
    P("Cơ chế hoạt động như sau. Một trạng thái chỉ được công nhận là ổn định sau khi "
      "cùng một kết quả có độ tin cậy đạt ngưỡng xuất hiện liên tiếp đủ số khung hình quy "
      "định. Khi đã ở trạng thái ổn định, việc chuyển sang trạng thái khác đòi hỏi số "
      "khung hình liên tiếp cao hơn — đây chính là phần trễ tạo nên tính hai chiều của "
      "cơ chế: dễ xác lập trạng thái ban đầu nhưng khó bị lật ngược bởi nhiễu. Các kết "
      "quả không đạt độ tin cậy không xoá ngay trạng thái ổn định mà chỉ được đếm dồn; "
      "vượt ngưỡng dung sai thì máy trạng thái mới quay về trạng thái không xác định."),
    P("Một cải tiến quan trọng là các ngưỡng này được quy đổi từ giây sang khung hình "
      "theo tốc độ khung hình thực của nguồn, thay vì dùng số khung hình cố định. Với "
      "ngưỡng cứng, cùng một giá trị sẽ tương ứng 0,23 giây ở nguồn 30 khung hình mỗi "
      "giây nhưng tới 2,3 giây ở nguồn 3 khung hình — nghĩa là hành vi hệ thống phụ "
      "thuộc vào thiết bị ghi hình theo cách không kiểm soát được. Sau khi quy đổi theo "
      "giây, độ trễ chuyển trạng thái đo được là 0,66 giây ở nguồn 6 khung hình và 0,67 "
      "giây ở nguồn 3 khung hình, so với 1,00 giây và 2,3 giây trước khi thay đổi."),

    H3("3.4.5. Máy trạng thái vòng đời hồ sơ vi phạm"),
    P("Hồ sơ vi phạm có vòng đời ba trạng thái, được lưu trong trường status của CSDL "
      "trung tâm."),
    IMG(T + "/state-violation-v2.png", 115,
        "Hình 3.10. Máy trạng thái vòng đời hồ sơ vi phạm"),
    P("Trạng thái chờ duyệt là trạng thái khởi tạo bắt buộc do node biên gán; node không "
      "có quyền tạo hồ sơ ở trạng thái đã duyệt. Hai trạng thái đã duyệt và đã từ chối "
      "chỉ đạt được qua thao tác của người dùng có thẩm quyền. Việc cho phép chuyển "
      "ngược từ hai trạng thái kết quả về trạng thái chờ là cần thiết trong thực tế vận "
      "hành, vì cán bộ có thể quyết định sai khi chưa xem đủ bằng chứng; tuy nhiên thao "
      "tác này được giới hạn ở vai trò phù hợp và thời điểm cập nhật luôn được ghi lại để "
      "truy vết."),

    H2("3.5. Sequence Diagram"),
    P("Bốn sơ đồ sequence dưới đây mô tả các luồng tương tác quan trọng nhất giữa các "
      "tầng, được chọn vì mỗi luồng chứa một quyết định thiết kế then chốt."),

    H3("3.5.1. Luồng phân phối hồ sơ bền vững"),
    P("Sơ đồ ở Hình 3.11 thể hiện cơ chế giao hàng hai bước. Điểm mấu chốt nằm ở thứ tự: "
      "hồ sơ được ghi vào kho SQLite tại node ngay tại thời điểm phát hiện, trước khi bất "
      "kỳ thao tác mạng nào diễn ra. Nhờ đó, khoảng thời gian rủi ro mất dữ liệu được "
      "thu về bằng không — nếu mạng hỏng hoặc node bị tắt, hồ sơ vẫn nằm trên đĩa và sẽ "
      "được gửi ở lần chạy kế tiếp."),
    IMG(T + "/sequence-delivery.png", 148,
        "Hình 3.11. Sơ đồ sequence phân phối hồ sơ bền vững"),
    P("Tính idempotent được bảo đảm ở cả hai đầu. Tại node, trường mã định danh trong kho "
      "outbox có ràng buộc duy nhất nên một sự kiện không thể được ghi hai lần. Tại trung "
      "tâm, trước khi lưu mỗi bản ghi hệ thống kiểm tra sự tồn tại của mã định danh bằng "
      "một truy vấn boolean, và bản ghi trùng được đếm riêng thay vì gây lỗi cho cả lô. "
      "Nhờ đó việc gửi lại sau một lần thất bại — tình huống không thể tránh khỏi trong "
      "mạng không tin cậy — luôn an toàn."),
    P("Ảnh bằng chứng được tải ở bước thứ hai, sau khi dữ liệu JSON đã được chấp nhận. "
      "Thứ tự này bảo đảm không bao giờ xuất hiện ảnh mồ côi trong kho đối tượng mà "
      "thiếu bản ghi tương ứng trong CSDL, đồng thời cho phép theo dõi riêng cờ trạng "
      "thái tải ảnh để thử lại độc lập."),

    H3("3.5.2. Luồng xác thực và làm mới token"),
    P("Sơ đồ ở Hình 3.12 mô tả cơ chế xác thực hai loại token. Token truy cập dạng JWT "
      "ký bằng HS256 với thời hạn mười hai giờ, tương đương một ca trực, mang theo tên "
      "người dùng và vai trò để trung tâm phân quyền mà không cần truy vấn CSDL ở mỗi yêu "
      "cầu. Token làm mới là chuỗi ngẫu nhiên 32 byte, chỉ được lưu dưới dạng giá trị băm "
      "SHA-256 trong CSDL nên ngay cả khi CSDL bị lộ thì token thật vẫn không thể khôi "
      "phục."),
    IMG(T + "/sequence-auth-v2.png", 125,
        "Hình 3.12. Sơ đồ sequence xác thực và làm mới token"),
    P("Mỗi lần làm mới, token cũ bị thu hồi và một token mới được cấp — cơ chế xoay vòng. "
      "Nếu một token đã bị thu hồi lại được trình bày lần nữa, hệ thống coi đó là dấu "
      "hiệu bị đánh cắp và thu hồi toàn bộ token làm mới của người dùng đó, buộc phải "
      "đăng nhập lại. Đây là biện pháp giảm thiểu thiệt hại tiêu chuẩn cho kịch bản token "
      "rò rỉ."),

    H3("3.5.3. Luồng hiệu chuẩn từ xa"),
    P("Sơ đồ ở Hình 3.13 thể hiện đường đi của một lệnh hiệu chuẩn, trong đó có hai lần "
      "xác thực độc lập: JWT của người dùng ở chặng web tới trung tâm, và token điều "
      "khiển của node ở chặng trung tâm tới node. Lệnh không đi thẳng từ trình duyệt tới "
      "node, nhờ đó địa chỉ nội bộ và token của node không bao giờ lộ ra phía client."),
    IMG(T + "/sequence-calibration-v2.png", 152,
        "Hình 3.13. Sơ đồ sequence hiệu chuẩn vạch dừng từ xa"),

    H3("3.5.4. Luồng trợ lý hỏi dữ liệu"),
    P("Sơ đồ ở Hình 3.14 mô tả luồng hỏi dữ liệu bằng tiếng Việt. Rào chắn an toàn nằm ở "
      "trung tâm chứ không phải ở phía giao diện, vì lời nhắc gửi cho mô hình ngôn ngữ "
      "chỉ là yêu cầu chứ không phải ràng buộc kỹ thuật — mô hình vẫn có thể sinh ra câu "
      "lệnh ngoài ý muốn. Do đó mọi câu lệnh đều phải qua bước kiểm tra trước khi thực "
      "thi."),
    IMG(T + "/sequence-ai-v2.png", 148,
        "Hình 3.14. Sơ đồ sequence trợ lý hỏi dữ liệu bằng tiếng Việt"),
    PB,
]
