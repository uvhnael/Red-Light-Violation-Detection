# -*- coding: utf-8 -*-
"""Nội dung báo cáo RLVD — phần 1: trang bìa, mục lục, Chương 1–3."""

ASSETS = "/home/uvhnael/projects/Red-Light-Violation-Detection/docs/report-assets"

BLOCKS_1 = [
    # ================= TRANG BÌA =================
    {"type": "paragraph", "style": "TitleBig", "text": "BÁO CÁO ĐỒ ÁN"},
    {"type": "paragraph", "style": "TitleSub", "text": "HỆ THỐNG PHÁT HIỆN VI PHẠM VƯỢT ĐÈN ĐỎ"},
    {"type": "paragraph", "style": "TitleSub2", "text": "(Red-Light Violation Detection — RLVD)"},
    {"type": "paragraph", "style": "TitleSub2", "text": "Ứng dụng Thị giác Máy tính, Deep Learning và Kiến trúc Edge — Central — Web"},
    {"type": "paragraph", "style": "TitleMeta", "text": "Tháng 8, năm 2026"},
    {"type": "page_break"},

    # ================= MỤC LỤC =================
    {"type": "heading", "text": "MỤC LỤC", "level": 1},
    {"type": "toc"},
    {"type": "page_break"},

    # ================= CHƯƠNG 1 =================
    {"type": "heading", "text": "CHƯƠNG 1. GIỚI THIỆU", "level": 1},

    {"type": "heading", "text": "1.1. Lý do chọn đề tài", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Vi phạm đèn đỏ là một trong những hành vi vi phạm giao thông phổ biến và nguy hiểm nhất tại Việt Nam, trực tiếp gây ra các vụ tai nạn nghiêm trọng tại nút giao, đặc biệt tại các ngã tư đông đục ở khu vực đô thị. Công tác phát hiện, xử lý hành vi này hiện vẫn phụ thuộc nhiều vào lực lượng Cảnh sát giao thông trực tiếp tại hiện trường hoặc xem lại đoạn video hình ảnh camera thủ công — cách làm tốn nhiều nhân lực, chỉ phủ được một tỷ lệ nhỏ số lượng vi phạm thực tế và khó đảm bảo tính minh bạch, khách quan khi lập hồ sơ xử phạt."},
    {"type": "paragraph", "style": "Body", "text": "Trong những năm gần đây, thị giác máy tính (Computer Vision) với các mô hình deep learning thế hệ mới như họ YOLO đã đạt độ chính xác và tốc độ xử lý đủ để vận hành thời gian thực trên phần cứng phổ thông. Đồng thời, kiến trúc tính toán biên (edge computing) cho phép phân tích video ngay tại vị trí camera, giảm chi phí băng thông và giúp hệ thống mở rộng theo số lượng nút giao một cách linh hoạt. Đây là cơ sở kỹ thuật để tự động hoá việc giám sát vi phạm vượt đèn đỏ một cách bền vững."},
    {"type": "paragraph", "style": "Body", "text": "Tuy nhiên, một hệ thống giám sát vi phạm thực thụ không dừng ở bài toán nhận diện: nó cần khả năng đọc biển số xe (OCR) theo chuẩn biển Việt Nam, cơ chế truyền hồ sơ đáng tin cậy khi mạng không ổn định, quy trình duyệt hồ sơ có con người tham gia (human-in-the-loop) để đảm bảo tính pháp lý, và giao diện quản trị giúp điều phối nhiều camera cùng lúc. Việc tích hợp tất cả các thành phần trên thành một hệ thống phân tán hoàn chỉnh là bài toán thực tiễn, có giá trị ứng dụng và học thuật cao."},
    {"type": "paragraph", "style": "Body", "text": "Xuất phát từ những lý do trên, nhóm lựa chọn đề tài “Xây dựng hệ thống phát hiện vi phạm vượt đèn đỏ đa tầng Edge — Central — Web” (RLVD), trong đó node biên chạy pipeline YOLO theo dõi phương tiện tại camera, máy chủ trung tâm lưu trữ và cung cấp API kèm trợ lý AI hỏi dữ liệu bằng tiếng Việt, và dashboard web phục vụ giám sát, hiệu chuẩn (calibration) từ xa lẫn quy trình duyệt vi phạm."},

    {"type": "heading", "text": "1.2. Mục tiêu", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Mục tiêu tổng quát: xây dựng một hệ thống phần mềm phân tán, tự động phát hiện phương tiện vượt đèn đỏ từ luồng video camera, tạo hồ sơ vi phạm kèm ảnh bằng chứng và biển số xe, phục vụ nghiệp vụ tra cứu, thống kê và duyệt xử lý."},
    {"type": "paragraph", "style": "Body", "text": "Các mục tiêu cụ thể của đề tài:"},
    {"type": "bullet_list", "items": [
        "Xây dựng pipeline thị giác máy tính thời gian thực tại node biên: phát hiện phương tiện (YOLO26m fine-tune), theo dõi đa đối tượng (ByteTrack), nhận diện trạng thái đèn tín hiệu (YOLO26n-cls kết hợp fusion HSV và vị trí bóng đèn), và logic xét vi phạm theo vạch dừng + hướng di chuyển.",
        "Huấn luyện và đánh giá ba mô hình deep learning chuyên biệt: phát hiện phương tiện (4 lớp), phân loại màu đèn giao thông (3 lớp), phát hiện biển số xe; benchmark so sánh với mô hình gốc COCO.",
        "Đọc và chuẩn hoá biển số Việt Nam bằng OCR (fast-plate-ocr) kèm bộ kiểm tra cấu trúc biển theo Thông tư 24/2023/TT-BGTVT.",
        "Thiết kế cơ chế truyền hồ sơ bền vững: durable outbox SQLite + batch sender, chịu được mất mạng và restart, chống trùng lặp theo event_id.",
        "Xây dựng backend trung tâm Spring Boot (REST API, PostgreSQL, MinIO) và dashboard web Next.js: thống kê, duyệt vi phạm, xem camera live (HLS), hiệu chuẩn vạch dừng — vùng đèn — hướng giám sát từ xa.",
        "Tích hợp trợ lý AI Text-to-SQL cho phép hỏi dữ liệu vi phạm bằng tiếng Việt, có lớp kiểm tra an toàn chỉ cho phép truy vấn SELECT.",
        "Kiểm thử toàn diện: unit test cho logic vi phạm, kiểm thử tích hợp API, kiểm thử end-to-end trên video thực tế và demo vận hành full stack bằng Docker.",
    ]},

    {"type": "heading", "text": "1.3. Đối tượng", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Đối tượng nghiên cứu của đề tài bao gồm:"},
    {"type": "bullet_list", "items": [
        "Phương tiện giao thông đường bộ được phát hiện bởi mô hình YOLO26m fine-tune gồm 4 lớp: ô tô con (car), xe mô tô/xe gắn máy (bike), xe van/xe khách (van/bus) và xe tải (truck).",
        "Tín hiệu đèn giao thông ba màu (đỏ/vàng/xanh) tại nút giao có camera cố định giám sát một hướng/làn cố định.",
        "Biển số xe cơ giới Việt Nam (ô tô và mô tô), bao gồm biển một dãy dạng NN-XXXXXXX với mã tỉnh, ký hiệu seri và số đăng ký.",
        "Luồng video đầu vào dạng tệp (MP4) hoặc chuẩn RTSP từ camera quan sát; trong phạm vi đồ án, video thực tế ghi tại các ngã tư Việt Nam được dùng làm nguồn đầu vào chính.",
    ]},

    {"type": "heading", "text": "1.4. Phạm vi", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Đề tài giới hạn trong các phạm vi sau:"},
    {"type": "bullet_list", "items": [
        "Hệ thống được xây dựng và kiểm thử trên một nút biên (edge node) chạy GPU NVIDIA RTX 2060, một máy chủ trung tâm và một dashboard web; kiến trúc cho phép mở rộng thêm node nhưng chưa triển khai nhiều node đồng thời trong demo.",
        "Vùng giám sát: một camera quan sát một hướng di chuyển tại nút giao; hướng giám sát được hiệu chuẩn để chỉ tính phương tiện đi đúng hướng (đường hai chiều).",
        "Loại vi phạm được xét: vượt vạch dừng khi đèn đỏ đã ổn định. Các loại vi phạm khác (chạy lấn làn, ngược chiều, chở người trên xe máy…) nằm ngoài phạm vi hiện tại, được đưa vào hướng phát triển.",
        "Hồ sơ vi phạm do hệ thống tạo ra mang tính hỗ trợ nghiệp vụ: quyết định xử phạt cuối cùng thuộc về cán bộ duyệt (human-in-the-loop); đồ án không phải là sản phẩm pháp lý hoàn chỉnh.",
        "Điều kiện ánh sáng, thời tiết và chất lượng video được đánh giá trên các bộ video thực tế có sẵn; chưa kiểm chứng trên toàn bộ các điều kiện vận hành thực địa (mưa, đêm, sương mù).",
    ]},

    {"type": "heading", "text": "1.5. Phương pháp thực hiện", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Đồ án được thực hiện theo quy trình phát triển phần mềm kết hợp nghiên cứu thực nghiệm, gồm các phương pháp chính:"},
    {"type": "numbered_list", "items": [
        "Khảo sát hiện trạng: phân tích quy trình giám sát, xử lý vi phạm vượt đèn đỏ hiện nay; khảo sát các hệ thống và mã nguồn mở liên quan (pipeline CV tự động hoá, hệ thống phạt nguội thương mại) để rút ra yêu cầu và khoảng trống kỹ thuật.",
        "Phân tích — thiết kế: sử dụng phương pháp hướngActor/Use Case với các sơ đồ UML (Use Case, Activity, Sequence, Class), sơ đồ kiến trúc hệ thống, ERD và thiết kế CSDL, API theo chuẩn REST.",
        "Huấn luyện mô hình (thực nghiệm): fine-tune ba mô hình YOLO trên các tập dữ liệu phương tiện (Roboflow, 7.920 ảnh train / 1.980 ảnh valid), biển số (tự thu thập, ~1.145 ảnh validate) và đèn giao thông (LISA dataset, gộp 7 lớp về 3 lớp màu); đánh giá bằng precision/recall/mAP/FPS và benchmark đối chiếu mô hình gốc COCO.",
        "Cài đặt prototype: lập trình 3 tầng theo kiến trúc Edge — Central — Web (Python/FastAPI — Java Spring Boot — Next.js), đóng gói Docker Compose toàn bộ stack.",
        "Kiểm thử: unit test logic vi phạm (pytest), kiểm thử tích hợp API (curl/httpx), kiểm thử end-to-end bằng phát lại video có ground-truth, kiểm thử UI thủ công trên dashboard, và kiểm thử độ bền của cơ chế phân phối outbox (kịch bản offline → online).",
        "Đánh giá kết quả: tổng hợp số liệu đo được (chỉ số mô hình, kết quả test case, số liệu vận hành demo) để rút ra kết luận, hạn chế và hướng phát triển.",
    ]},

    # ================= CHƯƠNG 2 =================
    {"type": "heading", "text": "CHƯƠNG 2. KHẢO SÁT VÀ PHÂN TÍCH", "level": 1},

    {"type": "heading", "text": "2.1. Khảo sát hiện trạng", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Quy trình giám sát và xử lý vi phạm vượt đèn đỏ tại các nút giao ở Việt Nam hiện nay chủ yếu dựa vào ba phương thức:"},
    {"type": "bullet_list", "items": [
        "Lực lượng trực tiếp: Cảnh sát giao thông phân công đứng nút giao, quan sát và lập biên bản tại chỗ. Cách này đảm bảo tính răn đe trực tiếp nhưng tốn nhiều nhân lực, chỉ phủ được giờ cao điểm ở một số nút, và phụ thuộc đánh giá chủ quan của người trực.",
        "Xem lại video camera: nhân viên xem lại đoạn video hình ảnh từ hệ thống camera giám sát để truy tìm vi phạm khi có vụ việc. Việc xem thủ công tốn thời gian lớn, không thể mở rộng cho toàn bộ camera và toàn bộ khung giờ.",
        "Phạt nguội qua hệ thống camera cố định: các hệ thống thương mại (trong nước và quốc tế) nhận diện biển số tự động tại một số trục, tuy nhiên phần lớn vẫn cần thao tác xác nhận thủ công từng case, chi phí đầu tư cao và hệ thống đóng kín khó tuỳ biến theo nghiệp vụ.",
    ]},
    {"type": "paragraph", "style": "Body", "text": "Từ khảo sát, các vấn đề cốt lõi được xác định: (1) khối lượng vi phạm lớn trong khi nguồn lực giám sát có hạn; (2) hồ sơ bằng chứng cần tính chính xác, kèm ảnh rõ ràng để đảm bảo minh bạch khi xử phạt; (3) camera đã có sẵn ở nhiều nút giao nhưng thiếu lớp phân tích tự động tại chỗ; (4) hệ thống cần vận hành liên tục 24/7, chịu được mất mạng giữa camera và trung tâm. Đây là căn cứ trực tiếp để hình thành yêu cầu chức năng và phi chức năng ở các mục 2.3 và 2.4."},

    {"type": "heading", "text": "2.2. Các hệ thống liên quan", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Nhóm đã khảo sát các hệ thống, mã nguồn và công nghệ liên quan đến bài toán phát hiện vượt đèn đỏ:"},
    {"type": "bullet_list", "items": [
        "Fully-Automated-red-light-Violation-Detection (AhmadYahya97, GitHub): pipeline thị giác máy tính dùng ngưỡng hoá ảnh (adaptive threshold) tìm vạch dừng và đèn tín hiệu, xử lý đơn luồng trên một máy. Hệ thống này là tham chiếu thuật toán cho việc phát hiện vạch dừng, tuy nhiên không có kiến trúc phân tán, không đọc biển số và không có giao diện quản trị — nhóm kế thừa ý tưởng ngưỡng vạch nhưng thay bằng hiệu chuẩn thủ công có kiểm soát của người vận hành.",
        "Hệ thống phạt nguội thương mại (Hikvision, Dahua, các nhà cung cấp trong nước): nhận diện biển số (ANPR) ổn định, đã triển khai thực tế; nhược điểm là chi phí cao, đóng kín, khó tuỳ biến và thường không có trợ lý truy vấn dữ liệu bằng ngôn ngữ tự nhiên.",
        "ByteTrack (Zhang và cộng sự, ECCV 2022): thuật toán theo dõi đa đối tượng liên kết mọi hộp giới hạn kể cả hộp có độ tin cậy thấp — được nhóm sử dụng làm nền tảng tracking trong pipeline.",
        "fast-plate-ocr: thư viện OCR biển số nguồn mở hiệu năng cao (ONNX runtime, ~vài chục ms/ảnh) hỗ trợ nhiều khu vực biển số; nhóm tích hợp kèm bộ kiểm tra cấu trúc biển riêng cho biển Việt Nam.",
        "LISA Traffic Light Dataset (Møgelmose và cộng sự): tập dữ liệu đèn giao thông phổ biến dùng để huấn luyện bộ phân loại màu đèn trong đồ án.",
    ]},
    {"type": "paragraph", "style": "Body", "text": "So sánh với các hệ thống trên, RLVD chọn hướng kết hợp: pipeline AI tại biên trên phần cứng phổ thông (GPU RTX 2060), backend trung tâm mã nguồn mở (Spring Boot + PostgreSQL + MinIO), dashboard web tự xây, hiệu chuẩn từ xa qua web, và lớp AI truy vấn tiếng Việt — thay vì mua hệ thống đóng kín hay dựng một script CV đơn lẻ."},

    {"type": "heading", "text": "2.3. Functional Requirements (Yêu cầu chức năng)", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Từ kết quả khảo sát, hệ thống xác định các yêu cầu chức năng (FR) như sau:"},
    {"type": "table", "header": ["Mã", "Yêu cầu chức năng", "Độ ưu tiên"], "rows": [
        ["FR-01", "Node biên đọc luồng video (tệp MP4 hoặc RTSP), tự động phát hiện và theo dõi 4 lớp phương tiện (car, bike, van/bus, truck) với định danh track ổn định.", "Cao"],
        ["FR-02", "Node biên phân loại trạng thái đèn tín hiệu (đỏ/vàng/xanh) trên vùng đèn đã hiệu chuẩn, có cơ chế ổn định (debounce) chống nhấp nháy trạng thái.", "Cao"],
        ["FR-03", "Node biên cho phép hiệu chuẩn từ xa: vạch dừng, vùng đèn và hướng giám sát được vẽ trên web và có hiệu lực ngay, không cần khởi động lại pipeline.", "Cao"],
        ["FR-04", "Khi đèn đỏ ổn định và phương tiện cắt vạch dừng theo đúng hướng giám sát, hệ thống tạo hồ sơ vi phạm gồm: mã sự kiện, thời điểm, trạng thái đèn, toạ độ cắt vạch, hộp giới hạn phương tiện.", "Cao"],
        ["FR-05", "Hệ thống tự động chụp ảnh bằng chứng (ảnh toàn cảnh có khung vi phạm và ảnh crop biển số) kèm hồ sơ vi phạm.", "Cao"],
        ["FR-06", "Hệ thống phát hiện biển số trên phương tiện, OCR đọc ký tự, kiểm tra và chuẩn hoá cấu trúc biển số Việt Nam; nếu OCR tại frame vi phạm hỏng thì dùng kết quả tốt nhất đã lưu của cùng phương tiện.", "Cao"],
        ["FR-07", "Hồ sơ vi phạm được lưu vào outbox SQLite tại biên và đẩy theo lô (batch) lên trung tâm; đẩy lại tự động khi mạng/kết nối phục hồi; chống trùng lặp theo event_id.", "Cao"],
        ["FR-08", "Trung tâm lưu trữ metadata vi phạm vào PostgreSQL, ảnh/video bằng chứng vào MinIO, cung cấp REST API đầy đủ (CRUD, phân trang, thống kê, duyệt).", "Cao"],
        ["FR-09", "Dashboard hiển thị thống kê tổng quan (KPI, xu hướng theo giờ, theo node), danh sách và chi tiết vi phạm kèm ảnh bằng chứng.", "Cao"],
        ["FR-10", "Dashboard hỗ trợ quy trình duyệt: cán bộ duyệt hồ sơ pending, phê duyệt hoặc từ chối kèm trạng thái rõ ràng (human-in-the-loop).", "Cao"],
        ["FR-11", "Dashboard phát luồng camera trực tiếp (HLS) và cho phép xem snapshot từng camera.", "Trung bình"],
        ["FR-12", "Trợ lý AI cho phép hỏi dữ liệu vi phạm bằng tiếng Việt: sinh SQL an toàn (chỉ SELECT), thực thi trên PostgreSQL và hiển thị kết quả dạng bảng/biểu đồ.", "Trung bình"],
        ["FR-13", "Node biên tự đăng ký và gửi heartbeat lên trung tâm; dashboard hiển thị danh sách node, trạng thái online/offline.", "Trung bình"],
    ], "style": "Table Grid"},

    {"type": "heading", "text": "2.4. Non-functional Requirements (Yêu cầu phi chức năng)", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Các yêu cầu phi chức năng (NFR) của hệ thống:"},
    {"type": "table", "header": ["Mã", "Hạng mục", "Yêu cầu"], "rows": [
        ["NFR-01", "Hiệu năng", "Pipeline biên đạt thời gian thực: mô hình phương tiện chạy ≥ 25 FPS trên GPU RTX 2060 (đo thực tế 73,4 FPS FP16); phân loại đèn ~0,3 ms/ảnh; OCR biển số dưới 100 ms/ảnh."],
        ["NFR-02", "Độ chính xác", "Mô hình phương tiện mAP50 ≥ 0,90 (thực tế 0,944); phân loại đèn đạt ≥ 99% trên video thực tế sau fusion; hồ sơ vi phạm không có vehicle giả (kiểm chứng bằng video đối chứng)."],
        ["NFR-03", "Độ tin cậy", "Hồ sơ vi phạm không mất khi mất mạng hoặc khởi động lại node biên (durable outbox); batch delivery idempotent theo event_id; frame lỗi không làm sập pipeline (bỏ qua an toàn)."],
        ["NFR-04", "Khả năng mở rộng", "Kiến trúc cho phép thêm node biên mới (node tự đăng ký) và nâng cấp từng tầng độc lập; Web và Central stateless có thể chạy nhiều bản sao."],
        ["NFR-05", "Bảo mật", "Xác thực JWT HS256 (BCrypt password, TTL 12h) + phân quyền 3 vai trò ADMIN/OPERATOR/OFFICER trên trung tâm; node biên dùng ingest token riêng (X-Ingest-Token, so sánh constant-time) khi đẩy hồ sơ; endpoint ghi của edge yêu cầu X-Edge-Token + rate limit 30 req/phút/IP + CORS whitelist + security headers; AI chỉ nhận SQL SELECT; JWT_SECRET bắt buộc (fail-fast); secret tách biệt qua biến môi trường .env không commit."],
        ["NFR-06", "Khả năng vận hành", "Toàn bộ stack đóng gói Docker Compose, một lệnh khởi động; có health check từng tầng; múi giờ toàn hệ thống thống nhất Asia/Ho_Chi_Minh."],
        ["NFR-07", "Tính khả dụng", "Dashboard truy cập qua trình duyệt hiện đại; luồng HLS hỗ trợ độ trễ thấp (~4 giây/segment); UI tiếng Việt."],
        ["NFR-08", "Bảo trì", "Mã nguồn phân tầng rõ ràng (edge/central/web), README tiếng Việt đầy đủ cho từng tầng, cấu hình tập trung qua biến môi trường."],
    ], "style": "Table Grid"},

    # ================= CHƯƠNG 3 =================
    {"type": "heading", "text": "CHƯƠNG 3. PHÂN TÍCH HỆ THỐNG", "level": 1},

    {"type": "heading", "text": "3.1. Actors", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Hệ thống có hai tác nhân người dùng chính và hai tác nhân hệ thống:"},
    {"type": "table", "header": ["Actor", "Mô tả"], "rows": [
        ["Kỹ thuật viên vận hành (Operator)", "Người phụ trách hạ tầng camera và node biên: đăng ký node, hiệu chuẩn vạch dừng — vùng đèn — hướng giám sát trên dashboard, theo dõi trạng thái node, xem camera trực tiếp."],
        ["Cán bộ xử lý vi phạm (Officer)", "Người phụ trách nghiệp vụ: duyệt/từ chối hồ sơ vi phạm đang chờ, tra cứu hồ sơ theo biển số/thời gian, xem thống kê, hỏi dữ liệu qua trợ lý AI."],
        ["Edge Node (hệ thống con)", "Tiến trình Python đặt tại camera: chạy pipeline phát hiện — theo dõi — xét vi phạm — OCR, duy trì outbox và đẩy hồ sơ lên trung tâm; cung cấp control-plane API để nhận hiệu chuẩn."],
        ["Central Server (hệ thống con)", "Backend Spring Boot: nhận và lưu hồ sơ, lưu trữ media, cung cấp API cho web, proxy yêu cầu hiệu chuẩn xuống edge, thực thi truy vấn AI."],
    ], "style": "Table Grid"},

    {"type": "heading", "text": "3.2. Use Case Diagram", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Sơ đồ use case tổng quát của hệ thống được thể hiện ở Hình 3.1, trong đó Operator phụ trách nhóm UC vận hành — hiệu chuẩn — giám sát, Officer phụ trách nhóm UC nghiệp vụ xử lý vi phạm; Edge Node và Central Server tham gia với vai trò hệ thống con."},
    {"type": "image", "path": ASSETS + "/vn/use-case.png", "width_mm": 160},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.1. Sơ đồ use case tổng quát của hệ thống RLVD"},
    {"type": "paragraph", "style": "Body", "text": "Danh sách các use case chính:"},
    {"type": "table", "header": ["Mã", "Use case", "Actor chính"], "rows": [
        ["UC-01", "Đăng ký node, gửi heartbeat", "Operator, Edge Node"],
        ["UC-02", "Hiệu chuẩn vạch dừng, vùng đèn, hướng giám sát", "Operator"],
        ["UC-03", "Phát hiện xe vượt đèn đỏ (detect + track + tripwire)", "Edge Node"],
        ["UC-04", "Đọc biển số xe (OCR) + chuẩn hoá biển VN", "Edge Node"],
        ["UC-05", "Gửi hồ sơ vi phạm (outbox + batch idempotent)", "Edge Node, Central Server"],
        ["UC-06", "Xem camera trực tiếp (HLS) + snapshot", "Operator"],
        ["UC-07", "Duyệt / từ chối hồ sơ vi phạm", "Officer"],
        ["UC-08", "Tra cứu vi phạm và xem thống kê", "Officer"],
        ["UC-09", "Hỏi dữ liệu bằng AI (Text-to-SQL)", "Officer"],
        ["UC-10", "Quản lý node biên (trạng thái, cấu hình)", "Operator"],
    ], "style": "Table Grid"},
    {"type": "paragraph", "style": "Body", "text": "Từ góc nhìn từng actor, hai sơ đồ use case riêng biệt làm rõ phạm vi trách nhiệm: Officer thao tác nghiệp vụ hồ sơ (Hình 3.2), Operator phụ trách vận hành — hiệu chuẩn (Hình 3.3)."},
    {"type": "image", "path": ASSETS + "/vn/use-case-officer.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.2. Sơ đồ use case nghiệp vụ của cán bộ xử lý vi phạm (Officer)"},
    {"type": "image", "path": ASSETS + "/vn/use-case-operator.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.3. Sơ đồ use case vận hành của kỹ thuật viên (Operator)"},

    {"type": "heading", "text": "3.3. Use Case Specification", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Ba use case trọng yếu nhất được đặc tả chi tiết như sau."},

    {"type": "heading", "text": "3.3.1. Đặc tả UC-03: Phát hiện xe vượt đèn đỏ", "level": 3},
    {"type": "table", "header": ["Trường", "Nội dung"], "rows": [
        ["Actor", "Edge Node (pipeline tự động)"],
        ["Mô tả", "Xác định phương tiện vượt vạch dừng khi đèn tín hiệu đã đỏ ổn định, và tạo hồ sơ vi phạm kèm bằng chứng."],
        ["Tiền điều kiện", "Node biên đang chạy; vùng đèn đã được hiệu chuẩn (UC-02); vạch dừng đã được hiệu chuẩn (UC-02)."],
        ["Luồng chính", "1. Đọc frame từ nguồn video. 2. Phân loại trạng thái đèn trên vùng đèn; nếu đèn đỏ ổn định ≥ 3 frame liên tiếp → đánh dấu RED ổn định. 3. Phát hiện phương tiện bằng YOLO26m; gán track ID bằng ByteTrack. 4. Với mỗi track: xác định phía của tâm hộp giới hạn so với vạch dừng. 5. Khi tâm hộp cắt qua vạch theo đúng hướng giám sát → ghi nhận vi phạm một lần cho mỗi track. 6. Chụp ảnh toàn cảnh + crop biển số, gắn metadata (event_id, track_id, light_state, toạ độ). 7. Ghi hồ sơ vào outbox (UC-05)."],
        ["Luồng thay thế", "4a. Chưa có vạch dừng → không xét vi phạm, pipeline vẫn chạy detect + track. 5a. Phương tiện đi ngược hướng giám sát → bỏ qua. 5b. Phương tiện lấp lửng trong vùng deadband quanh vạch → neo trên điểm phía gần nhất để tránh sót vi phạm. 2a. Đèn không ổn định/không xác định → frame bị bỏ qua, không xét vi phạm."],
        ["Hậu điều kiện", "Mỗi phương tiện vi phạm có đúng một hồ sơ với event_id duy nhất (chống trùng giữa các lần chạy), lưu trong outbox chờ đẩy lên trung tâm."],
    ], "style": "Table Grid"},

    {"type": "heading", "text": "3.3.2. Đặc tả UC-02: Hiệu chuẩn từ xa", "level": 3},
    {"type": "table", "header": ["Trường", "Nội dung"], "rows": [
        ["Actor", "Operator (kỹ thuật viên vận hành)"],
        ["Mô tả", "Vẽ vạch dừng, chọn hướng giám sát và khoanh vùng đèn tín hiệu cho một camera trên dashboard web, có hiệu lực ngay tại node biên."],
        ["Tiền điều kiện", "Node đã đăng ký với trung tâm (UC-01) và đang phát luồng video."],
        ["Luồng chính", "1. Operator mở trang Node trên web. 2. Web tải ảnh snapshot frame hiện tại từ edge (qua proxy trung tâm). 3. Operator kéo chuột vẽ vạch dừng (2 điểm) và chọn hướng giám sát (any/positive_to_negative/negative_to_positive). 4. Web POST vạch xuống trung tâm; trung tâm proxy (kèm token) xuống edge; edge cập nhật tripwire đang hoạt động — hiệu lực ngay không cần restart. 5. Operator khoanh vùng đèn (2 điểm). 6. Tương tự, vùng đèn được cập nhật trực tiếp vào pipeline."],
        ["Luồng thay thế", "4a. Nếu vẽ lại vạch mới, hướng giám sát đặt lại 'any' và hệ thống nhắc vẽ mũi tên hướng. 6a. Nếu bỏ qua vùng đèn → trạng thái đèn trả về UNKNOWN mỗi frame, không xét vi phạm."],
        ["Hậu điều kiện", "Node có bộ hiệu chuẩn hợp lệ; pipeline bắt đầu/hoàn thiện việc xét vi phạm mà không gián đoạn luồng video."],
    ], "style": "Table Grid"},

    {"type": "heading", "text": "3.3.3. Đặc tả UC-07: Duyệt hồ sơ vi phạm", "level": 3},
    {"type": "table", "header": ["Trường", "Nội dung"], "rows": [
        ["Actor", "Officer (cán bộ xử lý vi phạm)"],
        ["Mô tả", "Xem danh sách hồ sơ đang chờ, kiểm tra bằng chứng và phê duyệt hoặc từ chối từng hồ sơ."],
        ["Tiền điều kiện", "Có hồ sơ vi phạm trạng thái pending trong trung tâm; Officer đã truy cập dashboard."],
        ["Luồng chính", "1. Officer mở trang Review (danh sách pending, tải theo lô 50 bản ghi, có prefetch). 2. Chọn hồ sơ → xem chi tiết: ảnh toàn cảnh, crop biển số, trạng thái đèn, confidence, metadata. 3. Nhấn Duyệt hoặc Từ chối → web PATCH /api/violations/{id}/status. 4. Trung tâm cập nhật trạng thái (approved/rejected). 5. Danh sách và thống kê (KPI, badge số pending) tự làm mới."],
        ["Luồng thay thế", "2a. Officer lọc theo biển số/trạng thái/node hoặc dùng phân trang. 3a. Ảnh bằng chứng không load → hiển thị trạng thái lỗi, hồ sơ vẫn duyệt được dựa trên metadata."],
        ["Hậu điều kiện", "Hồ sơ có trạng thái cuối; thống kê approval rate được cập nhật; không thay đổi/xoá dữ liệu gốc của node biên."],
    ], "style": "Table Grid"},

    {"type": "heading", "text": "3.4. Activity Diagram", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Ba luồng hoạt động quan trọng nhất của hệ thống được mô tả bằng activity diagram: luồng phát hiện vi phạm tại node biên (Hình 3.4), luồng hiệu chuẩn từ xa (Hình 3.5) và luồng duyệt hồ sơ của cán bộ (Hình 3.6)."},
    {"type": "image", "path": ASSETS + "/vn/activity-detection.png", "width_mm": 140},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.4. Activity diagram — luồng phát hiện vi phạm tại node biên"},
    {"type": "paragraph", "style": "Body", "text": "Luồng phát hiện lặp qua từng frame video: trạng thái đèn phải đỏ ổn định (≥ 3 frame) thì pipeline mới bước vào nhánh xét vi phạm; nếu chưa có vạch dừng, hệ thống vẫn detect + track nhưng không xét vi phạm; khi một track cắt vạch đúng hướng, hồ sơ được tạo kèm ảnh bằng chứng, OCR biển số và ghi vào outbox."},
    {"type": "image", "path": ASSETS + "/vn/activity-calibration.png", "width_mm": 135},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.5. Activity diagram — luồng hiệu chuẩn vạch dừng/vùng đèn từ xa"},
    {"type": "paragraph", "style": "Body", "text": "Luồng hiệu chuẩn bắt đầu từ node tự đăng ký; operator thao tác hoàn toàn trên web: vẽ vạch → chọn hướng → vẽ vùng đèn; mỗi bước được đẩy qua trung tâm xuống edge và áp dụng ngay vào pipeline đang chạy."},
    {"type": "image", "path": ASSETS + "/vn/activity-review.png", "width_mm": 135},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.6. Activity diagram — luồng duyệt hồ sơ vi phạm (human-in-the-loop)"},

    {"type": "heading", "text": "3.5. Sequence Diagram", "level": 2},
    {"type": "paragraph", "style": "Body", "text": "Bốn kịch bản tuần tự chính của hệ thống: (1) chuỗi phát hiện và phân phối hồ sơ vi phạm từ camera tới dashboard (Hình 3.7); (2) chuỗi hiệu chuẩn từ xa (Hình 3.8); (3) chuỗi trợ lý AI Text-to-SQL (Hình 3.9); (4) chuỗi phân phối bền vững khi mất mạng — khôi phục (Hình 3.10)."},
    {"type": "image", "path": ASSETS + "/vn/sequence-violation.png", "width_mm": 155},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.7. Sequence diagram — phát hiện vi phạm và phân phối hồ sơ end-to-end"},
    {"type": "paragraph", "style": "Body", "text": "Chuỗi phân phối: pipeline ghi hồ sơ + ảnh vào outbox SQLite ngay khi phát hiện; sender lấy lô pending, POST batch lên trung tâm (dedup theo event_id); ảnh bằng chứng upload multipart lên MinIO qua MediaController; hồ sơ gán media_url; dashboard kéo danh sách phân trang và cập nhật trạng thái khi cán bộ duyệt."},
    {"type": "image", "path": ASSETS + "/vn/sequence-calibration.png", "width_mm": 155},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.8. Sequence diagram — hiệu chuẩn vạch dừng/vùng đèn/hướng từ xa qua proxy trung tâm"},
    {"type": "paragraph", "style": "Body", "text": "Trong chuỗi hiệu chuẩn, mọi request từ web đều qua trung tâm: EdgeProxyService giải quyết địa chỉ node từ thông tin đăng ký và kèm token X-Edge-Token khi chuyển tiếp xuống FastAPI của edge; pipeline đọc lại tripwire/ROI đang hoạt động mỗi frame nên thay đổi có hiệu lực tức thời."},
    {"type": "image", "path": ASSETS + "/vn/sequence-ai.png", "width_mm": 150},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.9. Sequence diagram — trợ lý AI Text-to-SQL (web → Gemini → PostgreSQL)"},
    {"type": "paragraph", "style": "Body", "text": "Chuỗi AI: câu hỏi tiếng Việt đi qua proxy mỏng của Next.js sang AiQueryService; lời nhắc (prompt) kèm schema CSDL và yêu cầu chỉ sinh SELECT; SQL sinh ra phải vượt qua lớp kiểm tra an toàn (chặn DROP/INSERT/UPDATE…, tự LIMIT 50) mới được thực thi qua JdbcTemplate; kết quả kèm gợi ý dạng hiển thị (bảng/biểu đồ) trả về AISidebar."},
    {"type": "image", "path": ASSETS + "/vn/sequence-outbox.png", "width_mm": 155},
    {"type": "paragraph", "style": "CaptionV", "text": "Hình 3.10. Sequence diagram — phân phối bền vững: mất mạng, khôi phục, chống trùng lặp"},
    {"type": "paragraph", "style": "Body", "text": "Chuỗi bền vững thể hiện ba chế độ vận hành của outbox: bình thường (ghi → đẩy → đánh dấu sent), mất mạng (hồ sơ pending tích luỹ, sender thử lại theo chu kỳ) và khôi phục (đẩy toàn bộ pending, trung tâm dedup theo event_id). event_id chứa run_id duy nhất theo lần chạy nên node khởi động lại không bao giờ sinh id trùng."},
]
