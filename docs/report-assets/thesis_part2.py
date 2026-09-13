# -*- coding: utf-8 -*-
"""Đồ án RLVD — nội dung phần 2: Chương 4 (Thiết kế) và Chương 5 (Cài đặt).

Số liệu lấy từ repository ngày 12/9/2026:
  * edge_node/ 6.597 dòng Python, 30 tệp
  * central_server/ 3.525 dòng Java, 45 tệp
  * web/ 6.547 dòng TypeScript/TSX, 35 tệp
  * docker-compose.full.yml (5 dịch vụ, cổng host 3000/8002/8082/9010/9011)
  * .env.sample, settings.py, IngestPaths.java, edge_node/api/server.py
"""

from thesis_part1 import T, V, S, P, H1, H2, H3, UL, PL, TB, IMG, PB

BLOCKS_2 = [
    # ============================================================== CHƯƠNG 4
    H1("CHƯƠNG 4. THIẾT KẾ HỆ THỐNG"),
    P("Chương này trình bày các quyết định thiết kế của hệ thống: kiến trúc tổng thể, "
      "sơ đồ lớp, mô hình dữ liệu, thiết kế cơ sở dữ liệu, thiết kế API và thiết kế "
      "giao diện. Mỗi mục đều nêu lý do lựa chọn phương án và những phương án đã bị "
      "loại bỏ, để người đọc thấy được quá trình cân nhắc chứ không chỉ thấy kết quả "
      "cuối cùng."),

    # ------------------------------------------------------- 4.1 Architecture
    H2("4.1. System Architecture"),
    P("Hệ thống được tổ chức theo kiến trúc ba tầng với ranh giới triển khai rõ ràng: "
      "node biên đặt gần camera, máy chủ trung tâm đặt tại trung tâm dữ liệu, bảng "
      "điều khiển web chạy trên trình duyệt của người vận hành. Hình 4.1 thể hiện "
      "kiến trúc tổng thể cùng các luồng giao tiếp giữa ba tầng."),
    IMG(T + "/architecture-v2.png", 150, "Hình 4.1. Kiến trúc hệ thống ba tầng"),
    P("Ba nguyên tắc thiết kế chi phối toàn bộ hệ thống:"),
    UL([
        "Xử lý gần nguồn dữ liệu. Video không bao giờ rời khỏi node biên. Node chỉ gửi "
        "về trung tâm những hồ sơ vi phạm đã hoàn chỉnh, gồm một bản ghi JSON khoảng "
        "vài trăm byte và một ảnh bằng chứng định dạng JPEG. Một luồng video 1080p "
        "chiếm băng thông hàng megabit mỗi giây, trong khi một hồ sơ vi phạm chỉ vài "
        "trăm kilobyte và chỉ sinh ra khi có sự kiện, chênh lệch ba đến bốn bậc độ lớn.",
        "Trung tâm là nguồn sự thật duy nhất. Mọi trạng thái nghiệp vụ lâu dài (hồ sơ "
        "vi phạm, kết quả duyệt, danh bạ node, tài khoản người dùng) đều nằm tại trung "
        "tâm. Node biên chỉ giữ trạng thái tạm thời: cấu hình hiệu chuẩn đang áp dụng "
        "và kho outbox chờ gửi. Nhờ vậy khi một node mất hoàn toàn, dữ liệu nghiệp vụ "
        "không bị ảnh hưởng.",
        "Ranh giới tầng không bị xuyên thủng. Trình duyệt không gọi trực tiếp tới "
        "trung tâm hay node biên; mọi yêu cầu đi qua lớp proxy của Next.js. Tầng "
        "trình diễn không chứa thư viện cơ sở dữ liệu hay thư viện trí tuệ nhân tạo "
        "nào. Quy ước này được giữ bằng cả thiết kế lẫn kiểm tra tự động.",
    ]),
    P("Một điểm khác biệt so với kiến trúc ba tầng cổ điển là luồng gọi ngược chiều: "
      "trung tâm chủ động gọi xuống node biên khi người vận hành hiệu chuẩn từ xa. "
      "Lý do là node biên nằm sau mạng nội bộ của nút giao, không có địa chỉ công khai "
      "cố định, nên trình duyệt không thể gọi thẳng tới node một cách tin cậy. Trung "
      "tâm đóng vai trò cầu nối: nó lưu địa chỉ node do chính node đăng ký lên, sau đó "
      "chuyển tiếp yêu cầu kèm token dành riêng cho node. Cách làm này cho phép giữ "
      "toàn bộ cơ chế xác thực người dùng tại một nơi, đồng thời không phải mở cổng "
      "của node ra ngoài."),
    IMG(T + "/component-v2.png", 152,
        "Hình 4.2. Sơ đồ thành phần và giao diện giữa các tầng"),
    P("Hình 4.2 liệt kê các thành phần bên trong từng tầng và giao diện kết nối giữa "
      "chúng. Toàn bộ giao tiếp điều khiển dùng HTTP với định dạng JSON; riêng luồng "
      "video trực tiếp dùng HLS qua HTTP, là giao thức phát phân đoạn nên tương thích "
      "với trình duyệt mà không cần plugin. Điểm cần lưu ý trong thiết kế thành phần "
      "là node biên chứa hai tiến trình logic chạy song song: vòng lặp xử lý khung hình "
      "(chạy liên tục, ưu tiên thời gian thực) và luồng gửi dữ liệu (chạy nền, chịu "
      "được trễ). Hai luồng này chỉ gặp nhau tại kho outbox, nhờ vậy việc mạng chậm "
      "hoặc trung tâm tạm ngừng không thể làm vòng lặp xử lý bị khựng."),
    P("Kiến trúc trên được hiện thực hoá thành năm dịch vụ chạy trong cùng một mạng "
      "nội bộ ảo do Docker Compose quản lý, thể hiện ở Hình 4.3. Hai dịch vụ lưu trữ "
      "(PostgreSQL và MinIO) chỉ được các dịch vụ ứng dụng truy cập qua tên dịch vụ "
      "trong mạng nội bộ, không cần mở cổng ra ngoài. Node biên là dịch vụ duy nhất "
      "được cấp quyền truy cập bộ xử lý đồ hoạ, vì toàn bộ khối lượng suy luận học sâu "
      "nằm ở tầng này. Việc đóng gói toàn bộ hệ thống thành một tệp cấu hình duy nhất "
      "giúp tái lập môi trường chạy giống hệt nhau trên bất kỳ máy nào có Docker và "
      "trình điều khiển GPU, đây cũng là cách nhóm nghiệm thu kết quả đo ở Chương 6."),
    IMG(T + "/deployment-v2.png", 150,
        "Hình 4.3. Sơ đồ triển khai bằng Docker Compose với năm dịch vụ"),

    # ------------------------------------------------------- 4.2 Class Diagram
    H2("4.2. Class Diagram"),
    P("Sơ đồ lớp ở Hình 4.4 phản ánh đúng cấu trúc mã nguồn hiện hành, chia theo ba "
      "gói tương ứng ba tầng triển khai."),
    IMG(T + "/class-diagram-v2.png", 165, "Hình 4.4. Sơ đồ lớp của hệ thống"),
    P("Thiết kế lớp của node biên dựa trên ba mẫu chính:"),
    UL([
        "Giao thức (Protocol) thay cho kế thừa. Các thành phần phát hiện, theo dõi, "
        "phân loại đèn và đọc biển số đều được khai báo dưới dạng giao thức trong "
        "edge_node/core/contracts.py, gồm ObjectDetector, MultiObjectTracker, "
        "TrafficLightClassifier và PlateRecognizer. Pipeline chỉ phụ thuộc vào giao "
        "thức, không phụ thuộc vào lớp cụ thể, nên có thể thay mô hình phát hiện "
        "hoặc thay thuật toán theo dõi mà không sửa logic xét vi phạm. Đây cũng là lý "
        "do 124 kiểm thử đơn vị chạy được trên máy không có card đồ hoạ.",
        "Đối tượng giá trị bất biến. Detection, Track, BoundingBox, Point, "
        "LightObservation và ViolationEvent đều là dataclass đóng băng. Dữ liệu đi qua "
        "nhiều tầng xử lý trong một khung hình mà không bị thay đổi ngầm, giúp suy "
        "diễn nguyên nhân lỗi dễ dàng hơn.",
        "Tách cấu hình khỏi đối tượng nghiệp vụ. Tham số nằm trong TripwireConfig, "
        "RedStabilizerConfig và ViolationConfig; trạng thái hiệu chuẩn đang áp dụng "
        "nằm trong biến toàn cục có khoá bảo vệ, đọc lại ở mỗi khung hình. Nhờ đó "
        "yêu cầu hiệu chuẩn có hiệu lực tức thời mà không cần khởi động lại pipeline.",
    ]),
    P("Lớp RedLightViolationPipeline giữ vai trò điều phối. Mỗi khung hình đi qua bảy "
      "bước có thứ tự cố định: phân loại đèn, làm ổn định trạng thái đèn, phát hiện "
      "phương tiện, cập nhật theo dõi, gắn biển số, xét vi phạm và ghi kho outbox. "
      "Thứ tự này không đổi được, vì bước xét vi phạm cần đồng thời track đã xác nhận "
      "và tín hiệu đèn đã ổn định."),
    P("Phía máy chủ trung tâm áp dụng phân lớp chuẩn của Spring: Controller nhận yêu "
      "cầu và kiểm tra dữ liệu đầu vào, Service chứa quy tắc nghiệp vụ, Repository "
      "truy cập dữ liệu qua Spring Data JPA, Entity ánh xạ bảng và DTO chuyển dữ liệu "
      "ra ngoài. Ba service đáng chú ý về mặt thiết kế: ViolationService xử lý nhận "
      "dữ liệu theo lô với đặc tính idempotent; ViolationStatsService tách riêng phần "
      "thống kê để thay đổi hình dạng số liệu tổng hợp không ảnh hưởng tới luồng "
      "nhận dữ liệu; EdgeProxyService đảm nhận việc gọi ngược xuống node và cố định "
      "giao thức HTTP/1.1."),
    P("Phía web không dùng sơ đồ lớp theo nghĩa truyền thống vì Next.js tổ chức theo "
      "trang và thành phần. Phần logic tái sử dụng nằm trong bốn mô-đun ở thư mục "
      "lib: api.ts bọc toàn bộ lời gọi mạng kèm cơ chế tự làm mới token, auth.ts quản "
      "lý phiên, types.ts định nghĩa kiểu dữ liệu phản chiếu DTO của trung tâm, và "
      "nodes.ts chuẩn hoá trạng thái node. Thiết kế này giữ cho các trang chỉ lo "
      "trình bày."),

    # ------------------------------------------------------- 4.3 ERD
    H2("4.3. ERD"),
    P("Cơ sở dữ liệu của máy chủ trung tâm có bốn thực thể, thể hiện ở Hình 4.5. "
      "Quan hệ giữa chúng phản ánh đúng luồng nghiệp vụ: một người dùng có nhiều "
      "phiên đăng nhập, một node biên sinh ra nhiều hồ sơ vi phạm, mỗi hồ sơ vi phạm "
      "có nhiều nhất một ảnh bằng chứng."),
    IMG(T + "/erd-v2.png", 150,
        "Hình 4.5. Mô hình thực thể quan hệ của cơ sở dữ liệu rlvd_central"),
    P("Hai đặc điểm thiết kế cần giải thích:"),
    UL([
        "Quan hệ node biên và hồ sơ vi phạm là quan hệ logic, không có khoá ngoại vật "
        "lý. Trường node_id trong bảng violations là chuỗi, không trỏ tới khoá chính "
        "của bảng edge_nodes. Lý do là hồ sơ vi phạm phải được chấp nhận ngay cả khi "
        "node chưa kịp đăng ký hoặc khi bảng node bị làm sạch để kiểm thử; nếu ràng "
        "buộc khoá ngoại, một sự cố đăng ký sẽ khiến toàn bộ lô dữ liệu bị từ chối và "
        "node phải gửi lại. Đổi lại, việc đối soát được bảo đảm bằng quy trình: node "
        "luôn đăng ký trước khi gửi dữ liệu và bộ đếm thống kê theo node_id vẫn hoạt "
        "động bình thường.",
        "Bảng refresh_tokens lưu giá trị băm SHA-256 chứ không lưu token gốc. Khi "
        "bảng này bị lộ, kẻ tấn công vẫn không thể dùng token để đăng nhập. Đây là "
        "cách xử lý tương tự cách lưu mật khẩu, áp dụng cho cả thông tin đăng nhập "
        "dài hạn.",
    ]),
    P("Ngoài bốn bảng của trung tâm, node biên có một kho SQLite riêng (xem mục 4.4) "
      "chỉ tồn tại trong phạm vi node và không bao giờ đồng bộ cấu trúc lên trung "
      "tâm. Việc tách riêng này là chủ ý: kho tạm không phải là một phần của mô hình "
      "dữ liệu nghiệp vụ, nó là hàng đợi giao hàng."),
    P("Hình 4.6 tổng hợp đường đi của dữ liệu qua ba tầng dưới dạng luồng biến đổi, "
      "bổ sung cho sơ đồ triển khai ở Hình 4.3: dữ liệu thô là khung hình video chỉ "
      "tồn tại ở node biên, dữ liệu có cấu trúc tồn tại ở trung tâm, còn dữ liệu hiển "
      "thị được tổng hợp tại tầng web. Mỗi mũi tên trên hình tương ứng với một điểm "
      "kiểm tra dữ liệu trong mã nguồn, nhờ vậy khi một hồ sơ hiển thị sai trên giao "
      "diện, có thể truy ngược theo hình để khoanh vùng tầng gây lỗi."),
    IMG(T + "/data-flow-v2.png", 120,
        "Hình 4.6. Luồng dữ liệu hồ sơ vi phạm qua ba tầng"),

    # ------------------------------------------------------- 4.4 Database Design
    H2("4.4. Database Design"),
    P("Bảng violations là bảng trung tâm của hệ thống. Thiết kế cột xuất phát trực "
      "tiếp từ yêu cầu truy vết: khi một hồ sơ bị nghi ngờ là báo động giả, người "
      "duyệt phải trả lời được câu hỏi hệ thống căn cứ vào đâu để kết luận. Vì vậy "
      "ngoài các trường mô tả sự kiện, bảng còn lưu cả dữ liệu hình học thô và trạng "
      "thái đèn tại thời điểm xảy ra vi phạm."),
    TB(["Cột", "Kiểu dữ liệu", "Ràng buộc và ý nghĩa"], [
        ["id", "BIGINT", "Khoá chính, tự tăng"],
        ["event_id", "VARCHAR", "Khoá duy nhất, có chỉ mục idx_event_id. Định dạng "
         "rlv-{mã phiên chạy}-{số khung hình}-{mã theo dõi}. Đây là nền tảng của cơ "
         "chế nhận dữ liệu idempotent"],
        ["node_id", "VARCHAR", "Có chỉ mục idx_node_id. Node gửi hồ sơ; dùng để lọc và "
         "thống kê theo node"],
        ["track_id, frame_index, timestamp_ms", "INTEGER, INTEGER, DOUBLE",
         "Vị trí của sự kiện trong luồng video, cho phép mở lại đúng khung hình để "
         "đối chiếu"],
        ["crossing_point_x, crossing_point_y", "DOUBLE",
         "Toạ độ điểm được dùng để xét cắt vạch tại thời điểm kết luận vi phạm"],
        ["previous_point_x, previous_point_y", "DOUBLE",
         "Toạ độ điểm neo liền trước, cùng với điểm cắt tạo thành đoạn thẳng dùng "
         "trong phép thử giao cắt"],
        ["bbox_x1, bbox_y1, bbox_x2, bbox_y2", "DOUBLE",
         "Hộp giới hạn của phương tiện lúc vi phạm"],
        ["previous_side, current_side", "INTEGER",
         "Hai phía của vạch dừng trước và sau khi cắt. Cặp giá trị này là bằng chứng "
         "hình học về hướng di chuyển"],
        ["light_state, light_confidence", "VARCHAR, DOUBLE",
         "Trạng thái đèn đã làm ổn định và độ tin cậy tại thời điểm xét vi phạm"],
        ["plate_text, plate_confidence", "VARCHAR, DOUBLE",
         "Biển số đã qua bộ kiểm tra cấu trúc; để trống nếu không đọc được biển hợp lệ"],
        ["status", "VARCHAR", "Có chỉ mục idx_status. Nhận một trong ba giá trị "
         "pending, approved, rejected. Mặc định pending"],
        ["media_url", "VARCHAR(1024)",
         "Khoá đối tượng trong MinIO theo mẫu violations/{event_id}/{uuid}.jpg; để "
         "trống khi chưa tải ảnh lên"],
        ["metadata", "TEXT", "Chuỗi JSON mở rộng: hướng giám sát, nguồn video, thông "
         "tin mô hình, nhãn phương tiện"],
        ["created_at, updated_at", "DATETIME",
         "Có chỉ mục idx_created_at. Mốc thời gian theo múi giờ Việt Nam; phục vụ sắp "
         "xếp và thống kê theo giờ"],
    ], "Bảng 4.1. Thiết kế bảng violations"),
    P("Bốn chỉ mục trên bảng violations được chọn theo đúng bốn dạng truy vấn mà hệ "
      "thống thực sự chạy: tra cứu theo mã sự kiện khi nhận dữ liệu, lọc theo trạng "
      "thái khi duyệt hồ sơ, lọc theo node khi xem chi tiết node, và sắp xếp theo "
      "thời gian khi liệt kê. Không có chỉ mục thừa, vì mỗi chỉ mục đều làm chậm "
      "thao tác ghi."),
    TB(["Cột", "Kiểu dữ liệu", "Ràng buộc và ý nghĩa"], [
        ["id", "UUID", "Khoá chính, sinh phía ứng dụng"],
        ["node_id", "VARCHAR", "Duy nhất, có chỉ mục idx_edge_node_node_id. Định danh "
         "node, dùng làm khoá upsert khi node đăng ký lại"],
        ["name", "VARCHAR", "Tên hiển thị trên giao diện; mặc định lấy bằng node_id"],
        ["ip_address", "VARCHAR", "Địa chỉ để trung tâm gọi ngược xuống node, đã tách "
         "bỏ giao thức và cổng"],
        ["status", "VARCHAR", "Có chỉ mục idx_edge_node_status. Trạng thái hoạt động "
         "của node"],
        ["last_ping", "DATETIME", "Có chỉ mục idx_edge_node_last_ping. Mốc heartbeat "
         "gần nhất; node được coi là trực tuyến nếu mốc này cách hiện tại dưới hai phút"],
        ["settings_json", "TEXT", "Toàn bộ cấu hình node gửi lên khi đăng ký, bao gồm "
         "cổng API điều khiển và token của node để trung tâm dùng khi gọi ngược"],
    ], "Bảng 4.2. Thiết kế bảng edge_nodes"),
    TB(["Bảng", "Cột chính", "Ràng buộc và ghi chú thiết kế"], [
        ["users",
         "id (khoá chính), username (duy nhất, tối đa 64 ký tự), password_hash (128 "
         "ký tự), full_name, role (16 ký tự), enabled, created_at, updated_at",
         "Chỉ mục duy nhất idx_users_username trên username. Mật khẩu băm bằng "
         "BCrypt. Trường role nhận ADMIN, OPERATOR hoặc OFFICER và là căn cứ duy nhất "
         "để phân quyền. Tài khoản quản trị đầu tiên được tạo từ biến môi trường, "
         "không ghi đè nếu đã tồn tại"],
        ["refresh_tokens",
         "id (khoá chính), user_id, token_hash (64 ký tự, duy nhất), expires_at, "
         "revoked_at, created_at, user_agent (255 ký tự)",
         "Hai chỉ mục: idx_refresh_tokens_user để thu hồi toàn bộ phiên của một người "
         "dùng, idx_refresh_tokens_hash để tra cứu khi làm mới token. Trường "
         "revoked_at khác rỗng nghĩa là token đã bị thu hồi; nếu một token đã thu hồi "
         "được dùng lại, hệ thống thu hồi toàn bộ token của người dùng đó"],
    ], "Bảng 4.3. Thiết kế bảng users và refresh_tokens"),
    P("Lược đồ cơ sở dữ liệu được Hibernate sinh tự động từ khai báo trong thực thể "
      "JPA, với chế độ ddl-auto ở mức update. Cách này phù hợp với quy mô đồ án vì "
      "lược đồ còn thay đổi trong quá trình phát triển, nhưng có nhược điểm là "
      "không kiểm soát được thứ tự thay đổi và không có lịch sử di chuyển. Mục 7.2 "
      "ghi nhận đây là hạn chế và đề xuất thay bằng công cụ di chuyển lược đồ có "
      "phiên bản khi hệ thống đi vào vận hành thật."),
    P("Kho outbox tại node biên có cấu trúc đơn giản hơn nhiều nhưng giữ vai trò then "
      "chốt trong thiết kế giao hàng:"),
    TB(["Cột", "Kiểu dữ liệu", "Ý nghĩa"], [
        ["id", "INTEGER", "Khoá chính tự tăng, dùng để lấy bản ghi theo thứ tự ghi"],
        ["event_id", "TEXT", "Duy nhất. Ràng buộc này chặn trùng ngay tại node, trước "
         "cả khi dữ liệu được gửi đi"],
        ["payload", "TEXT", "Chuỗi JSON của hồ sơ vi phạm, đúng định dạng API nhận dữ "
         "liệu của trung tâm"],
        ["image", "BLOB", "Ảnh bằng chứng dạng JPEG. Tách thành cột riêng để luồng gửi "
         "dữ liệu JSON theo lô không phải đọc ảnh"],
        ["status", "TEXT", "pending hoặc sent"],
        ["media_sent", "INTEGER", "Cờ riêng cho ảnh: 0 là chưa tải lên, 1 là đã xong. "
         "Tách khỏi status vì bản ghi JSON và ảnh có thể hoàn thành ở hai thời điểm "
         "khác nhau"],
        ["attempts", "INTEGER", "Số lần thử gửi, dùng làm cơ số cho chiến lược lùi "
         "theo hàm mũ"],
        ["created_at, sent_at", "TEXT", "Mốc ghi vào kho và mốc gửi thành công, theo "
         "giờ UTC"],
    ], "Bảng 4.4. Lược đồ kho outbox tại node biên (SQLite)"),
    P("Chỉ mục ghép idx_outbox_status trên cặp (status, id) phục vụ đúng câu truy vấn "
      "của luồng gửi dữ liệu: lấy các bản ghi chưa gửi theo thứ tự ghi vào, giới hạn "
      "số lượng mỗi đợt. Ảnh được lưu trong cùng tệp cơ sở dữ liệu để việc ghi hồ sơ "
      "và ghi ảnh là một thao tác nguyên tử; nếu tách ảnh ra thư mục riêng, một sự cố "
      "giữa hai lần ghi sẽ tạo ra hồ sơ không có ảnh hoặc ảnh mồ côi."),

    # ------------------------------------------------------- 4.5 API Design
    H2("4.5. API Design"),
    P("Hệ thống có hai bề mặt API riêng biệt: API của máy chủ trung tâm phục vụ bảng "
      "điều khiển web và nhận dữ liệu từ node, và API điều khiển của node biên phục "
      "việc xem camera và hiệu chuẩn. Tách hai bề mặt là cần thiết vì đối tượng gọi "
      "khác nhau và cơ chế xác thực khác nhau."),
    TB(["Quy ước", "Áp dụng trong hệ thống"], [
        ["Định danh tài nguyên bằng danh từ",
         "/api/violations, /api/v1/edge-nodes, /api/auth. Không có động từ trong đường "
         "dẫn, trừ hai trường hợp là hành vi nghiệp vụ thực sự: /status (đổi trạng "
         "thái duyệt) và /register (đăng ký node)"],
        ["Phương thức HTTP diễn tả hành động",
         "GET để đọc, POST để tạo hoặc thực hiện, PATCH/PUT để cập nhật, DELETE để "
         "xoá. Riêng thao tác đổi trạng thái duyệt có cả PATCH lẫn PUT cùng ý nghĩa "
         "để tương thích với các client cũ"],
        ["Phiên bản API trong đường dẫn",
         "Các endpoint nhận dữ liệu từ node có tiền tố /api/v1; endpoint phục vụ web "
         "giữ đường dẫn ngắn. Cả hai đều trỏ tới cùng một service"],
        ["Xác thực bằng tiêu đề, không dùng cookie phiên",
         "Người dùng web gửi Authorization kiểu Bearer; node biên gửi X-Ingest-Token; "
         "lời gọi ngược xuống node gửi X-Edge-Token"],
        ["Phân trang bằng tham số truy vấn",
         "page đánh số từ 0, size mặc định 20. Phản hồi gồm content, page, size, "
         "total_elements, total_pages, first, last để giao diện vẽ điều hướng mà "
         "không cần gọi thêm"],
        ["Lỗi trả về theo mã trạng thái chuẩn",
         "Không bọc lỗi trong phản hồi 200. Thông báo lỗi bằng tiếng Việt để hiển "
         "thị trực tiếp trên giao diện"],
    ], "Bảng 4.5. Quy ước thiết kế API"),
    P("Trước khi liệt kê endpoint, cần làm rõ các lớp bảo vệ mà mỗi yêu cầu phải đi "
      "qua, thể hiện ở Hình 4.7. Thiết kế API và thiết kế bảo mật trong hệ thống này "
      "là một: cùng một bảng endpoint nhưng mỗi nhóm đi qua một chuỗi bộ lọc khác "
      "nhau, và việc chọn sai nhóm sẽ khiến yêu cầu bị chặn hoặc, tệ hơn, được chấp "
      "nhận khi không nên."),
    IMG(T + "/security-layers-v2.png", 110,
        "Hình 4.7. Các lớp bảo vệ của hệ thống theo nhóm yêu cầu"),
    TB(["Nhóm", "Endpoint tiêu biểu", "Cơ chế bảo vệ"], [
        ["Xác thực người dùng",
         "POST /api/auth/login, POST /api/auth/refresh, POST /api/auth/logout, "
         "GET /api/auth/me",
         "login, refresh và logout mở công khai vì client chưa có token hợp lệ khi "
         "gọi; /me yêu cầu JWT"],
        ["Nhận hồ sơ vi phạm",
         "POST /api/violations, POST /api/violations/batch, POST /api/v1/violations, "
         "POST /api/v1/violations/batch",
         "X-Ingest-Token, so khớp theo thời gian hằng số"],
        ["Nhận ảnh bằng chứng",
         "POST /api/v1/violations/{eventId}/media (multipart)",
         "X-Ingest-Token"],
        ["Đăng ký node",
         "POST /api/v1/edge-nodes/register",
         "X-Ingest-Token"],
        ["Tra cứu hồ sơ",
         "GET /api/violations, GET /api/violations/page, GET /api/violations/counts, "
         "GET /api/violations/{id}, GET /api/violations/event/{eventId}",
         "JWT, mọi vai trò"],
        ["Thống kê",
         "GET /api/stats",
         "JWT, mọi vai trò"],
        ["Duyệt hồ sơ",
         "PATCH hoặc PUT /api/violations/{id}/status, DELETE /api/violations/{id}",
         "JWT với vai trò ADMIN hoặc OFFICER để duyệt; chỉ ADMIN được xoá"],
        ["Quản lý node",
         "GET /api/v1/edge-nodes, GET /api/v1/edge-nodes/{nodeId}, "
         "PUT /api/v1/edge-nodes/{nodeId}/settings",
         "JWT với vai trò ADMIN hoặc OPERATOR"],
        ["Hiệu chuẩn qua trung gian",
         "GET /api/v1/edge-nodes/{nodeId}/calibration, "
         "POST /api/v1/edge-nodes/{nodeId}/calibration/stop-line, "
         "POST .../calibration/light-roi, GET .../calibration/snapshot",
         "JWT ADMIN hoặc OPERATOR; trung tâm gắn thêm X-Edge-Token khi chuyển tiếp"],
        ["Ảnh bằng chứng",
         "GET /api/v1/violations/{eventId}/media/blob, GET .../media, DELETE .../media",
         "Luồng blob mở ở tầng bảo mật vì trung tâm đóng vai proxy đã xác thực; "
         "luồng xem và xoá metadata yêu cầu JWT"],
        ["Trợ lý trí tuệ nhân tạo",
         "POST /api/ai/query",
         "JWT mọi vai trò, kèm bộ kiểm tra chỉ cho phép câu lệnh SELECT"],
        ["Kiểm tra sức khoẻ",
         "GET /api/health, GET /actuator/health, GET /actuator/info",
         "Mở công khai để Docker healthcheck và bộ giám sát dùng"],
    ], "Bảng 4.6. Danh sách nhóm API của máy chủ trung tâm"),
    P("Tổng cộng máy chủ trung tâm có 31 phương thức xử lý yêu cầu phân bố trong năm "
      "bộ điều khiển: AuthController, ViolationController, EdgeNodeController, "
      "MediaController và AiQueryController. Danh sách đầy đủ kèm định dạng thân yêu "
      "cầu và phản hồi nằm trong tài liệu API của kho mã nguồn (docs/api/README.md)."),
    P("Một quyết định thiết kế đáng chú ý ở tầng bảo mật: danh sách đường dẫn nhận dữ "
      "liệu từ node được khai báo tập trung trong một lớp duy nhất tên IngestPaths, "
      "gồm hai danh sách là các đường dẫn cần kiểm tra token và các đường dẫn chỉ cần "
      "cho phép đi qua. Cả chuỗi lọc bảo mật của Spring Security lẫn bộ lọc kiểm tra "
      "token đều đọc từ lớp này. Trước khi có thiết kế này, hai nơi khai báo đường "
      "dẫn độc lập và đã từng lệch nhau: chuỗi lọc bảo mật cho phép đi qua nhưng bộ "
      "lọc token lại bỏ sót, dẫn tới 263 ảnh bằng chứng bị từ chối âm thầm. Bài học "
      "rút ra là khi một thông tin phải xuất hiện ở hai nơi, cần biến nó thành một "
      "nguồn duy nhất mà cả hai nơi cùng đọc."),
    TB(["Endpoint", "Phương thức", "Chức năng", "Bảo vệ"], [
        ["/health", "GET", "Trạng thái node và bộ đếm trong tiến trình: số khung hình "
         "đã xử lý, số vi phạm đã phát hiện, tốc độ xử lý thực tế. Trả mã 503 khi "
         "pipeline ngừng cập nhật quá mười giây", "Công khai"],
        ["/action/stop-line", "GET, POST", "Đọc và đặt vạch dừng ảo kèm hướng giám sát "
         "được suy ra từ hai điểm", "X-Edge-Token, giới hạn tốc độ"],
        ["/action/light-roi", "POST", "Đặt vùng quan sát đèn tín hiệu theo toạ độ và "
         "kích thước", "X-Edge-Token, giới hạn tốc độ"],
        ["/api/light-roi", "GET", "Đọc vùng quan sát đèn đang áp dụng", "Công khai"],
        ["/action/restart", "POST", "Khởi động lại dịch vụ của node", "X-Edge-Token, "
         "giới hạn tốc độ"],
        ["/api/calibration", "GET", "Trạng thái hiệu chuẩn đầy đủ để giao diện vẽ lại "
         "vạch dừng và vùng đèn", "Công khai"],
        ["/api/light-state", "GET", "Trạng thái đèn đã làm ổn định do pipeline công bố "
         "ở mỗi khung hình", "Công khai"],
        ["/api/calibration/snapshot", "GET", "Một khung hình JPEG từ nguồn video, dùng "
         "làm nền để vẽ hiệu chuẩn khi luồng trực tiếp chưa sẵn sàng", "Công khai"],
        ["/api/cameras", "GET", "Danh sách camera do node phục vụ", "Công khai"],
        ["/api/cameras/{id}/stream", "GET", "Danh sách phát HLS", "Công khai"],
        ["/api/cameras/{id}/stream/{tệp}", "GET", "Phân đoạn video HLS, có hai biến thể "
         "đường dẫn để tương thích cách trình duyệt sinh URL tương đối", "Công khai"],
        ["/api/cameras/{id}/snapshot", "GET", "Ảnh JPEG chụp từ luồng camera",
         "Công khai"],
    ], "Bảng 4.7. Danh sách API điều khiển tại node biên"),
    P("API của node biên tách phần đọc và phần ghi theo mức độ rủi ro. Các endpoint "
      "chỉ đọc đều mở, vì chúng chỉ trả trạng thái kỹ thuật và luồng video mà node "
      "vốn được giao phục vụ. Ba endpoint ghi nằm dưới nhóm /action và bắt buộc có "
      "token, kèm giới hạn tốc độ ba mươi yêu cầu mỗi phút cho mỗi địa chỉ IP theo "
      "cửa sổ trượt, vì đây là các thao tác làm thay đổi hành vi xét vi phạm. Mọi "
      "phản hồi của node đều kèm tiêu đề bảo mật: không suy diễn kiểu nội dung, không "
      "cho phép nhúng khung và không lưu bộ nhớ đệm."),
    TB(["Mã", "Ý nghĩa trong hệ thống"], [
        ["200 OK", "Yêu cầu đọc thành công, hoặc thao tác hiệu chuẩn đã được node áp "
         "dụng"],
        ["201 Created", "Đã tạo hồ sơ vi phạm mới hoặc đã nhận ảnh bằng chứng"],
        ["400 Bad Request", "Dữ liệu đầu vào không hợp lệ, ví dụ thiếu event_id hoặc "
         "toạ độ không phải số"],
        ["401 Unauthorized", "Thiếu hoặc sai thông tin xác thực: JWT hết hạn, sai chữ "
         "ký, hoặc token nhận dữ liệu không khớp"],
        ["403 Forbidden", "Đã xác thực nhưng vai trò không đủ quyền, ví dụ OPERATOR cố "
         "gắng duyệt hồ sơ"],
        ["404 Not Found", "Không tìm thấy hồ sơ, node hoặc ảnh theo định danh đã cho"],
        ["409 Conflict", "Mã sự kiện đã tồn tại khi nhận một hồ sơ đơn lẻ. Ở chế độ "
         "nhận theo lô, trùng lặp không gây lỗi mà được đếm riêng"],
        ["422 Unprocessable Entity", "Thân yêu cầu không đọc được. Đây là dấu hiệu đặc "
         "trưng của lỗi nâng cấp giao thức HTTP khi trung tâm gọi ngược xuống node"],
        ["429 Too Many Requests", "Vượt giới hạn tốc độ của API điều khiển node; phản "
         "hồi kèm các tiêu đề X-RateLimit để client biết khi nào được gọi lại"],
        ["502 Bad Gateway", "Trung tâm không kết nối được node biên khi chuyển tiếp "
         "yêu cầu hiệu chuẩn"],
        ["503 Service Unavailable", "Node còn chạy nhưng pipeline ngừng xử lý khung "
         "hình quá mười giây"],
    ], "Bảng 4.8. Mã trạng thái HTTP và ý nghĩa trong hệ thống"),
    P("Thiết kế phản hồi cho thao tác nhận dữ liệu theo lô cần được nói rõ vì nó khác "
      "với thói quen thông thường. Phản hồi gồm bốn phần: số bản ghi được chấp nhận, "
      "số bản ghi trùng, số bản ghi lỗi, cùng danh sách mã sự kiện của hai nhóm đầu. "
      "Một lô có bản ghi trùng vẫn trả mã thành công, vì trùng lặp là kết quả bình "
      "thường của cơ chế gửi lại chứ không phải lỗi. Node căn cứ vào danh sách mã sự "
      "kiện được chấp nhận để đánh dấu bản ghi đã gửi trong kho outbox; nếu phản hồi "
      "chỉ gồm một con số tổng, node sẽ không biết bản ghi nào cần đánh dấu."),

    # ------------------------------------------------------- 4.6 UI/UX Design
    H2("4.6. UI/UX Design"),
    P("Bảng điều khiển web phục vụ ba nhóm người dùng có nhịp làm việc rất khác "
      "nhau, nên kiến trúc thông tin được tổ chức theo nhiệm vụ thay vì theo cấu trúc "
      "dữ liệu. Hình 4.8 thể hiện sơ đồ trang và cơ chế bảo vệ ở tầng middleware."),
    IMG(V + "/ui-sitemap.png", 160,
        "Hình 4.8. Kiến trúc thông tin của bảng điều khiển web"),
    P("Toàn bộ trang nghiệp vụ nằm trong một nhóm tuyến đường được middleware bảo vệ: "
      "yêu cầu đi qua middleware trước khi tới trang, nếu thiếu cookie phiên thì bị "
      "chuyển về trang đăng nhập. Cách này tránh tình trạng mỗi trang tự kiểm tra "
      "phiên theo một kiểu khác nhau. Trang đăng nhập là trang duy nhất nằm ngoài "
      "nhóm."),
    TB(["Nguyên tắc thiết kế", "Thể hiện cụ thể trên giao diện"], [
        ["Ngữ cảnh địa phương",
         "Toàn bộ nhãn, thông báo lỗi và định dạng ngày giờ bằng tiếng Việt, theo "
         "ngôn ngữ vi-VN và múi giờ châu Á Thành phố Hồ Chí Minh. Người duyệt không "
         "phải quy đổi thời gian khi đối chiếu với biên bản"],
        ["Bằng chứng trước, quyết định sau",
         "Ở trang duyệt hồ sơ, ảnh bằng chứng chiếm phần lớn diện tích hiển thị; các "
         "trường metadata xếp bên cạnh; nút quyết định nằm ở vị trí cố định để thao "
         "tác lặp lại không phải di chuyển con trỏ"],
        ["Trạng thái luôn nhìn thấy được",
         "Huy hiệu số hồ sơ chờ duyệt trên thanh điều hướng, chấm trạng thái node "
         "trong danh sách, nhãn độ tin cậy đặt cạnh giá trị nhận dạng để người dùng "
         "biết mức độ nên tin tưởng"],
        ["Không phá vỡ công việc đang làm",
         "Trang duyệt tải trước hồ sơ kế tiếp theo lô năm mươi bản ghi; khi người "
         "dùng ra quyết định, hồ sơ mới đã sẵn sàng nên không có khoảng chờ"],
        ["Thao tác nguy hiểm phải xác nhận",
         "Xoá hồ sơ chỉ dành cho vai trò quản trị và yêu cầu xác nhận; thao tác đặt "
         "lại trạng thái về chờ duyệt bị vô hiệu hoá khi hồ sơ đang ở đúng trạng "
         "thái đó"],
        ["Phản hồi bằng hoạt ảnh có mục đích",
         "Hoạt ảnh chỉ dùng ở hai chỗ: chuyển thẻ trong trang duyệt để người dùng "
         "biết thao tác đã được ghi nhận, và trạng thái chờ khi gọi API. Không dùng "
         "hoạt ảnh trang trí làm chậm nhịp làm việc"],
        ["Hỗ trợ cả người dùng bàn phím",
         "Bảng lệnh mở bằng tổ hợp phím tắt để tìm biển số, tìm node và chuyển "
         "trang; các ô nhập có nhãn rõ ràng"],
    ], "Bảng 4.9. Nguyên tắc thiết kế giao diện và cách thể hiện"),
    P("Hệ thống có sáu chủ đề giao diện, trong đó hai chủ đề mô phỏng màu sắc đặc "
      "trưng của lực lượng cảnh sát giao thông với tông đỏ mận và vàng đồng, cùng các "
      "chủ đề tối, sáng và chế độ theo hệ điều hành. Lựa chọn này xuất phát từ khảo "
      "sát ở Chương 2: sản phẩm được dùng bởi cán bộ nghiệp vụ trong môi trường "
      "trực ban có ánh sáng thay đổi, nên cần cả chủ đề tối cho ca đêm lẫn chủ đề "
      "sáng cho ca ngày."),
    IMG(V + "/ui-wireflow.png", 155,
        "Hình 4.9. Wireflow nghiệp vụ duyệt hồ sơ của cán bộ xử lý vi phạm"),
    P("Hình 4.9 mô tả luồng thao tác của nghiệp vụ dùng nhiều nhất trong hệ thống. "
      "Điểm thiết kế đáng chú ý là quyết định tách trang tra cứu và trang duyệt thành "
      "hai nơi khác nhau. Ở các phiên bản đầu, danh sách hồ sơ có nút duyệt nhanh đặt "
      "ngay trên từng dòng. Cách làm này nhanh nhưng dẫn tới quyết định được đưa ra "
      "khi người duyệt chưa xem ảnh bằng chứng ở kích thước đủ lớn. Phiên bản hiện "
      "hành chuyển toàn bộ thao tác quyết định vào trang chi tiết hồ sơ, còn danh "
      "sách chỉ để tra cứu; trang duyệt hàng loạt giữ vai trò luồng làm việc liên tục "
      "cho ca trực. Đây là ví dụ về việc thiết kế giao diện phải phục vụ tính đúng "
      "đắn của quyết định nghiệp vụ, không chỉ phục vụ tốc độ thao tác."),
    P("Công cụ hiệu chuẩn là thành phần giao diện phức tạp nhất. Người vận hành xem "
      "luồng video trực tiếp của node, chọn chế độ vẽ, kéo chuột trên khung hình để "
      "tạo vạch dừng hoặc khoanh vùng đèn, chọn hướng giám sát rồi gửi lên node. Ba "
      "vấn đề kỹ thuật phải giải quyết trong thiết kế: toạ độ chuột phải được quy đổi "
      "từ kích thước hiển thị về kích thước gốc của khung hình, nếu không vạch sẽ "
      "lệch khi cửa sổ thay đổi; lớp vẽ phải nằm trên video mà không làm giật luồng "
      "phát, nên việc vẽ lại được gắn vào vòng lặp hoạt ảnh của trình duyệt thay vì "
      "vẽ theo sự kiện chuột; và giá trị hiệu chuẩn phải được đọc lại từ node sau khi "
      "gửi để xác nhận, tránh trường hợp giao diện hiển thị một đường vạch còn node "
      "lại đang áp dụng đường khác."),
    PB,

    # ============================================================== CHƯƠNG 5
    H1("CHƯƠNG 5. CÀI ĐẶT VÀ PROTOTYPE"),
    P("Chương này mô tả kết quả cài đặt: quy mô mã nguồn, cách hiện thực từng tầng, "
      "cấu hình cơ sở dữ liệu và hạ tầng triển khai, cuối cùng là phần giới thiệu sản "
      "phẩm chạy thật kèm ảnh chụp giao diện."),

    H2("5.1. Frontend"),
    TB(["Tầng", "Công nghệ", "Số tệp", "Số dòng mã"], [
        ["Node biên", "Python 3.11, FastAPI, OpenCV, ultralytics, supervision",
         "30", "6.597"],
        ["Máy chủ trung tâm", "Java 17, Spring Boot 3.3.2, Spring Data JPA, jjwt, "
         "MinIO SDK", "45", "3.525"],
        ["Bảng điều khiển web", "Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, "
         "hls.js, recharts, motion", "35", "6.547"],
        ["Kiểm thử", "pytest", "8", "1.717"],
        ["Tổng cộng", "", "118", "18.386"],
    ], "Bảng 5.1. Quy mô mã nguồn theo tầng"),
    P("Bảng điều khiển web dùng kiến trúc App Router của Next.js với một nhóm tuyến "
      "đường được bảo vệ. Chín tuyến đường phục vụ đủ các nghiệp vụ đã phân tích ở "
      "Chương 3."),
    TB(["Tuyến đường", "Chức năng cài đặt"], [
        ["/login", "Biểu mẫu đăng nhập, gọi API xác thực, lưu phiên và chuyển về "
         "trang tổng quan; thông báo lỗi bằng tiếng Việt"],
        ["/", "Bảng điều khiển tổng quan: bốn thẻ chỉ số, biểu đồ phân bố vi phạm "
         "theo giờ, danh sách hồ sơ gần đây, trạng thái các node"],
        ["/violations", "Tra cứu hồ sơ có phân trang phía máy chủ, lọc theo trạng "
         "thái, node và biển số; chỉ đọc, không có thao tác duyệt tại đây"],
        ["/violations/{id}", "Chi tiết hồ sơ: ảnh bằng chứng, biển số, trạng thái "
         "đèn, toạ độ cắt vạch, nhật ký kiểm toán và ba nút quyết định"],
        ["/review", "Duyệt hàng loạt theo lô năm mươi hồ sơ chờ, tải trước hồ sơ kế "
         "tiếp, ảnh bằng chứng lớn, hiệu ứng chuyển thẻ"],
        ["/nodes", "Danh sách node biên kèm trạng thái trực tuyến suy ra từ mốc "
         "heartbeat"],
        ["/nodes/{nodeId}", "Chi tiết node: luồng video trực tiếp, công cụ hiệu "
         "chuẩn vạch dừng, vùng đèn và hướng giám sát"],
        ["/cameras", "Lưới xem camera trực tiếp với bốn bố cục một, bốn, chín và "
         "mười sáu khung hình"],
        ["/settings", "Chọn chủ đề giao diện, xem thông tin phiên, đổi mật khẩu và "
         "đăng xuất"],
    ], "Bảng 5.2. Các trang của bảng điều khiển web"),
    P("Hai quyết định cài đặt ở tầng web đáng được ghi lại:"),
    UL([
        "Trình duyệt không bao giờ gọi trực tiếp tới trung tâm hay node. Tệp cấu "
        "hình next.config.ts khai báo ba luật viết lại đường dẫn: mọi yêu cầu tới "
        "/api được chuyển tới máy chủ trung tâm, yêu cầu tới /edge-api được chuyển "
        "tới node biên. Nhờ vậy cấu hình nội bộ không lộ ra phía client và không "
        "phát sinh vấn đề chia sẻ tài nguyên khác nguồn. Địa chỉ đích được cố định "
        "lúc biên dịch ảnh, nên ảnh Docker của tầng web nhận địa chỉ theo tên dịch vụ "
        "trong mạng nội bộ.",
        "Thư viện phát video hls.js được tách thành mô-đun riêng và nạp bằng cách "
        "nhập tĩnh trong thành phần phát video. Lý do là thư viện này can thiệp vào "
        "đối tượng toàn cục của trình duyệt; việc nạp theo kiểu động từng gây lỗi khi "
        "trang được dựng phía máy chủ.",
    ]),
    P("Thành phần phát video tự vẽ lớp phủ bằng vòng lặp hoạt ảnh của trình duyệt. "
      "Ở cách làm trước đó, mỗi sự kiện vẽ lại kích hoạt một lượt kết xuất React, và "
      "khi người dùng kéo vạch hiệu chuẩn thì luồng video bị giật thấy rõ. Chuyển "
      "sang vòng lặp hoạt án h giúp việc vẽ lớp phủ tách khỏi chu trình kết xuất, "
      "luồng video giữ được nhịp ổn định."),
    TB(["Mô-đun", "Trách nhiệm"], [
        ["lib/api.ts", "Bọc toàn bộ lời gọi mạng. Tự gắn tiêu đề xác thực, tự làm "
         "mới token đúng một lần khi gặp mã 401 rồi thử lại yêu cầu gốc, tự thử lại "
         "với lỗi mạng và lỗi phía máy chủ; không thử lại với lỗi phía client"],
        ["lib/auth.ts", "Quản lý phiên: lưu token, đồng bộ cookie cho middleware, "
         "làm mới và xoá phiên"],
        ["lib/types.ts", "Định nghĩa kiểu TypeScript phản chiếu DTO của máy chủ "
         "trung tâm"],
        ["lib/nodes.ts", "Chuẩn hoá trạng thái node từ mốc heartbeat"],
        ["components/VideoPlayer.tsx", "Phát luồng HLS và vẽ lớp phủ hiệu chuẩn bằng "
         "vòng lặp hoạt ảnh"],
        ["components/NodeCalibrationPanel.tsx", "Ba chế độ vẽ và chọn hướng giám sát, "
         "quy đổi toạ độ hiển thị về toạ độ gốc"],
        ["components/AppLayout.tsx", "Khung giao diện chung: thanh điều hướng, thanh "
         "trên, huy hiệu số hồ sơ chờ duyệt"],
        ["components/AISidebar.tsx", "Bảng trợ lý trượt từ cạnh phải, hiển thị câu "
         "lệnh SQL đã sinh cùng bảng kết quả hoặc biểu đồ"],
        ["components/CommandPalette.tsx", "Bảng lệnh tìm nhanh và chuyển trang bằng "
         "bàn phím"],
        ["components/DataTable.tsx, BarChart.tsx, StatusBadge.tsx, Toast.tsx, "
         "ThemeProvider.tsx, motion.tsx",
         "Các thành phần dùng lại: bảng dữ liệu, biểu đồ cột, huy hiệu trạng thái, "
         "thông báo ngắn, quản lý chủ đề và hiệu ứng chuyển động"],
    ], "Bảng 5.3. Các mô-đun chính của tầng web"),

    H2("5.2. Backend"),
    H3("5.2.1. Node biên"),
    P("Node biên là phần cài đặt nhiều thử nghiệm nhất của đồ án, vì đây là nơi quyết "
      "định chất lượng phát hiện. Bảng 5.4 liệt kê các mô-đun chính."),
    TB(["Mô-đun", "Số dòng", "Trách nhiệm và ghi chú cài đặt"], [
        ["core/byte_tracker.py", "824",
         "Lớp kế thừa có tinh chỉnh của ByteTrack trong thư viện supervision. Mở cấu "
         "hình cho khối theo dõi chưa xác nhận vốn bị cố định trong thư viện, thêm "
         "cơ chế lọc theo chuyển động và làm mượt độ tin cậy bằng trung bình trượt "
         "có trọng số, bỏ phiếu nhãn qua nhiều khung hình"],
        ["core/motion.py", "158",
         "Mô hình chuyển động của đối tượng theo dõi: tích luỹ quỹ đạo, ước lượng "
         "vận tốc bằng hồi quy bình phương tối thiểu theo thời gian thực thay vì "
         "theo số khung hình, suy ra nhãn hướng trong tám hướng"],
        ["core/detector.py", "234",
         "Bọc mô hình phát hiện phương tiện đã huấn luyện lại. Giữ nguyên tên lớp do "
         "mô hình sinh ra, không ánh xạ về bộ nhãn COCO"],
        ["core/traffic_light_yolo.py", "325",
         "Phân loại màu đèn bằng mô hình học sâu, kết hợp bằng chứng màu trong không "
         "gian HSV và vị trí bóng đèn đang sáng trong vùng quan sát"],
        ["core/traffic_light_cv.py", "",
         "Bộ phân loại dự phòng dùng HSV thuần, đồng thời đảm nhận việc tự tìm vùng "
         "quan sát đèn khi chưa được hiệu chuẩn"],
        ["core/violation_logic.py", "305",
         "Ba lớp: RedLightStabilizer làm ổn định trạng thái đèn, Tripwire mô hình "
         "hoá vạch dừng có hướng, ViolationDetector kết hợp hai thành phần trên với "
         "danh sách đối tượng theo dõi để sinh sự kiện vi phạm"],
        ["core/plate_detector.py, plate_associator.py, ocr_recognizer.py", "",
         "Phát hiện vùng biển số, gắn biển vào đối tượng theo dõi theo quan hệ bao "
         "chứa, đọc ký tự bằng fast-plate-ocr và lưu kết quả đọc tốt nhất theo từng "
         "đối tượng"],
        ["core/vn_plate.py", "180",
         "Bộ kiểm tra cấu trúc biển số Việt Nam và sửa các lỗi nhận dạng ký tự thường "
         "gặp; là cổng lọc cuối cùng trước khi biển số được ghi vào hồ sơ"],
        ["core/visualizer.py", "531",
         "Vẽ ảnh bằng chứng: hộp phương tiện, vạch dừng, nhãn đối tượng, thông tin "
         "đèn và biển số"],
        ["core/pipeline.py", "352",
         "Điều phối bảy bước xử lý mỗi khung hình và ghi hồ sơ vào kho outbox"],
        ["outbox.py", "223",
         "Kho SQLite bền vững, truy cập có khoá bảo vệ để dùng an toàn từ hai luồng"],
        ["violation_sender.py", "220",
         "Luồng gửi dữ liệu nền: lấy bản ghi chờ, gửi theo lô, tải ảnh lên sau, lùi "
         "theo hàm mũ khi thất bại"],
        ["api/server.py", "739",
         "API điều khiển bằng FastAPI: kiểm tra sức khoẻ, phục vụ camera và HLS, "
         "nhận lệnh hiệu chuẩn có token và giới hạn tốc độ"],
        ["camera_stream.py", "139",
         "Gọi FFmpeg để phát luồng HLS với phân đoạn bốn giây, tự khởi động lại khi "
         "gián đoạn, xoá phân đoạn cũ"],
        ["central_client.py", "",
         "Đăng ký node lên trung tâm và duy trì heartbeat mỗi sáu mươi giây; vẫn "
         "tiếp tục chạy khi trung tâm không phản hồi"],
        ["main.py", "433",
         "Khởi động toàn bộ: nạp cấu hình, tạo pipeline, mở API điều khiển, bật "
         "luồng gửi dữ liệu"],
        ["settings.py", "254",
         "Cấu hình tập trung đọc từ biến môi trường, đóng băng sau khi khởi tạo"],
        ["metrics.py", "",
         "Bộ đếm trong tiến trình và phát hiện pipeline ngừng hoạt động"],
    ], "Bảng 5.4. Các module chính của node biên"),
    P("Bốn điểm cài đặt thể hiện rõ nhất quá trình thử nghiệm của đồ án:"),
    UL([
        "Điểm neo để xét cắt vạch là điểm đáy giữa của hộp giới hạn, tức điểm phương "
         "tiện tiếp xúc mặt đường. Lựa chọn này từng được đổi sang tâm hộp trong một "
         "phiên bản trước để bắt được trường hợp phương tiện mới nhô đầu qua vạch, "
         "nhưng gây kết luận sớm với xe nghiêng. Phiên bản hiện hành quay lại điểm "
         "đáy giữa và chấp nhận kết luận muộn hơn một đến hai khung hình; nếu cần "
         "phát hiện sớm kiểu nhô đầu qua vạch, việc đó phải làm ở tầng máy trạng "
         "thái xét vi phạm chứ không đổi điểm neo.",
        "Ngưỡng làm ổn định đèn được quy đổi từ giây sang khung hình theo tốc độ "
         "khung hình của nguồn video. Trước đây ngưỡng là số khung hình cố định, dẫn "
         "tới độ trễ chuyển trạng thái khác nhau giữa các camera: bảy khung hình ở "
         "nguồn ba khung hình mỗi giây là hơn hai giây, ở nguồn ba mươi khung hình "
         "chỉ hơn hai trăm mili giây. Sau khi quy đổi theo giây, độ trễ đo được là "
         "0,66 giây ở nguồn sáu khung hình và 0,67 giây ở nguồn ba khung hình.",
        "Vạch dừng đang áp dụng được đọc lại ở mỗi khung hình từ biến toàn cục có "
         "khoá bảo vệ, thay vì giữ trong thuộc tính của bộ xét vi phạm. Nhờ vậy lệnh "
         "hiệu chuẩn có hiệu lực từ khung hình kế tiếp mà không cần khởi động lại "
         "tiến trình.",
        "Mỗi khung hình được xử lý trong một khối bắt lỗi riêng. Khung hình hỏng hay "
         "mô hình lỗi chỉ sinh ra một dòng cảnh báo rồi vòng lặp tiếp tục, vì một "
         "node dừng giữa ca trực sẽ bỏ lọt toàn bộ vi phạm trong khoảng thời gian đó.",
    ]),
    H3("5.2.2. Máy chủ trung tâm"),
    P("Máy chủ trung tâm cài đặt 45 tệp Java với 3.525 dòng mã, tổ chức theo sáu gói "
      "chuẩn của Spring Boot."),
    TB(["Gói", "Số lớp", "Nội dung cài đặt"], [
        ["controller", "5",
         "AuthController (4 phương thức), ViolationController (13), "
         "EdgeNodeController (7), MediaController (4), AiQueryController (1). Chỉ "
         "nhận yêu cầu, kiểm tra dữ liệu đầu vào và gọi service"],
        ["service", "7",
         "ViolationService (361 dòng, nhận dữ liệu idempotent và truy vấn có phân "
         "trang), ViolationStatsService (179 dòng, thống kê bằng truy vấn tổng hợp "
         "ở cơ sở dữ liệu), RefreshTokenService (207 dòng, băm SHA-256, xoay vòng "
         "token, phát hiện dùng lại), EdgeProxyService (188 dòng, gọi ngược xuống "
         "node bằng HTTP/1.1), EdgeNodeService, AiQueryService (226 dòng, sinh câu "
         "lệnh SQL và kiểm tra an toàn), MinioStorageService"],
        ["repository", "4",
         "Giao diện Spring Data JPA. Các truy vấn thường dùng được khai báo thành "
         "phương thức đặt tên; truy vấn tổng hợp dùng chú thích Query để chỉ chạy "
         "đếm và nhóm, không tải thực thể vào bộ nhớ"],
        ["entity", "4",
         "Violation, EdgeNode, User, RefreshToken. Chỉ mục được khai báo ngay trong "
         "chú thích Table để lược đồ và mã nguồn không lệch nhau"],
        ["dto", "16",
         "Đối tượng truyền dữ liệu cho yêu cầu và phản hồi, tách biệt hoàn toàn với "
         "thực thể để thay đổi lược đồ không ảnh hưởng tới giao tiếp API"],
        ["config", "8",
         "SecurityConfig (chuỗi lọc bảo mật và phân quyền), JwtService, "
         "JwtAuthFilter, IngestTokenFilter, IngestPaths (nguồn khai báo duy nhất "
         "cho đường dẫn nhận dữ liệu), MinioConfig, WebConfig, UserSeeder"],
    ], "Bảng 5.5. Các lớp chính của máy chủ trung tâm"),
    P("Ba chi tiết cài đặt ở tầng trung tâm xuất phát từ sự cố gặp phải trong quá "
      "trình phát triển, nên cần được ghi lại:"),
    UL([
        "EdgeProxyService cố định giao thức HTTP/1.1 khi gọi ngược xuống node. Node "
         "biên chạy trên uvicorn, và khi client đề nghị nâng cấp lên HTTP/2, phần "
         "thân yêu cầu POST bị bỏ rơi khiến node trả mã 422 và trung tâm báo 502 về "
         "cho web. Sự cố này rất khó truy nguyên vì lỗi nằm ở tầng giao thức chứ "
         "không ở dữ liệu.",
        "Chuỗi lọc bảo mật khai báo đường dẫn bằng cặp phương thức và mẫu đường dẫn, "
         "không ghép thành một chuỗi. Trên phiên bản Spring Security đang dùng, "
         "chuỗi ghép dạng phương thức cộng khoảng trắng cộng đường dẫn bị hiểu là "
         "một mẫu đường dẫn duy nhất nên không bao giờ khớp, dẫn tới mọi yêu cầu "
         "nhận dữ liệu bị từ chối dù token đúng.",
        "Ứng dụng tắt theo cơ chế êm với thời gian chờ ba mươi giây. Khi nâng cấp, "
         "một lô dữ liệu đang ghi dở sẽ được hoàn tất thay vì bị ngắt giữa chừng.",
    ]),
    P("Bộ trợ lý hỏi dữ liệu bằng tiếng Việt hoạt động theo bốn bước: dựng lời nhắc "
      "gồm mô tả lược đồ bốn bảng và yêu cầu chỉ viết câu lệnh SELECT, gửi tới mô "
      "hình ngôn ngữ qua giao diện REST của java.net.http, kiểm tra câu lệnh nhận về "
      "bằng bộ chặn từ khoá nguy hiểm rồi tự thêm giới hạn năm mươi dòng, cuối cùng "
      "thực thi bằng truy vấn chỉ đọc và trả về kết quả kèm gợi ý loại biểu đồ. Việc "
      "kiểm tra được đặt sau khi nhận câu lệnh chứ không đặt trong lời nhắc, vì lời "
      "nhắc chỉ là yêu cầu còn bộ kiểm tra mới là ràng buộc thực sự."),

    H2("5.3. Database"),
    P("Cơ sở dữ liệu PostgreSQL 16 chạy trong container, dữ liệu đặt trên volume "
      "riêng để không mất khi tạo lại container. Cơ sở dữ liệu tên rlvd_central, tài "
      "khoản truy cập và mật khẩu đọc từ tệp biến môi trường; giá trị mặc định chỉ "
      "dùng cho phát triển nội bộ."),
    P("Ảnh bằng chứng lưu trong MinIO, dịch vụ lưu trữ đối tượng tương thích giao "
      "diện S3, với một bucket tên violations. Khoá đối tượng theo mẫu "
      "violations/{event_id}/{uuid}.jpg; bảng violations chỉ lưu khoá này trong cột "
      "media_url chứ không lưu đường dẫn tuyệt đối, để việc di chuyển hệ thống lưu "
      "trữ không đòi hỏi cập nhật dữ liệu. Ảnh được phát lại cho trình duyệt qua "
      "một endpoint trung gian của máy chủ trung tâm, nên bucket không cần mở ra "
      "ngoài."),
    P("Kho outbox của node biên đặt trên volume outbox_data, thư mục HLS đặt trên "
      "volume hls_data. Tổng cộng bốn volume bền vững: postgres_data, minio_data, "
      "outbox_data và hls_data. Việc tách kho outbox thành volume riêng là điều kiện "
      "để node biên có thể bị tạo lại mà không mất hồ sơ chưa gửi."),
    P("Múi giờ châu Á Thành phố Hồ Chí Minh được đặt trên toàn bộ dịch vụ và trong "
      "tệp Dockerfile, để mốc thời gian không bị lệch khi so sánh giữa nhật ký node, "
      "cơ sở dữ liệu và giao diện."),

    H2("5.4. Demo"),
    P("Toàn bộ hệ thống được đóng gói trong tệp docker-compose.full.yml gồm năm dịch "
      "vụ chạy trên một mạng cầu nối nội bộ."),
    TB(["Dịch vụ", "Ảnh hoặc cách dựng", "Cổng máy chủ", "Kiểm tra sức khoẻ"], [
        ["postgres", "postgres:16-alpine", "5432", "pg_isready, mỗi mười giây"],
        ["minio", "minio/minio:latest", "9010 (API), 9011 (bảng điều khiển)",
         "mc ready local, mỗi mười giây"],
        ["central-server", "Dựng hai giai đoạn bằng Maven trong image "
         "maven:3.9-eclipse-temurin-17", "8002 → 8000",
         "wget /api/health, mỗi mười giây, cho phép ba mươi giây khởi động"],
        ["web-dashboard", "Dựng Next.js ở chế độ standalone, chạy bằng tài khoản "
         "không phải root", "3000",
         "Chờ central-server khoẻ mới khởi động"],
        ["edge-pipeline", "Dựng từ image python:3.11-slim, cài PyTorch bản CUDA "
         "12.4, nhận card đồ hoạ qua khai báo device reservation", "8082 → 8080",
         "/health do node tự cung cấp"],
    ], "Bảng 5.6. Ánh xạ cổng dịch vụ khi triển khai bằng Docker Compose"),
    P("Hệ thống khởi động bằng một lệnh thông qua tập lệnh start.sh, tập lệnh này "
      "kiểm tra điều kiện trước khi dựng: tệp trọng số mô hình có đủ không, tệp video "
      "đầu vào có tồn tại không, tệp biến môi trường đã có chưa. Ngoài chế độ đầy đủ, "
      "tập lệnh có chế độ tối thiểu chỉ dựng bốn dịch vụ web mà bỏ node biên, dùng "
      "khi cần làm việc với giao diện mà không cần card đồ hoạ. Các thao tác khác gồm "
      "xem nhật ký, xem trạng thái, dựng lại từng dịch vụ và dừng toàn bộ."),
    TB(["Biến môi trường", "Tầng", "Bắt buộc khi vận hành", "Ý nghĩa"], [
        ["JWT_SECRET", "Trung tâm", "Có, tối thiểu 32 ký tự",
         "Khoá ký token truy cập; ứng dụng từ chối khởi động nếu khoá yếu"],
        ["JWT_TTL_SECONDS", "Trung tâm", "Không, mặc định 43.200",
         "Tuổi thọ token truy cập, tương ứng một ca trực mười hai giờ"],
        ["INGEST_TOKEN", "Trung tâm và node", "Có",
         "Token dùng chung cho các yêu cầu nhận dữ liệu từ node"],
        ["ADMIN_USERNAME, ADMIN_PASSWORD", "Trung tâm", "Có, phải đổi giá trị mặc định",
         "Tài khoản quản trị tạo lần đầu; không ghi đè nếu đã tồn tại"],
        ["DB_PASS, DB_HOST, DB_NAME, DB_USER", "Trung tâm và postgres", "Có",
         "Thông tin kết nối PostgreSQL"],
        ["MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, MINIO_ACCESS_KEY, "
         "MINIO_SECRET_KEY", "Trung tâm và minio", "Có, phải đổi giá trị mặc định",
         "Thông tin truy cập kho đối tượng"],
        ["GEMINI_API_KEY", "Trung tâm", "Có nếu dùng trợ lý trí tuệ nhân tạo",
         "Khoá gọi mô hình ngôn ngữ"],
        ["EDGE_API_TOKEN", "Node", "Có",
         "Token bảo vệ nhóm endpoint /action của node"],
        ["EDGE_REQUIRE_TOKEN", "Node", "Có, đặt true khi vận hành",
         "Bật chế độ bắt buộc kiểm tra token ở node"],
        ["EDGE_ALLOWED_ORIGINS", "Node", "Có",
         "Danh sách nguồn được phép gọi API điều khiển của node"],
        ["NODE_ID", "Node", "Không, mặc định edge-node-01",
         "Định danh node, đồng thời là thành phần của mã sự kiện"],
        ["VIDEO_INPUT", "Node", "Có", "Nguồn video: đường dẫn tệp hoặc địa chỉ luồng"],
        ["YOLO_MODEL_PATH, TRAFFIC_LIGHT_MODEL_PATH", "Node", "Có",
         "Đường dẫn tệp trọng số mô hình"],
        ["OUTBOX_BATCH_SIZE, OUTBOX_FLUSH_INTERVAL", "Node",
         "Không, mặc định 20 và 5 giây", "Kích thước lô và chu kỳ gửi dữ liệu"],
        ["CENTRAL_SERVER_URL", "Node và web", "Có", "Địa chỉ máy chủ trung tâm"],
    ], "Bảng 5.7. Biến môi trường cấu hình hệ thống"),
    P("Các hình từ 5.1 đến 5.11 là ảnh chụp giao diện của hệ thống đang chạy thật "
      "trên máy phát triển, không phải ảnh dựng bằng công cụ thiết kế. Dữ liệu hiển "
      "thị là hồ sơ vi phạm do node biên sinh ra từ các tệp video thử nghiệm."),
    IMG(S + "/10-login.png", 150, "Hình 5.1. Trang đăng nhập"),
    IMG(S + "/11-dashboard-authed.png", 155,
        "Hình 5.2. Trang bảng điều khiển tổng quan sau khi đăng nhập"),
    IMG(S + "/12-violations-list.png", 155,
        "Hình 5.3. Trang tra cứu hồ sơ vi phạm có phân trang và lọc"),
    IMG(S + "/13-violation-detail.png", 158,
        "Hình 5.4. Trang chi tiết hồ sơ với ảnh bằng chứng và nút quyết định xử lý"),
    IMG(S + "/14-review.png", 158, "Hình 5.5. Trang duyệt hồ sơ hàng loạt"),
    IMG(S + "/15-cameras-grid.png", 158,
        "Hình 5.6. Trang xem camera trực tiếp với các bố cục lưới 1, 4, 9 và 16"),
    IMG(S + "/16-nodes-list.png", 158, "Hình 5.7. Trang danh sách node biên"),
    IMG(S + "/18-node-detail-live.png", 158,
        "Hình 5.8. Trang chi tiết node với luồng video trực tiếp"),
    IMG(S + "/19-node-calibration.png", 158,
        "Hình 5.9. Công cụ hiệu chuẩn vạch dừng trên video trực tiếp"),
    IMG(S + "/21-ai-sidebar.png", 158,
        "Hình 5.10. Trợ lý hỏi dữ liệu bằng tiếng Việt"),
    IMG(S + "/17-settings.png", 158, "Hình 5.11. Trang cài đặt giao diện"),
    P("Kịch bản giới thiệu sản phẩm được thực hiện theo đúng luồng nghiệp vụ đã phân "
      "tích, và là kịch bản đã chạy thật trên hệ thống:"),
    UL([
        "Khởi động hệ thống bằng một lệnh, chờ năm dịch vụ báo trạng thái khoẻ.",
        "Mở trang đăng nhập, đăng nhập bằng tài khoản quản trị, hệ thống chuyển về "
         "trang tổng quan với các chỉ số và biểu đồ theo giờ.",
        "Vào trang chi tiết node, mở công cụ hiệu chuẩn, kéo chuột trên khung hình "
         "video trực tiếp để tạo vạch dừng và chọn hướng giám sát, gửi xuống node. "
         "Đọc lại trạng thái hiệu chuẩn từ node để xác nhận giá trị đã được áp dụng.",
        "Chuyển sang trang camera để xem luồng video trực tiếp, đổi bố cục lưới.",
        "Chờ node xử lý video và sinh hồ sơ. Ở trang tổng quan, huy hiệu số hồ sơ chờ "
         "duyệt tăng lên.",
        "Mở trang duyệt, xem ảnh bằng chứng của một hồ sơ, đối chiếu trạng thái đèn "
         "và toạ độ cắt vạch, chọn duyệt hoặc từ chối. Hồ sơ kế tiếp hiện ra ngay mà "
         "không phải chờ tải.",
        "Mở bảng trợ lý, đặt một câu hỏi bằng tiếng Việt về dữ liệu vi phạm. Hệ thống "
         "trả về câu lệnh SQL đã sinh cùng bảng kết quả và biểu đồ.",
        "Dừng dịch vụ trung tâm để chứng minh cơ chế giao hàng bền vững: node vẫn "
         "tiếp tục phát hiện và ghi hồ sơ vào kho outbox, số bản ghi chờ tăng lên. "
         "Khởi động lại trung tâm, các hồ sơ được gửi hết mà không có bản nào bị "
         "trùng.",
    ]),
    PB,
]
