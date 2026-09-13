# -*- coding: utf-8 -*-
"""Đồ án RLVD — nội dung phần 3: Chương 6 (Kiểm thử) → tài liệu tham khảo.

Mọi số liệu đo lại vào ngày 12/9/2026 trên máy phát triển (Linux, RTX 2060):
  * pytest tests/ -q           -> 124 passed in 1.09s
  * run_pipeline.py aziz1.MP4  -> 120 frames, 1 vi phạm (frame 32, track 2)
  * run_pipeline.py 20221003   -> 384 frames, 0 vi phạm
  * /tmp/bench_ocr2.json       -> OCR 5,2 ms trung vị; 40/40 crop đọc được chữ,
                                  0/40 qua validator biển VN (video biển số Anh)
  * scripts/benchmark_results.csv -> mAP50 0,9436 (custom) vs 0,0101 (COCO)
  * GET /api/violations/counts trên stack Docker đang chạy
Số liệu nào chưa đo được thì ghi rõ "chưa đo", không suy diễn từ thiết kế.
"""

from thesis_part1 import P, H1, H2, H3, UL, PL, TB, IMG, PB, T, V, S, ROOT

BLOCKS_3 = [
    # ============================================================ CHƯƠNG 6
    H1("CHƯƠNG 6. KIỂM THỬ"),
    P("Mục tiêu của chương này là trả lời một câu hỏi duy nhất: hệ thống vừa xây dựng "
      "có làm đúng những gì đã phân tích ở Chương 3 không, và đúng tới mức nào. Vì vậy "
      "kiểm thử được tổ chức theo yêu cầu chứ không theo mô-đun: mỗi yêu cầu chức năng "
      "và phi chức năng đều có ít nhất một ca kiểm thử tương ứng, và mỗi con số đưa ra "
      "đều kèm cách đo để người đọc có thể lặp lại."),
    P("Đồ án dùng ba mức kiểm thử. Mức thứ nhất là kiểm thử đơn vị bằng pytest, chạy "
      "hoàn toàn không cần card đồ hoạ hay tệp trọng số, nên chạy được trong môi "
      "trường tích hợp liên tục với mỗi lần đẩy mã lên kho. Mức thứ hai là kiểm thử "
      "tích hợp trên hệ thống đang chạy thật bằng Docker Compose, dùng yêu cầu HTTP "
      "gửi tới các dịch vụ. Mức thứ ba là kiểm thử hồi quy trên video có nhãn, tức là "
      "chạy toàn bộ pipeline trên tệp video đã biết trước có bao nhiêu vi phạm rồi đối "
      "chiếu kết quả."),

    H2("6.1. Test Case"),
    P("Bảng 6.1 liệt kê các ca kiểm thử chính, mỗi ca gắn với một yêu cầu ở Chương 2 "
      "để có thể truy vết ngược. Cột kết quả được điền từ lần chạy ngày 12 tháng 9 năm "
      "2026."),
    TB(["Mã", "Yêu cầu liên quan", "Tình huống kiểm thử", "Kết quả mong đợi",
        "Cách thực hiện", "Kết quả"], [
        ["TC-01", "FR-04",
         "Phương tiện cắt vạch dừng khi đèn đỏ đã ổn định",
         "Sinh đúng một hồ sơ cho đối tượng đó",
         "Kiểm thử đơn vị, dựng vết và tín hiệu đèn giả", "Đạt"],
        ["TC-02", "FR-04",
         "Phương tiện cắt vạch khi đèn xanh hoặc đèn vàng",
         "Không sinh hồ sơ nào",
         "Kiểm thử đơn vị", "Đạt"],
        ["TC-03", "FR-04",
         "Phương tiện đi ngược hướng giám sát đã chọn",
         "Không sinh hồ sơ",
         "Kiểm thử đơn vị, đặt hướng một chiều", "Đạt"],
        ["TC-04", "FR-04",
         "Phương tiện vào vùng chết của vạch rồi lùi lại",
         "Không kết luận vi phạm sai",
         "Kiểm thử đơn vị, dựng quỹ đạo dao động quanh vạch", "Đạt"],
        ["TC-05", "FR-04",
         "Phương tiện cắt vạch sang rồi cắt về trong cùng một lượt",
         "Chỉ một hồ sơ duy nhất cho một đối tượng",
         "Kiểm thử đơn vị", "Đạt"],
        ["TC-06", "FR-03",
         "Chưa hiệu chuẩn vạch, đèn đỏ kéo dài nhiều khung hình, sau đó kẻ vạch",
         "Trước khi kẻ vạch không có hồ sơ; sau khi kẻ thì xét bình thường",
         "Kiểm thử đơn vị với hàm đặt vạch đang áp dụng", "Đạt"],
        ["TC-07", "FR-02",
         "Bộ phân loại đèn trả kết quả nhiễu xen kẽ giữa các khung hình",
         "Tín hiệu ổn định không bị lật theo nhiễu đơn lẻ",
         "Kiểm thử đơn vị máy trạng thái có trễ chuyển", "Đạt"],
        ["TC-08", "FR-02, NFR-02",
         "Đổi nguồn video từ ba sang ba mươi khung hình mỗi giây",
         "Thời gian chuyển trạng thái đèn tính bằng giây không đổi",
         "Kiểm thử đơn vị hàm quy đổi theo tốc độ khung hình", "Đạt"],
        ["TC-09", "FR-01",
         "Xe máy có độ tin cậy phát hiện thấp chạy nhanh qua nhiều khung hình",
         "Vẫn giữ cùng một mã theo dõi, không nhấp nháy",
         "Kiểm thử đơn vị bộ theo dõi đã tinh chỉnh", "Đạt"],
        ["TC-10", "FR-01",
         "Hai phương tiện giao nhau trong khung hình",
         "Giữ hai mã theo dõi riêng biệt",
         "Kiểm thử đơn vị", "Đạt"],
        ["TC-11", "FR-06",
         "Chuỗi biển số bị nhận dạng nhầm giữa chữ O và số 0, chữ I và số 1",
         "Sửa đúng về dạng hợp lệ hoặc trả về rỗng",
         "Kiểm thử đơn vị bộ kiểm tra cấu trúc biển số", "Đạt"],
        ["TC-12", "FR-06",
         "Biển số không theo cấu trúc Việt Nam",
         "Bị loại, không ghi vào hồ sơ",
         "Kiểm thử đơn vị và đo thực tế ở mục 6.4", "Đạt"],
        ["TC-13", "FR-07",
         "Gửi cùng một lô dữ liệu hai lần",
         "Lần thứ hai không tạo bản ghi mới, đếm là trùng và không báo lỗi",
         "Kiểm thử tích hợp qua HTTP trên stack đang chạy", "Đạt"],
        ["TC-14", "FR-07, NFR-03",
         "Ngắt kết nối tới trung tâm trong lúc pipeline đang chạy",
         "Hồ sơ tích luỹ trong kho outbox, không mất bản nào",
         "Kiểm thử tích hợp, dừng dịch vụ trung tâm", "Đạt"],
        ["TC-15", "NFR-04",
         "Khởi động lại container node biên khi còn hồ sơ chờ gửi",
         "Hồ sơ chờ cũ được gửi tiếp nhờ kho gắn trên volume",
         "Kiểm thử tích hợp", "Đạt"],
        ["TC-16", "FR-13",
         "Đăng nhập sai mật khẩu, hoặc dùng JWT không hợp lệ",
         "Trả 401 với thông báo chung, không tiết lộ tài khoản có tồn tại hay không",
         "Kiểm thử tích hợp", "Đạt"],
        ["TC-17", "FR-13",
         "Mỗi vai trò gọi tới endpoint không thuộc quyền của mình",
         "Trả 403 đúng chỗ, trả 200 khi đủ quyền",
         "Kiểm thử tích hợp theo ma trận vai trò và endpoint", "Đạt"],
        ["TC-18", "FR-13",
         "Dùng lại một token làm mới đã bị thu hồi",
         "Thu hồi toàn bộ phiên của người dùng đó",
         "Kiểm thử bằng đọc mã và kiểm tra hành vi qua HTTP", "Đạt"],
        ["TC-19", "FR-12",
         "Câu hỏi dẫn tới câu lệnh SQL nguy hiểm",
         "Bị chặn trước khi thực thi, trả thông báo lỗi",
         "Kiểm thử tích hợp với API trợ lý", "Đạt"],
        ["TC-20", "FR-03",
         "Gọi API điều khiển của node với token sai hoặc vượt giới hạn tốc độ",
         "Lần lượt trả 401 và 429 kèm tiêu đề giới hạn",
         "Kiểm thử tích hợp gọi thẳng node", "Đạt"],
        ["TC-21", "FR-03",
         "Hiệu chuẩn vạch dừng từ web khi pipeline đang chạy",
         "Vạch mới có hiệu lực ở khung hình kế tiếp, không cần khởi động lại",
         "Kiểm thử tích hợp: gửi yêu cầu rồi đọc lại trạng thái hiệu chuẩn",
         "Đạt"],
        ["TC-22", "NFR-01",
         "Chạy pipeline liên tục trên video, theo dõi tốc độ xử lý",
         "Tốc độ xử lý không thấp hơn tốc độ nguồn",
         "Đọc bộ đếm từ endpoint sức khoẻ của node", "Đạt"],
        ["TC-23", "Hồi quy",
         "Chạy pipeline trên video aziz1.MP4 đã biết trước kết quả",
         "Phát hiện đúng trường hợp vượt đèn đỏ có trong video",
         "Chạy thật, ghi sự kiện ra tệp JSON", "Đạt, xem mục 6.5"],
        ["TC-24", "Hồi quy",
         "Chạy pipeline trên video đối chứng không có vi phạm",
         "Không sinh hồ sơ giả nào",
         "Chạy thật trên 384 khung hình", "Đạt, xem mục 6.5"],
    ], "Bảng 6.1. Danh sách ca kiểm thử"),

    H2("6.2. Functional Testing"),
    H3("6.2.1. Kiểm thử đơn vị"),
    P("Bộ kiểm thử đơn vị gồm 124 ca trong bảy tệp, tổng cộng 1.717 dòng mã kiểm thử, "
      "chạy bằng pytest trong môi trường ảo Python 3.11 của dự án. Toàn bộ đều là kiểm "
      "thử logic thuần: không nạp trọng số mô hình, không mở card đồ hoạ, không cần "
      "mạng, nên thời gian chạy chỉ hơn một giây và chạy được trong môi trường tích "
      "hợp liên tục với mỗi lần đẩy mã."),
    P("Lệnh chạy và kết quả thực tế ngày 12 tháng 9 năm 2026:"),
    P("myenv/bin/python -m pytest tests/ -q  →  124 passed in 1.09s", "CodeV"),
    TB(["Tệp kiểm thử", "Số ca", "Phạm vi kiểm tra"], [
        ["tests/test_track_quality.py", "39",
         "Tầng chất lượng theo dõi bồi trên ByteTrack: điểm neo xét cắt vạch là điểm "
         "đáy giữa, quỹ đạo và vận tốc theo đơn vị pixel trên giây, khoảng trống trước "
         "khi tìm lại đối tượng, độ tin cậy làm mượt, bỏ phiếu nhãn, ghi nhật ký vòng "
         "đời và cảnh báo đổi mã, cơ chế lọc theo chuyển động"],
        ["tests/test_vn_plate.py", "34",
         "Bộ kiểm tra cấu trúc biển số Việt Nam: ba dạng ký tự seri, vị trí dấu gạch "
         "sau mã tỉnh, sửa các nhầm lẫn thường gặp của nhận dạng ký tự"],
        ["tests/test_violation_logic.py", "18",
         "Quy tắc xét vi phạm: cắt vạch khi đèn đỏ ổn định, hướng giám sát, vùng chết, "
         "trường hợp cắt sang rồi cắt về, mỗi đối tượng chỉ một hồ sơ"],
        ["tests/test_geometry.py", "15",
         "Hình học của vạch dừng: xác định phía của điểm so với đoạn thẳng, xét cắt "
         "nhau của hai đoạn, ngưỡng vùng chết"],
        ["tests/test_tracker_stabilizer.py", "10",
         "Máy trạng thái tín hiệu đèn có trễ chuyển và quy đổi ngưỡng theo tốc độ "
         "khung hình của nguồn video"],
        ["tests/test_metrics.py", "8",
         "Bộ đếm trong tiến trình và cơ chế phát hiện pipeline ngừng xử lý"],
        ["tests/conftest.py", "—",
         "Các fixture dùng chung: đặt vạch đang áp dụng, dựng vết giả"],
    ], "Bảng 6.2. Kết quả kiểm thử đơn vị theo tệp"),
    P("Cách phân bổ số ca phản ánh đúng chỗ rủi ro của hệ thống. Tệp có nhiều ca nhất "
      "là kiểm thử chất lượng theo dõi, vì đây là tầng được bổ sung sau cùng và ảnh "
      "hưởng trực tiếp tới việc một phương tiện có bị coi là cắt vạch hay không. Tệp "
      "nhiều ca thứ hai là bộ kiểm tra biển số, vì chuỗi ký tự đầu ra của nhận dạng ký "
      "tự rất đa dạng và cần được chặn lại trước khi ghi vào hồ sơ chính thức."),
    P("Phạm vi chưa được kiểm thử đơn vị cũng cần nói rõ: các mô-đun nạp mô hình học "
      "sâu, gọi thiết bị tính toán hoặc mở luồng video không có ca kiểm thử riêng, vì "
      "chúng phụ thuộc tệp trọng số và phần cứng cụ thể. Chất lượng của nhóm này được "
      "kiểm tra ở mức hồi quy trên video thật, trình bày ở mục 6.5."),

    H3("6.2.2. Kiểm thử chức năng trên hệ thống đang chạy"),
    P("Ngoài kiểm thử đơn vị, toàn bộ chức năng được kiểm tra trên hệ thống dựng bằng "
      "Docker Compose với năm dịch vụ đang chạy. Cách làm là gửi yêu cầu HTTP thật tới "
      "từng endpoint và đối chiếu mã trạng thái cùng nội dung phản hồi với thiết kế ở "
      "mục 4.5, sau đó mở bảng điều khiển web để kiểm tra dữ liệu hiển thị đúng."),
    P("Các kết quả chính thu được trên hệ thống đang chạy tại thời điểm hoàn thiện đồ "
      "án:"),
    UL([
        "Endpoint sức khoẻ của trung tâm trả trạng thái ok kèm dấu thời gian theo múi "
        "giờ Việt Nam, đúng thiết kế không yêu cầu xác thực.",
        "Endpoint sức khoẻ của node biên trả đủ bộ đếm: số khung hình đã xử lý, số vi "
        "phạm phát hiện, tốc độ xử lý thực tế và trạng thái các thành phần. Trong lần "
        "chạy quan sát được, node đạt khoảng 37 khung hình mỗi giây khi xử lý video "
        "nguồn.",
        "Đăng nhập bằng tài khoản quản trị trả về đầy đủ token truy cập, token làm mới, "
        "vai trò và thời hạn, đúng cấu trúc đã thiết kế.",
        "Truy vấn danh sách hồ sơ có phân trang trả đúng cấu trúc gồm nội dung, số "
        "trang, tổng số bản ghi và hai cờ đầu cuối; bộ lọc theo trạng thái hoạt động "
        "đúng.",
        "Endpoint thống kê trả về số liệu tổng hợp khớp với số đếm trực tiếp từ cơ sở "
        "dữ liệu, xác nhận các truy vấn gộp không bị lệch.",
        "Danh bạ node ghi nhận node biên đang trực tuyến kèm địa chỉ mạng nội bộ, cổng "
        "API và mốc heartbeat gần nhất.",
        "Luồng video trực tiếp phát được trong trình duyệt qua giao thức HLS, trang "
        "camera hiển thị đúng bốn bố cục lưới một, bốn, chín và mười sáu khung hình.",
        "Ảnh bằng chứng của hồ sơ được tải về qua endpoint trung gian và hiển thị đúng "
        "trong trang chi tiết, xác nhận cả chuỗi lưu trữ đối tượng và uỷ quyền đều "
        "hoạt động.",
        "Thao tác duyệt và từ chối trên trang chi tiết đổi trạng thái hồ sơ và làm mới "
        "số đếm trên huy hiệu chờ duyệt.",
    ]),
    P("Một điểm cần ghi nhận trung thực: các ca kiểm thử tích hợp này được thực hiện "
      "thủ công bằng tập lệnh và qua giao diện, chưa được tự động hoá thành bộ kiểm "
      "thử chạy lặp lại. Hệ quả là mỗi lần thay đổi mã ở tầng trung tâm, người phát "
      "triển phải tự chạy lại các bước này. Đây là hạn chế được nêu ở Chương 7 và là "
      "hạng mục ưu tiên trong hướng phát triển."),

    H3("6.2.3. Kiểm thử phân quyền"),
    P("Phân quyền là phần dễ sai nhất của hệ thống có nhiều vai trò, nên được kiểm tra "
      "theo ma trận: mỗi vai trò gọi tới từng nhóm endpoint nhạy cảm và ghi lại mã "
      "trạng thái. Kết quả khớp với Bảng 3.1 ở Chương 3: vai trò vận hành bị chặn khi "
      "cố duyệt hồ sơ, vai trò nghiệp vụ bị chặn khi cố hiệu chuẩn node, còn vai trò "
      "quản trị thực hiện được cả hai. Một chi tiết kỹ thuật phát hiện được trong quá "
      "trình này là Spring Boot chuyển tiếp lỗi tới đường dẫn nội bộ /error; nếu "
      "đường dẫn này không được mở trong chuỗi lọc bảo mật thì mã trạng thái thật bị "
      "che thành 403, khiến việc phân biệt lỗi xác thực và lỗi phân quyền trở nên "
      "không thể."),

    H2("6.3. Kiểm thử hiệu năng"),
    P("Hiệu năng của hệ thống được đo ở hai chỗ: tốc độ suy luận của các mô hình học "
      "sâu trên card đồ hoạ, và thời gian xử lý của khâu nhận dạng biển số. Các phép "
      "đo đều chạy trên máy phát triển của đồ án, cấu hình ghi ở Bảng 5.6."),
    TB(["Mô hình", "Độ chính xác", "Độ hồi quy", "mAP50", "mAP50-95",
        "Tốc độ (FP16)"], [
        ["Phát hiện phương tiện đã huấn luyện riêng (yolo26m_vehicle)",
         "0,9022", "0,8939", "0,9436", "0,6796", "73,4 khung hình/giây"],
        ["Mô hình gốc tiền huấn luyện trên bộ COCO (đối chứng)",
         "0,0303", "0,0777", "0,0101", "0,0023", "69,2 khung hình/giây"],
    ], "Bảng 6.3. Kết quả đánh giá mô hình phát hiện phương tiện"),
    P("Bảng 6.3 là kết quả quan trọng nhất về mặt kỹ thuật của đồ án. Hai mô hình có "
      "cùng kiến trúc và cùng tốc độ suy luận, chênh lệch chỉ nằm ở dữ liệu huấn "
      "luyện, nhưng chỉ số mAP50 chênh nhau gần chín mươi bốn lần. Nguyên nhân nằm ở "
      "cột số lượng phát hiện theo lớp của tập kiểm chứng: mô hình gốc gần như không "
      "phát hiện được lớp xe máy nào, trong khi đây lại là lớp phương tiện chiếm đa số "
      "trong dữ liệu thực tế của đồ án. Kết luận rút ra là việc huấn luyện lại trên dữ "
      "liệu đúng miền không phải là bước tối ưu tuỳ chọn mà là điều kiện bắt buộc để "
      "hệ thống hoạt động được tại Việt Nam."),
    P("Cần lưu ý về phạm vi của con số này: mAP50 bằng 0,9436 được đo trên tập kiểm "
      "chứng của chính bộ dữ liệu dùng để huấn luyện, nên nó phản ánh chất lượng khớp "
      "mô hình chứ không phải độ chính xác trên mọi nút giao bất kỳ. Khi chạy trên "
      "video thực tế có khác biệt miền, chất lượng thấp hơn con số này, và đó là lý do "
      "đồ án phải bổ sung tầng làm ổn định tín hiệu cùng cơ chế duyệt hồ sơ có con "
      "người tham gia."),
    P("Hai mô hình còn lại được đánh giá theo cách khác nhau vì bản chất dữ liệu khác "
      "nhau. Mô hình phân loại đèn giao thông là bài toán phân lớp ba nhãn nên được đo "
      "bằng tỉ lệ dự đoán đúng hạng nhất, trước hết trên tập kiểm chứng tách rời rồi "
      "mới đo lại trên video thực tế có khác biệt miền, vì đây chính là chỗ bộ phân "
      "lớp đơn lẻ thường sai. Mô hình phát hiện biển số là bài toán phát hiện một lớp "
      "nên được đo bằng mAP50 trên tập kiểm chứng của nó. Kết quả tổng hợp ở Bảng 6.4."),
    TB(["Mô hình", "Chỉ tiêu", "Trên tập kiểm chứng", "Trên video thực tế",
        "Thời gian suy luận"], [
        ["Phân loại màu đèn (yolo26n-cls)", "Tỉ lệ đúng hạng nhất",
         "100% (bộ dữ liệu LISA)", "99,0% sau khi hợp nhất bằng chứng",
         "khoảng 0,3 mili giây mỗi ảnh"],
        ["Phát hiện biển số", "mAP50", "0,993", "không đo", "xem Bảng 6.5"],
    ], "Bảng 6.4. Kết quả đánh giá mô hình phân loại đèn và mô hình phát hiện biển số"),
    P("Về nguồn gốc số liệu của Bảng 6.4 cần nói rõ hai điểm để người đọc đánh giá "
      "đúng mức độ tin cậy. Thứ nhất, hai chỉ tiêu này được ghi nhận trong các phiên "
      "đo trước của đồ án và lưu tại docs/test-report.md; kho mã nguồn hiện không kèm "
      "tập huấn luyện, tập kiểm chứng và kịch bản đo của hai mô hình này, nên chúng "
      "không thể tái lập bằng một lệnh như Bảng 6.3. Thứ hai, con số 99,0% của khâu "
      "phân loại đèn là kết quả sau khi hợp nhất bằng chứng chứ không phải của riêng "
      "mạng nơ-ron; bản thân bộ phân lớp đơn lẻ cho kết quả thấp hơn trên video thực "
      "tế, và đây chính là lý do kiến trúc ở mục 4.1 có thêm tầng hợp nhất ba nguồn "
      "bằng chứng. Bổ sung kịch bản đo tái lập được cho hai mô hình này là hạng mục "
      "đầu tiên trong hướng phát triển ở Chương 7."),
    TB(["Hạng mục", "Giá trị đo", "Điều kiện đo"], [
        ["Thời gian nhận dạng một ảnh biển số (trung vị)", "5,2 mili giây",
         "40 ảnh cắt từ video, chạy trên CUDA"],
        ["Thời gian nhận dạng một ảnh biển số (trung bình)", "10,0 mili giây",
         "Trung bình cao hơn trung vị do vài ảnh đầu chịu chi phí khởi động"],
        ["Thời gian phát hiện biển số trên một khung hình (trung vị)",
         "30,2 mili giây", "60 khung hình lấy mẫu cách nhau sáu khung hình"],
        ["Thời gian phát hiện biển số trên một khung hình (trung bình)",
         "76,4 mili giây", "Bị kéo lên bởi khung hình đầu tiên nạp mô hình"],
        ["Số biển phát hiện trung bình mỗi khung hình", "1,95",
         "117 hộp giới hạn trên 60 khung hình"],
        ["Tốc độ xử lý của pipeline đầy đủ trên node", "khoảng 37 khung hình/giây",
         "Đọc từ bộ đếm của node đang chạy trong Docker, video nguồn tốc độ thấp hơn"],
        ["Thời gian nạp mô hình nhận dạng biển số", "dưới 0,1 giây",
         "Đo từ lúc khởi tạo tới lúc sẵn sàng suy luận"],
    ], "Bảng 6.5. Kết quả đo thời gian xử lý khâu biển số"),
    P("Một kết quả đo đáng chú ý liên quan tới thiết bị tính toán. Thư viện nhận dạng "
      "ký tự dùng trong đồ án cần CUDA phiên bản mười ba ở bản dựng có hỗ trợ card đồ "
      "hoạ, trong khi máy phát triển cài CUDA phiên bản 12.4, nên khi khởi động thư "
      "viện báo không nạp được thư viện CUDA và tự lùi về chạy trên bộ xử lý trung "
      "tâm. Điều này giải thích vì sao trong nhật ký chạy pipeline có cảnh báo về nhà "
      "cung cấp thực thi CUDA. Ở lần đo ngày 12 tháng 9 năm 2026, khâu nhận dạng chạy "
      "được trên CUDA với thời gian trung vị 5,2 mili giây mỗi ảnh, thấp hơn nhiều so "
      "với con số khoảng hai mươi mili giây ghi nhận trước đó khi chạy trên bộ xử lý "
      "trung tâm. Kết luận thực tế là khâu nhận dạng biển số không phải nút thắt của "
      "pipeline; nút thắt nằm ở khâu phát hiện phương tiện."),

    H2("6.4. Kiểm thử khâu nhận dạng biển số"),
    P("Khâu nhận dạng biển số cần được kiểm tra riêng vì nó là chỗ duy nhất trong "
      "pipeline có bộ lọc nghiệp vụ nằm sau mô hình học sâu. Phép đo thực hiện trên "
      "sáu mươi khung hình lấy mẫu từ một video kiểm thử về nhận dạng biển số, cắt ra "
      "bốn mươi ảnh biển và cho chạy qua cả hai tầng: mô hình nhận dạng ký tự và bộ "
      "kiểm tra cấu trúc biển số Việt Nam."),
    TB(["Chỉ tiêu", "Kết quả", "Diễn giải"], [
        ["Số ảnh cắt đưa vào nhận dạng", "40",
         "Lấy từ 117 hộp giới hạn do mô hình phát hiện biển trả về"],
        ["Số ảnh mô hình đọc ra được chuỗi ký tự", "40 trên 40",
         "Mô hình nhận dạng ký tự hoạt động tốt, không trả chuỗi rỗng"],
        ["Độ tin cậy trung bình của chuỗi đọc được", "khoảng 0,9",
         "Dao động từ 0,73 đến 0,98 trên các mẫu quan sát được"],
        ["Số chuỗi qua được bộ kiểm tra cấu trúc biển Việt Nam", "0 trên 40",
         "Toàn bộ bị loại có chủ đích"],
    ], "Bảng 6.6. Kết quả đo khâu nhận dạng biển số"),
    P("Kết quả bằng không ở dòng cuối không phải là lỗi mà là bằng chứng bộ lọc hoạt "
      "động đúng. Các chuỗi mô hình đọc được trong video kiểm thử này là biển số Vương "
      "quốc Anh, ví dụ NA13NRU hay MW51VSU, tức hai chữ cái rồi hai chữ số rồi ba chữ "
      "cái. Cấu trúc này không khớp với quy định về biển số xe cơ giới Việt Nam, vốn "
      "bắt đầu bằng mã tỉnh hai chữ số từ mười một đến chín mươi chín, tiếp theo là ký "
      "tự seri rồi bốn đến năm chữ số. Bộ kiểm tra vì thế loại toàn bộ, đúng như thiết "
      "kế ở mục 5.2.1 nhằm tránh ghi biển số sai định dạng vào hồ sơ xử phạt."),
    P("Phép đo này đồng thời chỉ ra một hạn chế cần nói thẳng: đồ án chưa có video "
      "giao thông Việt Nam nào có biển số đủ rõ và đủ nhiều để đo tỉ lệ đọc đúng của "
      "khâu nhận dạng trên dữ liệu đúng miền. Con số có thể khẳng định được hiện nay "
      "chỉ gồm hai phần: mô hình phát hiện biển số đạt mAP50 bằng 0,993 trên tập kiểm "
      "chứng của nó, và mô hình nhận dạng ký tự đọc được chuỗi trên toàn bộ mẫu thử "
      "với độ tin cậy cao. Tỉ lệ đọc đúng biển số Việt Nam trên hiện trường là chỉ "
      "tiêu chưa đo được, và đồ án không suy diễn chỉ tiêu này từ hai con số trên."),

    H2("6.5. Kiểm thử hồi quy trên video có nhãn"),
    P("Đây là mức kiểm thử sát với thực tế vận hành nhất: chạy toàn bộ pipeline từ đầu "
      "tới cuối trên tệp video, không bỏ qua khâu nào, rồi đối chiếu số hồ sơ sinh ra "
      "với kết quả đã biết trước của video đó. Hai video được chọn có tính chất bổ "
      "sung cho nhau: một video có chứa trường hợp vượt đèn đỏ để kiểm tra khả năng "
      "phát hiện đúng, và một video không có vi phạm để kiểm tra khả năng không sinh "
      "hồ sơ giả."),
    TB(["Video", "Số khung hình đã xử lý", "Số hồ sơ sinh ra", "Kỳ vọng",
        "Nhận xét"], [
        ["aziz1.MP4 (nút giao thông thực tế, có hiệu chuẩn vạch dừng và vùng đèn sẵn)",
         "120", "1", "Có vi phạm",
         "Phát hiện đúng trường hợp vượt đèn đỏ: khung hình 32, mã theo dõi số 2, "
         "trạng thái đèn đỏ với độ tin cậy 0,95, điểm cắt vạch tại toạ độ 1779 và 1413 "
         "trên ảnh độ phân giải cao"],
        ["20221003-102556.mp4 (video đối chứng, không có vi phạm)",
         "384", "0", "Không có vi phạm",
         "Không sinh hồ sơ giả nào trên toàn bộ độ dài video"],
    ], "Bảng 6.7. Kết quả kiểm thử hồi quy trên video có nhãn"),
    P("Chi tiết hồ sơ do video thứ nhất sinh ra được ghi lại nguyên văn trong tệp kết "
      "quả, gồm mã sự kiện, mã theo dõi, chỉ số khung hình, mốc thời gian, trạng thái "
      "đèn kèm độ tin cậy, điểm cắt vạch và hộp giới hạn của phương tiện. Bộ trường "
      "này khớp đúng với thiết kế bảng violations ở mục 4.4, xác nhận dữ liệu đi hết "
      "chuỗi mà không rơi mất trường nào."),
    P("Một điểm cần giải thích để tránh hiểu nhầm về tính nhất quán. Ở các phiên bản "
      "trước của đồ án, cùng video này cho ra hai hồ sơ. Khác biệt xuất phát từ một "
      "thay đổi thiết kế có chủ đích thực hiện ngày 12 tháng 9 năm 2026: điểm neo dùng "
      "để xét cắt vạch được chuyển từ tâm hộp giới hạn sang điểm đáy giữa, tức điểm "
      "phương tiện tiếp xúc mặt đường. Với điểm neo mới, kết luận đưa ra muộn hơn một "
      "đến hai khung hình nhưng ổn định hơn khi phương tiện nghiêng hoặc hộp giới hạn "
      "co giãn. Trường hợp bị bỏ qua là phương tiện mới nhô đầu qua vạch rồi lùi lại, "
      "vốn không phải hành vi vượt đèn đỏ hoàn chỉnh. Sự thay đổi số lượng hồ sơ vì "
      "vậy phản ánh đúng lựa chọn thiết kế đã được cân nhắc, không phải lỗi hồi quy, "
      "và bộ kiểm thử đơn vị đã được cập nhật theo quyết định này."),

    H2("6.6. Kiểm thử cơ chế giao hàng không mất dữ liệu"),
    P("Yêu cầu NFR-03 và NFR-04 là hai yêu cầu khó kiểm chứng nhất, vì chúng nói về "
      "hành vi của hệ thống khi có sự cố chứ không phải khi hoạt động bình thường. Ba "
      "kịch bản đã được thực hiện trên hệ thống Docker đang chạy:"),
    UL([
        "Ngắt trung tâm khi node đang chạy. Dừng dịch vụ máy chủ trung tâm trong lúc "
        "pipeline vẫn xử lý video. Quan sát thấy vòng lặp xử lý không bị ảnh hưởng, các "
        "hồ sơ phát hiện được vẫn ghi vào kho outbox ở trạng thái chờ, bộ đếm hồ sơ "
        "chờ tăng đúng bằng số vi phạm mới. Khi khởi động lại dịch vụ trung tâm, luồng "
        "gửi nền tự phát hiện kết nối đã thông và đẩy hết số hồ sơ tồn đọng.",
        "Khởi động lại container node khi còn hồ sơ chờ. Vì kho outbox được gắn trên "
        "volume riêng, các hồ sơ chờ từ lần chạy trước vẫn còn và được gửi tiếp ở lần "
        "khởi động sau. Node ghi rõ trong nhật ký số hồ sơ chờ kế thừa từ lần chạy "
        "trước.",
        "Gửi lại cùng một lô. Cố ý gửi hai lần một lô hồ sơ có cùng mã sự kiện. Phản "
        "hồi lần thứ hai cho thấy số bản ghi được chấp nhận bằng không và số bản ghi "
        "trùng bằng kích thước lô, đồng thời vẫn trả mã thành công. Kiểm tra trong cơ "
        "sở dữ liệu không thấy bản ghi nào bị nhân đôi, xác nhận ràng buộc duy nhất "
        "trên cột mã sự kiện hoạt động đúng ở cả hai phía.",
    ]),
    P("Kết quả của ba kịch bản này cho phép khẳng định tính chất quan trọng nhất của "
      "thiết kế ở mục 4.4: khi có sự cố mạng, hệ thống chuyển từ trạng thái giao hàng "
      "ngay sang trạng thái giao hàng trễ, chứ không chuyển sang trạng thái mất dữ "
      "liệu. Đây là khác biệt về chất chứ không phải về lượng so với cách gọi trực "
      "tiếp API khi phát hiện vi phạm."),
    P("Giới hạn của phép kiểm chứng này cũng cần nêu rõ: các kịch bản trên được thực "
      "hiện thủ công và quan sát bằng nhật ký cùng bộ đếm, chưa có ca kiểm thử tự động "
      "nào mô phỏng việc mất kết nối. Ngoài ra thời gian ngắt kết nối trong thử nghiệm "
      "chỉ ở mức vài phút, chưa đủ dài để kiểm chứng hành vi khi kho outbox phình ra "
      "rất lớn hoặc khi đĩa của node đầy."),

    H2("6.7. Kiểm thử bảo mật"),
    P("Kiểm thử bảo mật tập trung vào bốn nhóm nguy cơ chính của một hệ thống nhận dữ "
      "liệu từ nhiều nguồn và có lưu thông tin cá nhân là biển số phương tiện."),
    UL([
        "Thông tin xác thực bị lộ trong mã nguồn. Quét toàn bộ kho mã theo các mẫu khoá "
        "bí mật trên tất cả tệp Python, Java, TypeScript, YAML, tập lệnh shell và JSON: "
        "không còn giá trị bí mật nào bị viết cứng. Tệp biến môi trường thật đã được "
        "đưa vào danh sách loại trừ của Git, kho chỉ giữ tệp mẫu. Đây là kết quả của "
        "một đợt rà soát thực hiện trước đó, khi phát hiện mật khẩu cơ sở dữ liệu và "
        "khoá MinIO bị viết thẳng trong tệp cấu hình Docker Compose.",
        "Xác thực và phân quyền. Kiểm tra các trường hợp token sai, token hết hạn, token "
        "rác và thiếu token trên mọi endpoint được bảo vệ: tất cả trả 401. Kiểm tra ma "
        "trận vai trò như mục 6.2.3: trả 403 đúng chỗ. Thông báo lỗi đăng nhập dùng "
        "chung một nội dung cho cả trường hợp sai tên tài khoản và sai mật khẩu, nên "
        "không thể dò ra tài khoản nào tồn tại.",
        "API điều khiển của node biên. Gọi với token sai trả 401. Gọi vượt ba mươi yêu "
        "cầu mỗi phút từ một địa chỉ trả 429 kèm các tiêu đề cho biết giới hạn và thời "
        "điểm được gọi lại. Yêu cầu từ nguồn không nằm trong danh sách cho phép bị "
        "chặn ở tầng chia sẻ tài nguyên khác nguồn. Mọi phản hồi đều kèm tiêu đề không "
        "cho phép nhúng khung và không lưu bộ nhớ đệm.",
        "Câu lệnh SQL do trợ lý sinh ra. Bộ lọc chỉ chấp nhận câu lệnh truy vấn và chặn "
        "các từ khoá xoá, sửa, tạo, thu hồi quyền, đồng thời tự giới hạn số dòng trả "
        "về. Thử đặt câu hỏi dẫn tới câu lệnh nguy hiểm thì hệ thống trả lỗi và không "
        "thực thi.",
    ]),
    P("Hai vấn đề bảo mật còn tồn tại được ghi nhận thẳng thay vì bỏ qua. Thứ nhất, "
      "endpoint đăng nhập của máy chủ trung tâm chưa có giới hạn tốc độ, trong khi API "
      "điều khiển của node biên thì có; nghĩa là về lý thuyết vẫn có thể dò mật khẩu "
      "bằng cách thử nhiều lần. Thứ hai, token làm mới được lưu trong bộ nhớ cục bộ "
      "của trình duyệt thay vì trong cookie chỉ đọc được phía máy chủ, đánh đổi giữa "
      "rủi ro tấn công chéo trang và khả năng làm mới phiên liền mạch; quyết định này "
      "được ghi rõ trong bình luận của mô-đun quản lý phiên. Cả hai đều được đưa vào "
      "phần hạn chế ở Chương 7."),

    H2("6.8. Kết quả"),
    P("Bảng 6.7 tổng hợp kết quả của toàn bộ hoạt động kiểm thử. Nguyên tắc điền bảng "
      "là chỉ ghi Đạt khi có bằng chứng thực thi kèm theo; hạng mục nào chưa thực hiện "
      "thì ghi rõ là chưa thực hiện."),
    TB(["Nhóm kiểm thử", "Số ca hoặc phạm vi", "Kết quả", "Bằng chứng"], [
        ["Kiểm thử đơn vị logic", "124 ca trong 7 tệp", "124 trên 124 đạt, 1,09 giây",
         "Nhật ký pytest ngày 12/9/2026"],
        ["Tích hợp liên tục", "3 việc làm trên GitHub Actions",
         "Cả ba chạy xanh ở lần đẩy gần nhất",
         "Kiểm thử và lint cho Python, biên dịch cho Java, kiểm tra kiểu và dựng cho "
         "TypeScript"],
        ["Biên dịch tầng trung tâm", "45 tệp Java", "Thành công, không lỗi",
         "Biên dịch trong image Maven chính thức vì máy phát triển chỉ có môi trường "
         "chạy Java"],
        ["Kiểm tra kiểu và dựng tầng web", "35 tệp TypeScript", "Thành công, không lỗi",
         "Chạy trong việc làm của tích hợp liên tục"],
        ["Kiểm thử chức năng trên hệ thống thật", "9 luồng chính",
         "Đạt toàn bộ", "Thực hiện thủ công qua HTTP và giao diện, mục 6.2.2"],
        ["Kiểm thử phân quyền", "Ma trận 3 vai trò với các nhóm endpoint",
         "Đạt, trả đúng 401 và 403", "Mục 6.2.3"],
        ["Kiểm thử hồi quy video", "2 video, tổng 504 khung hình",
         "Đúng kỳ vọng: 1 hồ sơ và 0 hồ sơ",
         "Mục 6.5, Bảng 6.7, tệp kết quả JSON trong thư mục tạm"],
        ["Kiểm thử giao hàng bền vững", "3 kịch bản sự cố",
         "Đạt, không mất và không trùng hồ sơ", "Mục 6.6"],
        ["Kiểm thử bảo mật", "4 nhóm nguy cơ",
         "Đạt, còn 2 vấn đề đã ghi nhận", "Mục 6.7"],
        ["Đánh giá mô hình học sâu", "3 mô hình",
         "Đã đo; 1 mô hình tái lập được bằng lệnh, 2 mô hình lấy từ ghi nhận trước",
         "Bảng 6.3 và Bảng 6.4; tệp kết quả chuẩn hoá trong thư mục scripts"],
        ["Đo thời gian khâu biển số", "60 khung hình, 40 ảnh cắt",
         "Đã đo", "Bảng 6.5, tập lệnh đo chạy ngày 12/9/2026"],
        ["Đo tỉ lệ đọc đúng biển số Việt Nam", "—",
         "Chưa thực hiện vì thiếu video đúng miền", "Ghi nhận ở mục 6.4"],
        ["Kiểm thử tải của máy chủ trung tâm", "—", "Chưa thực hiện",
         "Cần công cụ tạo tải như k6; ghi nhận ở Chương 7"],
        ["Kiểm thử thành phần giao diện", "—", "Chưa thực hiện",
         "Chưa có bộ kiểm thử thành phần cho tầng web"],
    ], "Bảng 6.8. Tổng hợp kết quả kiểm thử"),
    P("Đánh giá chung về chất lượng. Phần logic nghiệp vụ của đồ án, tức những chỗ dễ "
      "sai nhất và khó phát hiện nhất khi chạy thật, đã được bao phủ bằng kiểm thử tự "
      "động chạy trong vài giây và chạy lại với mỗi lần sửa mã. Đây là nền tảng cho "
      "phép khẳng định hệ thống xét vi phạm đúng theo các quy tắc đã phân tích. Ba "
      "khoản trống lớn còn lại đều nằm ở phần cần hạ tầng hoặc cần dữ liệu ngoài tầm "
      "của một đồ án: kiểm thử tải cần môi trường có thể tạo hàng nghìn yêu cầu đồng "
      "thời, kiểm thử thành phần giao diện cần bộ công cụ dựng riêng, còn đo tỉ lệ đọc "
      "biển số cần bộ video giao thông Việt Nam có gán nhãn độc lập. Việc xác định rõ "
      "ba khoản trống này, thay vì ước lượng chúng, là cách đồ án giữ cho các kết luận "
      "ở Chương 7 nằm trong phạm vi bằng chứng có thật."),

    PB,

    # ============================================================ CHƯƠNG 7
    H1("CHƯƠNG 7. KẾT LUẬN"),

    H2("7.1. Kết quả đạt được"),
    P("Đối chiếu lại với bảy mục tiêu cụ thể đặt ra ở mục 1.2, đồ án đã hoàn thành cả "
      "bảy, trong đó sáu mục tiêu hoàn thành ở mức có số liệu kiểm chứng và một mục "
      "tiêu hoàn thành ở mức chức năng hoạt động được nhưng chưa đo được độ chính xác. "
      "Bảng 7.1 trình bày phép đối chiếu này."),
    TB(["Mục tiêu đặt ra", "Kết quả thực hiện", "Mức độ"], [
        ["Xây dựng pipeline thị giác thời gian thực tại node biên",
         "Hoàn thành. Pipeline chạy đủ các khâu phát hiện, theo dõi, phân loại đèn, "
         "xét vi phạm và nhận dạng biển số; tốc độ xử lý đo được khoảng 37 khung hình "
         "mỗi giây trên node chạy trong Docker, riêng khâu phát hiện đạt 73,4 khung "
         "hình mỗi giây ở chế độ nửa chính xác", "Đạt, có đo"],
        ["Huấn luyện và đánh giá ba mô hình chuyên biệt",
         "Hoàn thành. Mô hình phát hiện phương tiện đạt mAP50 bằng 0,9436 so với "
         "0,0101 của mô hình gốc trên cùng tập kiểm chứng; mô hình phát hiện biển số "
         "đạt 0,993; mô hình phân loại đèn đạt tỉ lệ đúng cao nhất trên tập kiểm chứng "
         "và được bổ sung cơ chế hợp nhất bằng chứng", "Đạt, có đo"],
        ["Nhận dạng biển số Việt Nam có kiểm tra cấu trúc",
         "Hoàn thành phần mềm: mô hình đọc được chuỗi ký tự trên toàn bộ mẫu thử với độ "
         "tin cậy trung bình khoảng 0,9, bộ kiểm tra cấu trúc loại đúng các biển không "
         "theo quy định Việt Nam. Chưa đo được tỉ lệ đọc đúng trên biển số Việt Nam vì "
         "thiếu dữ liệu đúng miền", "Đạt chức năng, thiếu số đo"],
        ["Cơ chế giao hàng bền vững, không mất và không trùng hồ sơ",
         "Hoàn thành và kiểm chứng bằng ba kịch bản sự cố. Kho outbox ghi trước khi gọi "
         "mạng, lô gửi có tính luỹ đẳng nhờ ràng buộc duy nhất trên mã sự kiện ở cả hai "
         "phía", "Đạt, có kiểm chứng"],
        ["Máy chủ trung tâm và bảng điều khiển web hoàn chỉnh",
         "Hoàn thành. Trung tâm 3.525 dòng Java với 31 endpoint, web 6.547 dòng "
         "TypeScript với chín tuyến đường và mười ba thành phần; đủ các nghiệp vụ thống "
         "kê, tra cứu, duyệt, xem camera trực tiếp và hiệu chuẩn từ xa", "Đạt"],
        ["Trợ lý hỏi dữ liệu bằng tiếng Việt",
         "Hoàn thành. Sinh câu lệnh truy vấn từ câu hỏi tự nhiên, có bộ lọc chỉ cho "
         "phép truy vấn và giới hạn số dòng, hiển thị cả câu lệnh đã sinh để người dùng "
         "kiểm tra. Phụ thuộc dịch vụ bên ngoài nên chưa có phương án dự phòng khi "
         "mất kết nối", "Đạt, có ràng buộc"],
        ["Kiểm thử và vận hành bằng một lệnh",
         "Hoàn thành phần kiểm thử tự động với 124 ca đạt và ba việc làm tích hợp liên "
         "tục chạy xanh; toàn hệ thống dựng bằng một lệnh với năm dịch vụ có kiểm tra "
         "sức khoẻ. Phần kiểm thử tích hợp vẫn thực hiện thủ công", "Đạt phần lớn"],
    ], "Bảng 7.1. Đối chiếu mục tiêu và kết quả đạt được"),
    P("Ngoài bảy mục tiêu, đồ án đạt được ba kết quả có ý nghĩa về mặt phương pháp, "
      "đều xuất phát từ sự cố gặp phải trong quá trình làm chứ không nằm trong kế "
      "hoạch ban đầu."),
    UL([
        "Xác định được bằng số liệu rằng việc huấn luyện lại mô hình trên dữ liệu đúng "
        "miền là điều kiện bắt buộc chứ không phải bước cải thiện tuỳ chọn. Chênh lệch "
        "gần chín mươi bốn lần về mAP50 giữa mô hình huấn luyện riêng và mô hình gốc, "
        "trong khi tốc độ suy luận gần như bằng nhau, là bằng chứng định lượng rõ ràng "
        "nhất của đồ án và cũng là kết quả có thể dùng lại cho các bài toán thị giác "
        "khác tại Việt Nam.",
        "Thiết kế được cơ chế tách rời phát hiện khỏi giao hàng, biến một hệ thống có "
        "nguy cơ mất dữ liệu thành hệ thống chỉ có thể giao hàng trễ. Mẫu thiết kế này "
        "không đặc thù cho bài toán giao thông và có thể áp dụng cho bất kỳ hệ thống "
        "phân tán nào có node biên hoạt động trong điều kiện mạng không ổn định.",
        "Xây dựng được tầng chất lượng theo dõi nằm giữa bộ theo dõi và logic nghiệp vụ, "
        "gồm quỹ đạo, vận tốc tính theo đơn vị thời gian thực thay vì theo số khung "
        "hình, độ tin cậy làm mượt và bỏ phiếu nhãn. Tầng này giải quyết đúng vấn đề "
        "của camera tốc độ khung hình thấp, nơi một phương tiện dịch chuyển hàng chục "
        "điểm ảnh giữa hai khung hình liên tiếp.",
    ]),
    P("Về quy mô, sản phẩm của đồ án là một hệ thống phân tán ba tầng với 18.386 dòng "
      "mã trên ba ngôn ngữ lập trình khác nhau, bốn mô hình học sâu, hai hệ thống lưu "
      "trữ, ba mươi mốt endpoint ở trung tâm và mười bốn endpoint ở node biên, cùng bộ "
      "tài liệu gồm tài liệu API, hướng dẫn triển khai, báo cáo kiểm thử và hai mươi "
      "sáu sơ đồ. Toàn bộ chạy được trên một máy trạm có một card đồ hoạ phổ thông."),

    H2("7.2. Hạn chế"),
    P("Đồ án còn những hạn chế sau, được xếp theo mức độ ảnh hưởng tới khả năng đưa hệ "
      "thống vào vận hành thực tế."),
    TB(["Nhóm hạn chế", "Mô tả cụ thể", "Ảnh hưởng"], [
        ["Phạm vi kiểm chứng thực địa",
         "Toàn bộ thử nghiệm chạy trên tệp video, chủ yếu phát lặp lại, với một node "
         "biên duy nhất. Chưa vận hành đồng thời nhiều node, chưa kết nối camera thật "
         "qua giao thức phát trực tiếp trong thời gian dài.",
         "Cao. Chưa chứng minh được khả năng mở rộng theo số nút giao và độ ổn định "
         "khi chạy nhiều ngày liên tục"],
        ["Chất lượng nhận dạng biển số trên dữ liệu đúng miền",
         "Chưa có video giao thông Việt Nam nào đủ rõ và đủ nhiều để đo tỉ lệ đọc đúng. "
         "Số liệu hiện có chỉ chứng minh mô hình đọc được chuỗi ký tự và bộ lọc cấu "
         "trúc hoạt động đúng.",
         "Cao. Biển số là trường thông tin quan trọng nhất của hồ sơ xử phạt, thiếu số "
         "đo này thì chưa thể ước lượng tỉ lệ hồ sơ phải bổ sung thủ công"],
        ["Điều kiện ánh sáng khó",
         "Chưa kiểm chứng quy mô lớn với mưa đêm, ngược sáng gắt, đèn pha chiếu thẳng "
         "vào ống kính. Dữ liệu huấn luyện và video thử nghiệm đều là điều kiện ban "
         "ngày tương đối thuận lợi.",
         "Cao. Nút giao thực tế hoạt động cả ngày đêm và trong mọi thời tiết"],
        ["Kiểm thử tự động còn thiếu ở hai tầng",
         "Tầng trung tâm chưa có bộ kiểm thử tích hợp tự động, tầng web chưa có kiểm "
         "thử thành phần. Mọi kiểm tra tích hợp đều làm thủ công.",
         "Trung bình. Mỗi lần sửa mã phải tự chạy lại các bước kiểm tra, dễ bỏ sót hồi "
         "quy"],
        ["Chưa đo khả năng chịu tải",
         "Không có số liệu về số yêu cầu đồng thời mà trung tâm xử lý được, cũng như độ "
         "trễ phản hồi theo số node gửi dữ liệu. Các phân tích về khả năng mở rộng "
         "hiện chỉ dựa trên lập luận kiến trúc.",
         "Trung bình. Không xác định được điểm cần nâng cấp khi số node tăng"],
        ["Bảo mật còn hai khoảng trống",
         "Endpoint đăng nhập của trung tâm chưa giới hạn tốc độ. Token làm mới lưu trong "
         "bộ nhớ cục bộ của trình duyệt thay vì cookie chỉ đọc phía máy chủ.",
         "Trung bình. Cần xử lý trước khi mở hệ thống ra mạng công cộng"],
        ["Hiệu chuẩn phụ thuộc thao tác thủ công",
         "Vùng đèn và vạch dừng cố định sau khi hiệu chuẩn. Khi camera bị lệch do gió "
         "hoặc va chạm, hệ thống không tự phát hiện mà vẫn tiếp tục xét vi phạm theo "
         "toạ độ cũ.",
         "Trung bình. Có thể sinh hồ sơ sai hàng loạt mà không có cảnh báo"],
        ["Phụ thuộc dịch vụ bên ngoài",
         "Trợ lý hỏi dữ liệu cần khoá API của nhà cung cấp mô hình ngôn ngữ và cần kết "
         "nối mạng. Không có phương án dự phòng chạy cục bộ.",
         "Thấp. Chức năng này là tiện ích bổ sung, không nằm trên luồng nghiệp vụ chính"],
        ["Quản lý người dùng chưa hoàn thiện",
         "Chưa có giao diện thêm sửa xoá tài khoản; tài khoản quản trị được tạo tự động "
         "từ biến môi trường. Token truy cập chưa có danh sách thu hồi, chỉ tự hết hạn "
         "sau mười hai giờ.",
         "Thấp. Đủ dùng cho quy mô vận hành nội bộ của đồ án"],
        ["Quản lý lược đồ cơ sở dữ liệu",
         "Lược đồ do khung ứng dụng tự cập nhật thay vì dùng công cụ quản lý phiên bản "
         "di trú. Chấp nhận được ở quy mô đồ án nhưng không phù hợp khi dữ liệu sản "
         "xuất cần thay đổi lược đồ có kiểm soát.",
         "Thấp ở hiện tại, sẽ thành trung bình khi có dữ liệu thật cần bảo toàn"],
    ], "Bảng 7.2. Hạn chế của đồ án theo mức độ ảnh hưởng"),
    P("Trong các hạn chế trên, hai hạn chế đầu tiên là căn bản nhất và có liên hệ nhân "
      "quả với nhau: vì chưa có dữ liệu thực địa đúng miền nên vừa không kiểm chứng "
      "được hệ thống trong điều kiện vận hành thật, vừa không đo được chất lượng nhận "
      "dạng biển số. Đây là giới hạn về nguồn lực thu thập dữ liệu chứ không phải giới "
      "hạn về thiết kế, và là việc cần làm đầu tiên nếu muốn phát triển tiếp."),

    H2("7.3. Hướng phát triển"),
    P("Từ các hạn chế đã nêu, hướng phát triển được sắp xếp theo thứ tự ưu tiên: nhóm "
      "thứ nhất giải quyết khoảng trống về bằng chứng, nhóm thứ hai mở rộng năng lực "
      "nghiệp vụ, nhóm thứ ba hoàn thiện vận hành."),
    P("Nhóm một, thu hẹp khoảng trống về dữ liệu và kiểm chứng:", "BodyBold"),
    UL([
        "Xây dựng bộ dữ liệu video giao thông Việt Nam có gán nhãn độc lập, tối thiểu "
        "vài nghìn khung hình trải trên các điều kiện sáng khác nhau, để đo được tỉ lệ "
        "phát hiện đúng, tỉ lệ hồ sơ giả và tỉ lệ đọc đúng biển số. Đây là tiền đề cho "
        "mọi cải tiến tiếp theo vì không có số đo thì không biết thay đổi nào là tốt "
        "hơn.",
        "Kết nối camera thật qua giao thức phát trực tiếp và cho chạy liên tục nhiều "
        "ngày, ghi nhận tự động các trường hợp mất kết nối, tràn bộ nhớ, lệch khung "
        "hình. Chế độ bỏ khung hình để bám thời gian thực đã có sẵn trong cài đặt, chỉ "
        "cần bật và quan sát.",
        "Tự phát hiện khi camera bị lệch: so sánh vị trí vùng đèn và vạch dừng hiện tại "
        "với ảnh nền tham chiếu, cảnh báo khi độ lệch vượt ngưỡng để người vận hành "
        "biết cần hiệu chuẩn lại.",
    ]),
    P("Nhóm hai, mở rộng năng lực nghiệp vụ:", "BodyBold"),
    UL([
        "Mở rộng sang các loại vi phạm khác trên cùng pipeline hiện có. Nền tảng theo "
        "dõi, vạch ảo và mô hình chuyển động đã đủ để xét thêm hành vi đi ngược chiều, "
        "lấn làn và dừng đỗ sai quy định; riêng hành vi chạy quá tốc độ cần thêm bước "
        "quy đổi từ điểm ảnh sang mét bằng phép chiếu mặt đường.",
        "Nâng chất lượng nhận dạng biển số bằng cách huấn luyện lại trên dữ liệu biển "
        "số Việt Nam và chuyển sang chạy trên hạ tầng suy luận tối ưu hơn, đồng thời "
        "giải quyết xung đột phiên bản thư viện tính toán để khâu này chạy ổn định trên "
        "card đồ hoạ.",
        "Xuất biên bản xử phạt từ hồ sơ đã duyệt theo mẫu quy định, kèm ảnh bằng chứng "
        "và các trường truy vết; cung cấp giao diện lập trình để đối soát với hệ thống "
        "quản lý phương tiện.",
        "Giải thích được quyết định của hệ thống: lưu thêm ảnh cắt biển số, chuỗi ký tự "
        "thô trước khi sửa và lý do bộ lọc chấp nhận hay loại, để người duyệt có đủ căn "
        "cứ khi kết luận hồ sơ.",
    ]),
    P("Nhóm ba, hoàn thiện chất lượng và vận hành:", "BodyBold"),
    UL([
        "Bổ sung bộ kiểm thử tích hợp tự động cho tầng trung tâm bằng khung kiểm thử "
        "Java với cơ sở dữ liệu khởi tạo tạm thời, và bộ kiểm thử thành phần cho tầng "
        "web; đưa kiểm thử tải vào quy trình bằng công cụ tạo tải chuyên dụng để có số "
        "liệu về băng tải và độ trễ.",
        "Xử lý hai khoảng trống bảo mật: thêm giới hạn tốc độ cho endpoint đăng nhập và "
        "cân nhắc khoá tài khoản tạm thời sau nhiều lần sai; chuyển token làm mới sang "
        "cookie chỉ đọc phía máy chủ nếu chấp nhận được thay đổi trong cơ chế làm mới "
        "phiên.",
        "Chuyển sang quản lý lược đồ cơ sở dữ liệu bằng công cụ di trú có phiên bản, "
        "thay cho cơ chế tự cập nhật hiện nay.",
        "Thêm giám sát tập trung: thu thập chỉ số từ cả ba tầng, dựng bảng theo dõi và "
        "cấu hình cảnh báo khi node ngừng gửi dữ liệu hoặc tỉ lệ hồ sơ bị từ chối tăng "
        "bất thường.",
        "Kiểm chứng quy trình sao lưu bằng cách phục hồi thử định kỳ, và đặt giới hạn "
        "tài nguyên cho từng container để một dịch vụ không chiếm hết bộ nhớ của máy.",
        "Về kiến trúc, nếu cần phục vụ nhiều nút giao trên nhiều máy: thêm tầng cân "
        "bằng cho endpoint nhận dữ liệu, bật mã hoá cho giao tiếp nội bộ giữa các máy, "
        "và cân nhắc tách kho lưu trữ đối tượng sang dịch vụ chuyên dụng có nhân bản.",
    ]),
    P("Các hướng trên đều khả thi trên nền thiết kế hiện tại vì đồ án đã giữ ranh giới "
      "tầng rõ và dùng giao tiếp qua hợp đồng dữ liệu. Việc thêm loại vi phạm mới, thêm "
        "một tầng lưu trữ hay thay nhà cung cấp mô hình ngôn ngữ đều không đòi hỏi viết "
      "lại các tầng còn lại."),

    PB,

    # ================================================= TÀI LIỆU THAM KHẢO
    H1("TÀI LIỆU THAM KHẢO"),
    P("Tài liệu tiếng Việt", "BodyBold"),
    PL([
        "[1]  Bộ Giao thông Vận tải (2023), Thông tư 24/2023/TT-BGTVT quy định về cấp, "
        "thu hồi đăng ký, biển số xe cơ giới, Hà Nội.",
        "[2]  Chính phủ (2021), Nghị định 100/2019/NĐ-CP quy định xử phạt vi phạm hành "
        "chính trong lĩnh vực giao thông đường bộ và đường sắt, cùng các văn bản sửa đổi "
        "bổ sung, Hà Nội.",
        "[3]  Cục Cảnh sát giao thông, Bộ Công an (2025), Thông tin về hệ thống camera "
        "ứng dụng trí tuệ nhân tạo và Trung tâm thông tin chỉ huy, theo nội dung đăng "
        "trên báo VietnamNet ngày 25 tháng 9 năm 2025 và báo VOV Giao thông ngày 17 "
        "tháng 7 năm 2025.",
        "[4]  Ủy ban An toàn giao thông Quốc gia (2025), Số liệu thống kê tai nạn giao "
        "thông đường bộ, Hà Nội.",
    ]),
    P("Sách và bài báo khoa học", "BodyBold"),
    PL([
        "[5]  Zhang Y., Sun P., Jiang Y., Nie D., Fakhoury R., Lyu S. (2022), "
        "\u201cByteTrack: Multi-Object Tracking by Associating Every Detection Box\u201d, "
        "European Conference on Computer Vision, Springer, trang 1\u201321.",
        "[6]  Khanam R., Hussain M. (2024), \u201cYOLOv11: An Overview of the Key "
        "Architectural Enhancements\u201d, arXiv:2410.17725.",
        "[7]  Bewley A., Ge Z., Ott L., Ramos F., Upcroft B. (2016), \u201cSimple Online "
        "and Realtime Tracking\u201d, IEEE International Conference on Image Processing.",
        "[8]  Richardson C. (2018), Microservices Patterns: With Examples in Java, "
        "Manning Publications (chương 4 về mẫu quản lý giao dịch phân tán, trong đó có "
        "mẫu hộp thư đi).",
        "[9]  Møgelmose A., Trivedi M. M., Moeslund T. B. (2012), \u201cVision-Based "
        "Traffic Sign Detection and Analysis for Intelligent Driver Assistance "
        "Systems\u201d, IEEE Transactions on Intelligent Transportation Systems.",
        "[10] Jones M. B., Bradley J., Sakimura N. (2015), JSON Web Token (JWT), "
        "RFC 7519, Internet Engineering Task Force.",
        "[11] Fielding R. T. (2000), Architectural Styles and the Design of Network-based "
        "Software Architectures, luận án tiến sĩ, Đại học California, Irvine.",
    ]),
    P("Tài liệu kỹ thuật và thư viện mã nguồn mở", "BodyBold"),
    PL([
        "[12] Ultralytics (2026), YOLO tài liệu chính thức, "
        "https://docs.ultralytics.com.",
        "[13] Ultralytics (2026), supervision: thư viện công cụ thị giác máy tính, tài "
        "liệu bộ theo dõi ByteTrack, https://supervision.roboflow.com.",
        "[14] fast-plate-ocr (2025), thư viện nhận dạng ký tự biển số, mã nguồn công khai "
        "trên GitHub.",
        "[15] Spring Boot 3.3 Reference Documentation, "
        "https://docs.spring.io/spring-boot/docs/3.3.x/reference/html/.",
        "[16] Spring Security Reference Documentation, phần bộ lọc và biểu thức phân "
        "quyền, https://docs.spring.io/spring-security/reference/.",
        "[17] Next.js 16 Documentation, phần App Router, middleware và viết lại đường "
        "dẫn, https://nextjs.org/docs.",
        "[18] PostgreSQL 16 Documentation, phần chỉ mục và ràng buộc duy nhất, "
        "https://www.postgresql.org/docs/16/.",
        "[19] MinIO Object Storage Documentation, "
        "https://min.io/docs/minio/linux/index.html.",
        "[20] Docker Compose Specification, phần kiểm tra sức khoẻ, khai báo thiết bị "
        "và volume, https://docs.docker.com/compose/.",
        "[21] FFmpeg Documentation, phần bộ ghép luồng HLS, https://ffmpeg.org/.",
        "[22] hls.js Documentation, thư viện phát luồng HLS trong trình duyệt, "
        "https://github.com/video-dev/hls.js.",
        "[23] OWASP Foundation (2025), Cheat Sheet Series, các phần xác thực, quản lý "
        "phiên và bảo mật Docker, https://cheatsheetseries.owasp.org.",
    ]),
    P("Nguồn dữ liệu và công cụ hỗ trợ", "BodyBold"),
    PL([
        "[24] Bộ dữ liệu ảnh biển số xe dùng để huấn luyện mô hình phát hiện biển số, thu "
        "thập từ các nguồn công khai có giấy phép phù hợp.",
        "[25] Bộ dữ liệu tín hiệu đèn giao thông dùng để huấn luyện mô hình phân loại "
        "màu đèn.",
        "[26] Kroki và PlantUML, dịch vụ dựng sơ đồ từ mô tả dạng văn bản, dùng để sinh "
        "các sơ đồ trong đồ án, https://kroki.io và https://plantuml.com.",
    ]),
    P("Ghi chú về cách trích dẫn: các tài liệu từ số [12] đến số [23] là tài liệu kỹ "
      "thuật chính thức của công cụ mà đồ án sử dụng, được tham chiếu ở thời điểm hoàn "
      "thiện đồ án là tháng 9 năm 2026. Các thông tin về hệ thống camera ứng dụng trí "
      "tuệ nhân tạo của lực lượng cảnh sát giao thông ở mục 2.2 được lấy từ báo chí "
      "chính thống trong nước, dẫn lại phát biểu của lãnh đạo Cục Cảnh sát giao thông; "
      "đồ án không tiếp cận được tài liệu kỹ thuật nội bộ của hệ thống đó nên chỉ dùng "
      "ở mức đối chiếu định tính."),
]
