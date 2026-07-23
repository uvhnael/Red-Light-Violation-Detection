Dưới đây là danh sách chi tiết các công việc cần làm cho Central Server, được chia thành các module cốt lõi để bạn dễ dàng provisioning (cấp phát) và quản lý:

1. Dựng Core Backend & API Gateway
Đây là "cửa ngõ" đón nhận dữ liệu từ tất cả các Edge Nodes gửi lên.

Xây dựng RESTful API: Thiết lập các endpoint (ví dụ: POST /api/v1/violations) để nhận payload JSON từ Celery worker của Edge Node.

Công nghệ đề xuất: Sử dụng Java Spring Boot 3 để xây dựng bộ core vững chắc. Spring Boot xử lý concurrency (đồng thời) cực tốt khi có hàng trăm node cùng đẩy data về.

Nhiệm vụ chính: Validate (kiểm tra tính hợp lệ) của gói tin, bóc tách thông tin vi phạm, và chuyển tiếp các file media (ảnh, video) sang hệ thống lưu trữ.

2. Thiết kế Cơ sở dữ liệu & Storage 
Dữ liệu vi phạm cần được lưu trữ có tổ chức và an toàn.

Relational Database (Cơ sở dữ liệu quan hệ): Triển khai PostgreSQL để lưu trữ metadata (Biển số, ID camera, Thời gian, Lỗi vi phạm, Trạng thái xử lý).

Object Storage (Lưu trữ file): Không nên lưu ảnh/video trực tiếp vào thư mục local của server backend. Hãy deploy một container MinIO (tương thích chuẩn S3). Backend sẽ upload file lên MinIO, sau đó lấy đường dẫn URL lưu vào Database.

4. Triển khai Advanced AI Engine (Phân tích chuyên sâu)
Server là nơi "não bộ" hoạt động mạnh nhất với các model không bị giới hạn về phần cứng.

AI Đối soát (MMCR): Dựng một service nhận diện Hãng xe, Dòng xe, Màu sắc (có thể dùng Python/FastAPI) để đối chiếu với kết quả OCR từ Edge, nhằm phát hiện biển số giả.

AI Phân tích dữ liệu (LLM): Khởi chạy một local server Ollama để host các mô hình ngôn ngữ lớn. Bạn có thể xây dựng tính năng Text-to-SQL, cho phép người dùng gõ: "Liệt kê các xe vượt đèn đỏ tại ngã tư X hôm qua", Ollama sẽ dịch câu này thành query SQL chọc vào PostgreSQL và trả về kết quả.

5. Phát triển Frontend Dashboard (Giao diện Quản trị)
Giao diện trực quan để người dùng cuối tương tác với hệ thống.

Tech Stack: Sử dụng Next.js kết hợp Tailwind CSS để build một Single Page Application (SPA) mượt mà.

Tính năng chính:

Bảng điều khiển (Dashboard) hiển thị biểu đồ thống kê vi phạm theo ngày/tháng (lấy data từ API Spring Boot).

Giao diện duyệt hồ sơ vi phạm (Xem ảnh toàn cảnh, ảnh crop biển số, video 5s).

Tính năng xác nhận thủ công (Human-in-the-loop) để CSGT duyệt lại các case AI nhận diện có độ tự tin (confidence) thấp.
