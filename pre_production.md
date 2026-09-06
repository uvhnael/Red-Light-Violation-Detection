Bạn là Senior Software Architect + DevOps Engineer + Technical Writer + Software Engineering Report Reviewer.

Hãy kiểm tra TOÀN BỘ project hiện tại trước khi đưa lên PRODUCTION.

Bối cảnh kiến trúc hệ thống:

- Web application được tổ chức theo kiến trúc 3 tầng:
  1. Presentation Layer / Frontend
  2. Business Logic Layer / Backend
  3. Data Access / Database Layer

- Hệ thống có CENTRAL SERVER làm server trung tâm.
- Có nhiều EDGE NODE / EDGE DEVICE kết nối về Central Server.
- Central Server chịu trách nhiệm xử lý dữ liệu, API, authentication, business logic, lưu trữ dữ liệu và giao tiếp với các Edge Node.
- Edge Node thu thập/xử lý dữ liệu tại biên và gửi dữ liệu về Central Server.
- Web frontend dùng để quản trị, giám sát và hiển thị dữ liệu.
- Cần chuẩn bị hệ thống đủ tốt để deploy production và đồng thời dùng làm đồ án kỹ sư phần mềm.

==================================================
PHASE 1 — PHÂN TÍCH TOÀN BỘ SOURCE CODE
==================================================

Đầu tiên KHÔNG được tự ý sửa code.

Hãy scan toàn bộ repository và xác định:

1. Frontend
2. Backend
3. Database
4. API
5. Authentication / Authorization
6. Central Server
7. Edge Node
8. Communication giữa Edge Node ↔ Central Server
9. Configuration
10. Docker / Docker Compose
11. Environment variables
12. Logging
13. Monitoring
14. Error handling
15. Security
16. Deployment
17. Testing
18. Documentation

Tạo một architecture map thể hiện:

User
 ↓
Frontend
 ↓
API Gateway / Backend
 ↓
Business Logic
 ↓
Data Access
 ↓
Database

Đồng thời thể hiện:

Edge Node 1 ─┐
Edge Node 2 ─┤
Edge Node 3 ─┼──> Central Server ──> Database
Edge Node N ─┘
                     ↑
                  Backend API
                     ↑
                  Frontend

Nếu kiến trúc thực tế khác sơ đồ trên thì PHẢI dựa vào source code để vẽ lại kiến trúc chính xác.

Không được tự đoán.

==================================================
PHASE 2 — KIỂM TRA KIẾN TRÚC 3 TẦNG
==================================================

Kiểm tra frontend/backend/database có thực sự tách biệt hay không.

Kiểm tra:

- Presentation Layer
- Controller / API Layer
- Service / Business Layer
- Repository / DAO
- Entity / Model
- DTO
- Database

Phát hiện:

- Business logic nằm trong Controller
- SQL nằm trực tiếp trong Controller
- Frontend truy cập database trực tiếp
- Service phụ thuộc ngược Controller
- Circular dependency
- Duplicate business logic
- God class
- God component
- Hard-coded configuration
- Code smell
- Vi phạm Separation of Concerns

Đề xuất và sửa nếu cần.

Sau mỗi thay đổi phải giải thích:

- Vấn đề
- Nguyên nhân
- Giải pháp
- File đã thay đổi
- Tại sao giải pháp phù hợp production

==================================================
PHASE 3 — KIỂM TRA CENTRAL SERVER
==================================================

Audit Central Server.

Kiểm tra:

- API architecture
- Authentication
- Authorization
- User management
- Edge node registration
- Edge node identification
- API authentication giữa Edge Node và Central Server
- Request validation
- Rate limiting
- Error handling
- Retry
- Timeout
- Connection pooling
- Database transaction
- Logging
- Health check
- Graceful shutdown
- Configuration
- Secret management

Đặc biệt kiểm tra khả năng:

Edge Node mất mạng
→ lưu dữ liệu local
→ reconnect
→ retry
→ đồng bộ dữ liệu
→ tránh duplicate dữ liệu

Nếu chưa có cơ chế này, hãy đề xuất implementation phù hợp.

==================================================
PHASE 4 — KIỂM TRA EDGE NODE
==================================================

Audit toàn bộ Edge Node.

Kiểm tra:

- Device identity
- Registration
- Authentication
- Data collection
- Local processing
- Local storage/cache
- Communication protocol
- Retry mechanism
- Offline mode
- Reconnection
- Heartbeat
- Health status
- Firmware/software version
- Configuration
- Remote update nếu phù hợp
- Security

Thiết kế lifecycle:

BOOT
 ↓
INITIALIZE
 ↓
CONNECT CENTRAL SERVER
 ↓
AUTHENTICATE
 ↓
REGISTER
 ↓
COLLECT DATA
 ↓
SEND DATA
 ↓
WAIT
 ↓
ERROR?
 ├── NO → CONTINUE
 └── YES → RETRY / OFFLINE BUFFER
                 ↓
              RECONNECT

Nếu project hiện tại chưa có các thành phần trên thì đánh giá mức độ cần thiết trước khi implementation.

==================================================
PHASE 5 — KIỂM TRA COMMUNICATION
==================================================

Phân tích chính xác Edge Node ↔ Central Server đang sử dụng:

- HTTP/HTTPS
- REST API
- WebSocket
- MQTT
- TCP
- UDP
- hoặc protocol khác.

Đánh giá:

- Security
- Reliability
- Latency
- Scalability
- Retry
- Timeout
- Message format
- Versioning
- Authentication
- Duplicate message
- Data integrity

Đề xuất protocol tốt nhất cho project nhưng KHÔNG thay đổi kiến trúc lớn nếu không cần thiết.

==================================================
PHASE 6 — DATABASE AUDIT
==================================================

Kiểm tra:

- Database schema
- Primary key
- Foreign key
- Index
- Unique constraint
- Normalization
- Relationship
- Audit fields
- Timestamp
- Soft delete nếu cần
- Transaction
- Migration
- Backup
- Restore

Tìm:

- Missing index
- N+1 query
- Duplicate data
- Inconsistent naming
- Bad relationship
- Missing constraints

Đề xuất migration nếu cần.

==================================================
PHASE 7 — SECURITY AUDIT
==================================================

Kiểm tra toàn bộ security trước production:

- Authentication
- Authorization
- JWT/session
- Password hashing
- CORS
- CSRF nếu phù hợp
- XSS
- SQL Injection
- Command Injection
- SSRF
- Path traversal
- Input validation
- File upload
- Rate limiting
- Secrets
- API keys
- Database credentials
- HTTPS
- Security headers
- Docker security
- Environment variables

Tìm toàn bộ:

- API key hard-code
- Password hard-code
- Secret commit trong Git
- Development configuration dùng production
- Debug mode
- Swagger exposure không phù hợp
- Stack trace leak

Không được expose secret trong report.

==================================================
PHASE 8 — PRODUCTION READINESS
==================================================

Tạo checklist:

[ ] Build production
[ ] Environment configuration
[ ] Docker image
[ ] Docker Compose
[ ] Database migration
[ ] Database backup
[ ] Health check
[ ] Logging
[ ] Monitoring
[ ] Error tracking
[ ] Reverse proxy
[ ] HTTPS
[ ] CORS
[ ] Authentication
[ ] Authorization
[ ] Rate limiting
[ ] Resource limits
[ ] Restart policy
[ ] Persistent volumes
[ ] Backup
[ ] Restore test
[ ] CI/CD
[ ] Rollback strategy

Chấm điểm production readiness từ 0–100.

Phân loại:

CRITICAL
HIGH
MEDIUM
LOW

Không được kết luận "production ready" nếu còn CRITICAL issue.

==================================================
PHASE 9 — TESTING
==================================================

Kiểm tra testing hiện tại.

Nếu thiếu, bổ sung test phù hợp:

Backend:
- Unit test
- Service test
- Controller/API test
- Repository test
- Authentication test
- Integration test

Frontend:
- Component test
- API integration test
- Critical user flow

Edge Node:
- Connection test
- Retry test
- Offline test
- Reconnection test
- Data synchronization test

System:
- End-to-end test
- Load test nếu cần

Tạo test report.

==================================================
PHASE 10 — PERFORMANCE & SCALABILITY
==================================================

Đánh giá:

- API latency
- Database performance
- Connection pool
- Concurrent users
- Number of Edge Nodes
- Request throughput
- Memory
- CPU
- Network bandwidth

Trả lời:

"Kiến trúc hiện tại có thể hỗ trợ khoảng bao nhiêu Edge Node?"

Nếu không đủ dữ liệu để ước lượng thì nói rõ cần benchmark nào.

Không được bịa số liệu.

==================================================
PHASE 11 — SỬA CODE
==================================================

Sau khi audit xong:

1. Liệt kê toàn bộ issue.
2. Ưu tiên CRITICAL/HIGH.
3. Sửa các issue thực sự cần thiết.
4. Không refactor lan man.
5. Không phá API đang hoạt động nếu không cần.
6. Không thay đổi technology stack nếu không có lý do.
7. Chạy build.
8. Chạy test.
9. Kiểm tra lint/type check.
10. Kiểm tra Docker build nếu project dùng Docker.
11. Kiểm tra production configuration.

Sau khi sửa phải tạo CHANGELOG kỹ thuật.

==================================================
PHASE 12 — VẼ KIẾN TRÚC
==================================================

Tạo các sơ đồ cần thiết bằng Mermaid hoặc PlantUML.

BẮT BUỘC xem xét các sơ đồ:

1. System Architecture Diagram
2. 3-Tier Architecture Diagram
3. Deployment Diagram
4. Component Diagram
5. Use Case Diagram
6. Sequence Diagram
7. ERD / Database Diagram
8. Edge Node ↔ Central Server communication diagram
9. Authentication flow
10. Data flow diagram

Chỉ tạo những diagram thực sự phù hợp.

Diagram phải phản ánh SOURCE CODE THỰC TẾ.

==================================================
PHASE 13 — USE CASE
==================================================

Xác định Actors thực tế.

Ví dụ:

- Administrator
- User
- Edge Node
- Central Server

Nhưng phải dựa trên hệ thống thực tế.

Xây dựng:

Actor
→ Use Case

Phân biệt:

- Primary actor
- Secondary actor

Xác định:

- include
- extend
- generalization

Không tạo use case chỉ để làm báo cáo đẹp.

==================================================
PHASE 14 — SEQUENCE DIAGRAM
==================================================

Tạo sequence diagram cho các flow quan trọng.

Ít nhất xem xét:

1. Login
2. User truy cập dashboard
3. Edge Node đăng ký Central Server
4. Edge Node gửi dữ liệu
5. Central Server xử lý dữ liệu
6. Central Server lưu database
7. Frontend lấy dữ liệu
8. Edge Node mất kết nối
9. Edge Node reconnect
10. Authentication failure

Chỉ giữ các diagram thực sự cần thiết cho đồ án.

==================================================
PHASE 15 — VIẾT BÁO CÁO ĐỒ ÁN
==================================================

Sau khi hiểu và kiểm tra source code, hãy tạo báo cáo đồ án kỹ sư phần mềm hoàn chỉnh.

Báo cáo phải dựa trên SOURCE CODE THỰC TẾ.

TUYỆT ĐỐI KHÔNG BỊA:

- chức năng
- công nghệ
- số liệu benchmark
- kiến trúc
- API
- database
- use case
- kết quả thử nghiệm

Nếu thiếu thông tin:
[THIẾU DỮ LIỆU — CẦN BỔ SUNG]

==================================================
CẤU TRÚC BÁO CÁO
==================================================

Trang bìa

Lời cảm ơn

Tóm tắt

Abstract

Mục lục

Danh sách hình

Danh sách bảng

Danh sách từ viết tắt

CHƯƠNG 1 — TỔNG QUAN ĐỀ TÀI

1.1. Lý do chọn đề tài
1.2. Bối cảnh
1.3. Vấn đề cần giải quyết
1.4. Mục tiêu
1.5. Phạm vi
1.6. Đối tượng nghiên cứu
1.7. Phương pháp thực hiện
1.8. Kết quả đạt được
1.9. Cấu trúc báo cáo

CHƯƠNG 2 — CƠ SỞ LÝ THUYẾT

2.1. Kiến trúc Three-Tier
2.2. Client-Server
2.3. Centralized Server
2.4. Edge Computing / Edge Node
2.5. REST API
2.6. Authentication / Authorization
2.7. Database
2.8. Docker / Containerization
2.9. Các công nghệ thực tế được sử dụng

Không viết lan man lý thuyết không liên quan.

CHƯƠNG 3 — PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

3.1. Yêu cầu chức năng
3.2. Yêu cầu phi chức năng
3.3. Actor
3.4. Use Case Diagram
3.5. Đặc tả Use Case quan trọng
3.6. System Architecture
3.7. Three-Tier Architecture
3.8. Central Server Architecture
3.9. Edge Node Architecture
3.10. Deployment Architecture
3.11. Component Diagram
3.12. Sequence Diagram
3.13. Data Flow
3.14. Database Design
3.15. ERD

CHƯƠNG 4 — XÂY DỰNG HỆ THỐNG

4.1. Môi trường phát triển
4.2. Frontend
4.3. Backend
4.4. Database
4.5. Central Server
4.6. Edge Node
4.7. API
4.8. Authentication
4.9. Data synchronization
4.10. Docker
4.11. Deployment

Mỗi phần phải có:

- Mô tả
- Hình ảnh thực tế
- Code snippet quan trọng nếu cần
- Giải thích implementation

CHƯƠNG 5 — KIỂM THỬ VÀ ĐÁNH GIÁ

5.1. Test strategy
5.2. Unit testing
5.3. Integration testing
5.4. API testing
5.5. Edge Node testing
5.6. End-to-end testing
5.7. Security testing
5.8. Performance testing
5.9. Test cases
5.10. Kết quả
5.11. Đánh giá

Tuyệt đối không tự tạo kết quả test.

Nếu chưa test:
ghi rõ "Chưa thực hiện".

CHƯƠNG 6 — TRIỂN KHAI VÀ VẬN HÀNH

6.1. Production architecture
6.2. Deployment
6.3. Docker
6.4. Environment
6.5. Reverse proxy
6.6. HTTPS
6.7. Monitoring
6.8. Logging
6.9. Backup
6.10. Recovery
6.11. Security

CHƯƠNG 7 — KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

7.1. Kết quả đạt được
7.2. Hạn chế
7.3. Bài học
7.4. Hướng phát triển
7.5. Khả năng mở rộng

TÀI LIỆU THAM KHẢO

PHỤ LỤC

- API
- Database schema
- Configuration
- Test cases
- Diagram
- Screenshot

==================================================
PHASE 16 — HÌNH ẢNH CHO BÁO CÁO
==================================================

Tự động xác định các screenshot/hình cần thiết.

Ví dụ:

Figure 1 — Tổng quan hệ thống
Figure 2 — Kiến trúc 3 tầng
Figure 3 — Central Server
Figure 4 — Edge Node
Figure 5 — Deployment Architecture
Figure 6 — Login
Figure 7 — Dashboard
Figure 8 — Edge Node monitoring
Figure 9 — Database
Figure 10 — API
Figure 11 — Docker containers
Figure 12 — Test result

Ưu tiên screenshot THỰC TẾ từ project.

Nếu cần hình minh họa kiến trúc:
tạo Mermaid / PlantUML.

Không dùng hình stock nếu không cần.

Mỗi hình phải có:

Figure X. Tên hình

và trong nội dung phải có câu tham chiếu:

"Như thể hiện trong Hình X..."

==================================================
PHASE 17 — UML
==================================================

Không cố nhồi UML.

Chọn UML phù hợp:

- Use Case Diagram
- Class Diagram
- Sequence Diagram
- Component Diagram
- Deployment Diagram
- Activity Diagram nếu cần

Mỗi diagram phải có:

- Tên
- Mục đích
- Giải thích
- Quan hệ chính

Diagram phải khớp source code.

==================================================
PHASE 18 — TẠO DOCUMENT
==================================================

Tạo báo cáo dạng DOCX chuyên nghiệp.

Yêu cầu:

- Font Times New Roman
- Cỡ chữ phù hợp chuẩn báo cáo đại học
- Heading 1/2/3 đúng cấu trúc
- Header/Footer
- Page number
- Table of Contents
- List of Figures
- List of Tables
- Caption cho hình
- Caption cho bảng
- Cross-reference nếu có thể
- Code block
- Bảng biểu
- Khoảng cách đoạn hợp lý
- Không để heading nằm cuối trang một mình
- Không để hình bị vỡ layout
- Không để bảng tràn lề

Tạo style nhất quán toàn bộ tài liệu.

==================================================
PHASE 19 — QUALITY REVIEW
==================================================

Sau khi viết báo cáo:

Đọc lại TOÀN BỘ báo cáo.

Kiểm tra:

- Logic
- Chính tả
- Thuật ngữ
- Tính nhất quán
- Tên module
- Tên API
- Tên database
- Tên class
- Tên Edge Node
- Tên Central Server
- Hình ↔ nội dung
- UML ↔ source code
- Use Case ↔ chức năng thực tế
- Database diagram ↔ database thực tế

Không được có:

- Nội dung bịa
- Diagram sai
- API không tồn tại
- Function không tồn tại
- Screenshot không liên quan
- Placeholder bị bỏ quên

==================================================
FINAL DELIVERABLE
==================================================

Cuối cùng trả về:

1. Production Readiness Score
2. Architecture Review
3. Security Review
4. Performance Review
5. Danh sách issue đã sửa
6. Danh sách issue còn tồn tại
7. Test result
8. Architecture diagrams
9. UML diagrams
10. Use Case
11. Database diagram
12. Production deployment diagram
13. Báo cáo DOCX hoàn chỉnh
14. README cập nhật
15. Deployment guide
16. API documentation
17. Changelog

Quan trọng:

KHÔNG chỉ viết báo cáo.

Hãy thực sự:

SCAN → ANALYZE → AUDIT → FIX → TEST → DOCUMENT → REVIEW

Nếu có thể tạo file trực tiếp, hãy tạo:

/docs/final-report.docx
/docs/diagrams/
/docs/screenshots/
/docs/api/
/docs/deployment/
/docs/test-report.md

README phải có:

- Architecture
- Setup
- Development
- Production
- Docker
- Environment variables
- Database
- API
- Edge Node
- Central Server
- Troubleshooting

Mục tiêu cuối cùng:

Project phải vừa có chất lượng đủ tốt để deploy production, vừa có đầy đủ bằng chứng kỹ thuật để bảo vệ đồ án.

Không được đánh đổi tính chính xác để làm báo cáo đẹp.