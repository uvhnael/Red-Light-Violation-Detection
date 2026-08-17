### 2. File `CENTRAL_SERVER_README.md` (Đặt tại thư mục `central-server/`)

```markdown
# Central Server - Traffic Management System
**Author:** anhvu / vulee
**Role:** API Gateway, Data Storage, Business Logic, and Automation Workflows.

## 📌 Architecture Overview
Trung tâm đầu não xử lý dữ liệu. Chịu trách nhiệm nhận payload từ hàng nghìn Edge Node, lưu trữ file đa phương tiện an toàn, xử lý logic kiểm duyệt, và định tuyến dữ liệu đến các Workflow tự động hóa hoặc AI Engine phân tích sâu.

## 🛠️ Tech Stack & Constraints
- **Core API:** Java Spring Boot 3 (JDK 17+).
- **Database:** PostgreSQL (High-Availability schema).
- **Object Storage:** MinIO S3.
- **Automation & ML:** n8n, Ollama (Text-to-SQL API).
- **Deployment:** Docker Compose, Portainer, Expose via Cloudflare Tunnels.

## 📁 Project Structure (Yêu cầu Agent thiết lập)
```text
central-server/
├── src/main/java/com/traffic/
│   ├── controllers/         # EdgeIngestController, DashboardController
│   ├── services/            # MinIOService, ViolationService, WebhookService
│   ├── models/              # JPA Entities (Violation, EdgeNode, MediaAsset)
│   └── repositories/        # Spring Data JPA Repositories
├── src/main/resources/
│   ├── application.yml      # Cấu hình DB, MinIO, Port, JWT
│   └── db/migration/        # Flyway SQL scripts
└── docker-compose.yml       # postgres, minio, spring-boot, n8n, ollama
⚙️ Detailed Implementation Instructions for AI Agent
1. Database Schema (Flyway/PostgreSQL)
Thiết kế các entity JPA với liên kết chặt chẽ:

EdgeNode: id (UUID), name, ip_address, last_ping, status.

Violation: id (UUID), node_id (FK), plate_number, violation_type, timestamp_ms, ai_confidence, status (Enum: PENDING, APPROVED, REJECTED, SUSPICIOUS).

MediaAsset: id, violation_id (FK), asset_type (FULL_IMG, CROP_IMG, VIDEO), s3_object_key, presigned_url_cache.

2. Spring Boot Core Logic
Endpoint POST /api/v1/ingest:

Validate API Key của Edge Node.

Parse multipart/form-data. Tách biệt file và JSON metadata.

Upload tuần tự các file lên MinIO bucket (vd: traffic-violations). Lấy object paths.

Insert record vào table Violations và MediaAssets bằng một Database Transaction duy nhất (Rollback nếu rớt mạng MinIO).

Endpoint PUT /api/v1/violations/{id}/status: Dành cho Frontend gọi vào để Approve/Reject lỗi.

Webhook Service: Khi một Violation mới được tạo, đẩy JSON event sang n8n webhook URL.

3. Ecosystem Integration (n8n & Ollama)
Cấu hình file docker-compose.yml để spin up n8n và Ollama trên cùng một bridge network.

Spring Boot không trực tiếp gọi Telegram. Nó bắn webhook sang n8n. n8n sẽ chịu trách nhiệm định dạng tin nhắn cảnh báo và gửi đi.

Tùy chọn nâng cao: Mở một service Python FastAPI nhỏ nằm cạnh Ollama, được n8n gọi để phân tích xem chuỗi nhận diện biển số có bất thường hay không.