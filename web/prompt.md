### 3. File `WEB_DASHBOARD_README.md` (Đặt tại thư mục `web-dashboard/`)

```markdown
# Web Dashboard - Traffic Control Center
**Author:** anhvu / vulee
**Role:** User Interface for operators to review, approve, and analyze traffic violations.

## 📌 Architecture Overview
Giao diện quản trị hiện đại, Single Page Application (SPA), tiêu thụ API từ Spring Boot Central Server. Hệ thống cần ưu tiên UX/UI cho thao tác nhanh của con người (Human-in-the-loop).

## 🛠️ Tech Stack & Constraints
- **Framework:** Next.js (App Router, React 18+).
- **Styling & UI:** Tailwind CSS, Shadcn UI, Lucide Icons.
- **State & Data Fetching:** React Query (@tanstack/react-query) hoặc SWR, Axios.
- **Charts:** Recharts hoặc Chart.js.

## 📁 Project Structure (Yêu cầu Agent thiết lập)
```text
web-dashboard/
├── src/
│   ├── app/
│   │   ├── (dashboard)/     # Layout chính có Sidebar
│   │   │   ├── page.tsx     # Bảng thống kê tổng quan
│   │   │   ├── violations/  # Trang duyệt hồ sơ vi phạm
│   │   │   └── nodes/       # Quản lý thiết bị Edge
│   │   └── api/             # Next.js API routes (nếu cần BFF pattern)
│   ├── components/
│   │   ├── ui/              # Shadcn component (Button, Modal, Table)
│   │   └── violation/       # ReviewModal.tsx, VideoPlayer.tsx
│   ├── lib/
│   │   └── api-client.ts    # Cấu hình Axios interceptor
│   └── hooks/               # Custom hooks cho API calls
├── tailwind.config.ts
└── package.json
⚙️ Detailed Implementation Instructions for AI Agent
1. Dashboard Analytics (/app/(dashboard)/page.tsx)
Render giao diện chia Grid.

Top Cards: Tổng số vi phạm hôm nay, Tỉ lệ duyệt thành công, Số node đang offline.

Center: Biểu đồ Recharts hiển thị xu hướng vi phạm theo từng giờ trong ngày.

2. Violation Review System (/app/violations/)
Đây là tính năng quan trọng nhất. Cần xây dựng:

Data Table: Hiển thị danh sách các lỗi có trạng thái PENDING. Có input tìm kiếm theo biển số xe.

Review Modal Component: Khi user bấm vào 1 dòng, popup hiện lên chia 2 cột:

Cột trái: Media. Hiển thị ảnh Full, ảnh Crop biển số. Phía dưới là một thẻ <video> HTML5 tự động autoplay loop đoạn clip 5s minh họa vi phạm. URL file được lấy từ Presigned URL của MinIO.

Cột phải: Thông tin trích xuất (Biển số OCR, loại lỗi, thời gian) và 2 nút Action to bự:

[✅ Approve & Generate Ticket] (Gọi API PUT status = APPROVED).

[❌ Reject (False Positive)] (Gọi API PUT status = REJECTED).

Tối ưu UX: Hỗ trợ dùng phím tắt (Keyboard shortcuts) mũi tên Trái/Phải để chuyển hồ sơ, phím Enter để Approve.

3. API & Proxy Configuration
Do Frontend và Backend khác port, cấu hình next.config.js sử dụng rewrites để điều hướng các request /api/v1/* về Spring Boot Backend URL (giải quyết triệt để lỗi CORS).

Đảm bảo các image domains (nếu dùng component <Image /> của Next.js) được config để allow hostname của MinIO server.