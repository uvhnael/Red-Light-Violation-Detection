# RLVD — Red-Light Violation Detection

Hệ thống phát hiện vi phạm vượt đèn đỏ theo kiến trúc **Edge – Central – Web**:

- **Edge Node** (Python) đặt tại mỗi camera: xử lý video realtime, phát hiện xe vượt đèn đỏ, đọc biển số, đẩy hồ sơ vi phạm về trung tâm.
- **Central Server** (Java Spring Boot) nhận và lưu trữ vi phạm (PostgreSQL + MinIO), cung cấp REST API và AI hỏi dữ liệu bằng tiếng Việt.
- **Web Dashboard** (Next.js) giám sát, duyệt vi phạm, xem camera live, kẻ vạch dừng / vùng đèn / hướng giám sát từ xa.

## Kiến trúc tổng quan

```
   Camera/Video
        │
        ▼
┌──────────────┐   violation batch + media    ┌──────────────────┐
│  EDGE NODE   │ ───────────────────────────▶ │  CENTRAL SERVER  │
│  Python      │   (durable SQLite outbox)    │  Spring Boot     │
│  YOLO+Byte   │                              │        │         │
│  Track+OCR   │ ◀── calibration (vạch/hướng) │   ┌────┴────┐    │
│  FastAPI     │        (proxy từ web)        │   ▼         ▼    │
│  HLS stream  │                              │ PostgreSQL  MinIO│
└──────────────┘                              └────────┬─────────┘
        ▲                                              │ /api (rewrite)
        │ xem live / kẻ vạch                           ▼
        └──────────────────────  WEB DASHBOARD (Next.js) ── AI Text-to-SQL (Gemini)
```

## Thư mục

```
.
├── edge_node/          # Node xử lý tại camera (Python) — xem edge_node/README.md
├── central_server/     # Backend trung tâm (Spring Boot) — xem central_server/README.md
├── web/                # Dashboard (Next.js) — xem web/README.md
├── train_model/        # Script huấn luyện YOLO (biển số, đèn, phương tiện)
├── scripts/            # Benchmark model + cài nvidia-container-toolkit
├── models/             # Weights YOLO (không commit — *.pt trong .gitignore)
├── data/               # Video, calibration, outbox, HLS (không commit)
├── run_pipeline.py     # Runner test pipeline local (calibration bằng chuột, không gửi server)
├── docker-compose.full.yml  # Full stack: postgres + minio + central + web + edge
└── start.sh            # Script dựng/chạy toàn bộ stack bằng Docker
```

## Yêu cầu

- **Docker** + Docker Compose plugin (chạy full stack)
- **GPU NVIDIA** + nvidia-container-toolkit (tăng tốc YOLO/OCR cho edge — khuyến nghị)
- Weights model trong `models/`: `yolo26m_vehicle.pt` (bắt buộc), `traffic_light_cls.pt`, `license_plate_yolo26.pt`
- Video đầu vào trong `data/videos/` (mặc định `data/videos/16h30.25.9.22.mp4`)

## Chạy nhanh (Docker full stack)

```bash
./start.sh              # build + chạy tất cả service
./start.sh --no-build   # bỏ qua build lại image
./start.sh minimal      # chỉ web + central + postgres + minio (không edge/GPU)
./start.sh status       # trạng thái container
./start.sh logs         # tail logs
./start.sh rebuild web-dashboard   # build lại 1 service
./start.sh down         # dừng + xoá container
```

Cổng mặc định (host):

| Service | Cổng | Ghi chú |
|---|---|---|
| web-dashboard | `3000` | Mở trình duyệt tại đây |
| central-server | `8002` | API nội bộ `:8000` |
| edge-pipeline | `8082` | Control-plane API nội bộ `:8080` |
| postgres | `5432` | DB `rlvd_central` |
| minio | `9010` (S3) / `9011` (console) | Bucket `violations` |

## Quy trình vận hành

1. **Edge node khởi động** → tự đăng ký + heartbeat lên Central.
2. **Mở web** `http://localhost:3000` → trang Nodes → chọn node.
3. **Kẻ calibration** trên ảnh snapshot: vạch dừng (2 điểm), **mũi tên hướng giám sát** (đường 2 chiều — chỉ tính xe đi đúng hướng), vùng đèn (2 điểm). Lưu có hiệu lực ngay, không cần restart.
4. **Pipeline xét vi phạm**: xe cắt vạch dừng khi đèn đỏ đã ổn định → tạo hồ sơ (ảnh toàn cảnh + crop biển số + metadata) → đẩy lên Central qua outbox.
5. **Duyệt trên web**: trang Review để xác nhận / từ chối (human-in-the-loop); trang Violations để tra cứu; AI chat để hỏi bằng tiếng Việt.

Chưa kẻ vạch → pipeline vẫn chạy detection/tracking nhưng không xét vi phạm.

## Test pipeline local (không cần Docker/server)

```bash
python run_pipeline.py                          # video mặc định
python run_pipeline.py data/videos/aziz1.MP4
python run_pipeline.py --recalibrate            # kẻ lại bằng chuột
```

Calibration bằng chuột trên frame đầu, lưu vào `data/calibration.json`. Live view: box xe vi phạm tô **đỏ**, overlay tự scale theo độ phân giải.

## Công nghệ chính

| Thành phần | Công nghệ |
|---|---|
| Phát hiện phương tiện | YOLO26m fine-tuned (car/bike/van-bus/truck), CUDA + FP16, hỗ trợ ONNX/TensorRT |
| Tracking | ByteTrack (supervision) |
| Đèn giao thông | YOLO26n-cls fine-tuned (LISA) + fusion HSV; fallback OpenCV HSV |
| OCR biển số | fast-plate-ocr (ONNX) + validator chuẩn biển VN |
| Delivery | Durable SQLite outbox + batch sender (chịu mất mạng/restart) |
| Backend | Spring Boot 3.3.2 / Java 17, JPA, MinIO SDK |
| AI query | Gemini Text-to-SQL (chỉ SELECT, chặn SQL nguy hiểm) |
| Frontend | Next.js 16 / React 19 / Tailwind 4 / recharts / hls.js |
| Hạ tầng | PostgreSQL 16, MinIO, Docker Compose |

## Huấn luyện model

Script trong `train_model/`:

- `vehical_detection/` — detect phương tiện (ra `models/yolo26m_vehicle.pt`)
- `lincenseplate/` — detect biển số (ra `models/license_plate_yolo26.pt`)
- `traffic_light_cls/` — phân loại màu đèn (ra `models/traffic_light_cls.pt`)

Benchmark model: `scripts/benchmark_vehicle_models.py`.

## Tài liệu chi tiết

- [Edge Node](edge_node/README.md)
- [Central Server](central_server/README.md)
- [Web Dashboard](web/README.md)
