# Red-Light Violation Detection – Edge Node

Hệ thống phát hiện vi phạm vượt đèn đỏ chạy trên thiết bị edge. Xử lý video realtime, phát hiện phương tiện vi phạm, và đẩy dữ liệu lên Central Server qua message queue.

## Kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                      EDGE NODE                          │
│                                                         │
│  ┌──────────┐   ┌───────────┐   ┌────────────────────┐  │
│  │  Camera   │──▶│  Pipeline  │──▶│  Violation Payload │  │
│  │ (RTSP/file)│  │ YOLO+Byte │  │    (dict in RAM)   │  │
│  └──────────┘   │  Tracker   │  └────────┬───────────┘  │
│                 └───────────┘           │               │
│                                          ▼               │
│  ┌──────────┐   ┌───────────┐   ┌──────────────────┐   │
│  │  FastAPI  │   │   Redis    │◀──│  Celery Worker   │   │
│  │ /health   │   │  (broker)  │──▶│ push_violation   │──────▶ Central Server
│  │ /restart  │   └───────────┘   └──────────────────┘   │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

## Quy trình vận hành

1. **Camera đăng ký** — edge node khởi động và đăng ký lên Central Server (kèm heartbeat định kỳ).
2. **Operator duyệt trên web** — mở trang node trên web dashboard, xem ảnh snapshot từ camera.
3. **Kẻ vạch dừng + vùng đèn** — kéo thả trực tiếp trên ảnh; cấu hình được đẩy xuống edge node qua API và có hiệu lực ngay (không cần restart).
4. **Pipeline hoạt động đầy đủ** — trước khi kẻ vạch, pipeline vẫn chạy detection/tracking nhưng **không xét vi phạm** (chưa có mốc so sánh).

## Cấu trúc project

```
.
├── edge_node/                    # Package chính
│   ├── core/                     # Vision pipeline
│   │   ├── contracts.py          # Protocols & data classes
│   │   ├── config.py             # Config runtime của pipeline
│   │   ├── detector.py           # YOLO detector (.pt/.onnx/.engine)
│   │   ├── byte_tracker.py       # ByteTrack (supervision)
│   │   ├── pipeline.py           # Điều phối + dispatch queue
│   │   ├── traffic_light_yolo.py # Classifier màu đèn YOLO26n-cls
│   │   ├── traffic_light_cv.py   # Classifier HSV OpenCV (fallback)
│   │   ├── violation_logic.py    # Ổn định trạng thái đèn + tripwire
│   │   ├── vn_plate.py           # Validator/chuẩn hoá biển số VN
│   │   ├── ocr_recognizer.py     # OCR biển số (fast-plate-ocr)
│   │   ├── plate_detector.py     # Detector biển số (YOLO26 fine-tuned)
│   │   ├── plate_associator.py   # Gán biển số với track xe
│   │   ├── geometry.py           # Hình học tripwire
│   │   ├── video_io.py           # Nguồn frame (file/RTSP)
│   │   ├── calibration.py        # Frame ảnh nền cho web calibration
│   │   ├── visualizer.py         # Vẽ overlay phục vụ debug
│   │   └── export.py             # Xuất JSON/CSV
│   ├── worker/                   # Background tasks
│   │   ├── celery_app.py         # Celery + Redis config
│   │   └── tasks.py              # push_violation_to_server
│   ├── api/                      # Control plane
│   │   └── server.py             # FastAPI /health, /action/*
│   ├── settings.py               # Cấu hình qua env var
│   ├── main.py                   # CLI entry point
│   ├── models/                   # Weights YOLO (.pt/.onnx/.engine)
│   ├── data/                     # Video/ảnh đầu vào
│   ├── Dockerfile                # Docker image
│   ├── docker-compose.yml        # Full stack (Redis + Worker + Pipeline)
│   └── requirements.txt          # Python dependencies
└── .gitignore
```

## Cài đặt

```bash
pip install -r edge_node/requirements.txt
```

### GPU (CUDA) — khuyến nghị

```bash
# Cài torch bản CUDA trước, sau đó cài phần còn lại
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r edge_node/requirements.txt

# Tăng tốc OCR bằng GPU (tùy chọn)
pip install onnxruntime-gpu
```

Mặc định edge node tự dò GPU: nếu có CUDA sẽ chạy YOLO trên GPU với FP16 và OCR qua CUDA; không có GPU sẽ tự fallback về CPU.

## Sử dụng

### Chạy pipeline

```bash
# Với video file (dùng --loop để video lặp lại liên tục)
python -m edge_node --input edge_node/data/videos/aziz1.MP4 \
    --start-api \
    --loop

# Với RTSP camera
python -m edge_node --input rtsp://camera:554/stream --start-api

# Kẻ vạch tạm bằng CLI (nếu chưa kẻ trên web)
python -m edge_node --input ... \
    --stop-line 100,400,800,400 \
    --direction negative_to_positive

# Vùng đèn tạm bằng CLI (nếu chưa kẻ trên web)
python -m edge_node --input ... --light-roi 620,80,60,160

# Offline mode (không cần Redis/Celery)
python -m edge_node --input ... --no-queue

# Tắt OCR biển số (mặc định bật)
python -m edge_node --input ... --no-ocr
```

Sau khi khởi động mà chưa kẻ vạch (qua CLI lẫn web), pipeline chạy ở chế độ theo dõi: log báo "violations DISABLED" cho tới khi operator kẻ vạch trên web.

### Export model (tối ưu cho edge)

```bash
# Export sang ONNX (nhanh hơn ~2x)
python -m edge_node --export-onnx edge_node/models/yolo26m.pt

# Export sang TensorRT (nhanh hơn ~4x, cần NVIDIA GPU)
python -m edge_node --export-tensorrt edge_node/models/yolo26m.pt
```

### Chạy Celery worker

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start worker
celery -A edge_node.worker.celery_app worker --loglevel=info --concurrency=2
```

### Docker deployment

```bash
# Chạy full stack
docker compose -f edge_node/docker-compose.yml up -d

# Xem logs
docker compose -f edge_node/docker-compose.yml logs -f edge-pipeline
```

## API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/health` | GET | Health check (Redis, Celery, Camera) |
| `/action/restart` | POST | Restart services |
| `/action/stop-line` | GET/POST | Xem/set vạch dừng (operator kẻ từ web) |
| `/action/light-roi` | GET/POST | Xem/set vùng đèn tín hiệu |
| `/api/calibration` | GET | Trạng thái vạch + vùng đèn hiện tại |
| `/api/calibration/snapshot` | GET | Ảnh frame JPEG để kẻ line trên web |
| `/api/light-state` | GET | Trạng thái đèn realtime (red/yellow/green) |

```bash
# Health check
curl http://localhost:8080/health

# Set vạch dừng thủ công
curl -X POST http://localhost:8080/action/stop-line \
    -H 'Content-Type: application/json' \
    -d '{"x1": 0, "y1": 420, "x2": 1280, "y2": 420, "direction": "any"}'
```

## Cấu hình (Environment Variables)

| Variable | Default | Mô tả |
|----------|---------|-------|
| `VIDEO_INPUT` | _none_ | Đường dẫn video/RTSP cho docker |
| `VIDEO_LOOP` | `false` | Lặp lại video (cho debug) |
| `YOLO_MODEL_PATH` | `edge_node/models/yolo26m_vehicle.pt` | Model phát hiện phương tiện |
| `YOLO_CONFIDENCE` | `0.35` | Ngưỡng confidence |
| `YOLO_IMG_SIZE` | `640` | Input image size |
| `YOLO_DEVICE` | _(auto)_ | `cuda`, `cpu`, hoặc để trống để tự dò GPU |
| `YOLO_FP16` | _(auto)_ | `1`/`0` để ép bật/tắt FP16 (tự bật khi chạy GPU) |
| `ENABLE_OCR` | `true` | Bật/tắt OCR biển số (fast-plate-ocr) |
| `OCR_MODEL_NAME` | `global-plates-mobile-vit-v2-model` | Model OCR (hỗ trợ biển số Việt Nam) |
| `OCR_DEVICE` | `auto` | `cuda`, `cpu`, hoặc `auto` (tự dò GPU) |
| `TRAFFIC_LIGHT_MODEL_PATH` | `edge_node/models/traffic_light_cls.pt` | Model phân loại màu đèn |
| `TRAFFIC_LIGHT_FUSION` | `true` | Đối chiếu YOLO với HSV + vị trí đèn |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker URL |
| `CENTRAL_SERVER_URL` | `http://central-server:8000/api/violations` | API nhận violations |
| `ENABLE_QUEUE` | `true` | Bật/tắt Celery queue |
| `NODE_ID` | `edge-node-01` | ID định danh node |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8080` | API bind port |

Mỗi báo cáo vi phạm gửi lên Central Server gồm trường `plate`:

```json
{
  "event_id": "rlv-...",
  "track_id": 12,
  "light_state": "red",
  "plate": {
    "text": "30-K12345",
    "confidence": 0.92
  }
}
```

Biển số được chuẩn hoá về dạng `NN-XXXXXXX` (gạch sau mã tỉnh) và phải khớp cấu trúc biển VN: mã tỉnh 11-99, seri 2 chữ | 1 chữ + 1 số | 1 chữ, số thứ tự 4-5 chữ số. Chuỗi OCR không đúng cấu trúc (trừ khi sửa được lỗi ký tự phổ biến như O↔0, I↔1) bị loại bỏ, không tạo vi phạm giả.

Nếu OCR không đọc được biển số đúng tại frame vi phạm, edge node sẽ dùng biển số tốt nhất đã đọc trước đó của cùng xe (plate memory) để báo cáo luôn có trường biển số.

## Công nghệ

- **Detection**: YOLO26m fine-tuned 4 lớp phương tiện (`car`, `bike`, `van/bus`, `truck`) – hỗ trợ `.pt`, `.onnx`, `.engine`, chạy CUDA + FP16
- **Tracking**: ByteTrack (supervision) – fix lỗi nhảy ID xe
- **OCR biển số**: fast-plate-ocr + validator biển VN (`vn_plate.py`)
- **Traffic Light**: YOLO26n-cls fine-tuned trên LISA dataset (3 màu red/yellow/green, weights tại `edge_node/models/traffic_light_cls.pt`); fusion với HSV + vị trí bóng đèn; fallback OpenCV HSV nếu thiếu weights
- **Queue**: Redis + Celery – tách biệt luồng xử lý ảnh và đẩy dữ liệu
- **API**: FastAPI + Uvicorn – control plane cho Central Server
