# Red-Light Violation Detection – Edge Node

Hệ thống phát hiện vi phạm vượt đèn đỏ chạy trên thiết bị edge (Edge Node). Xử lý video realtime, phát hiện phương tiện vi phạm, và đẩy dữ liệu lên Central Server qua message queue.

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

## Cấu trúc project

```
.
├── edge_node/                    # Main package
│   ├── core/                     # Vision pipeline
│   │   ├── contracts.py          # Protocols & data classes
│   │   ├── config.py             # Pipeline configs
│   │   ├── detector.py           # YOLO detector (.pt/.onnx/.engine)
│   │   ├── byte_tracker.py       # ByteTrack (supervision)
│   │   ├── pipeline.py           # Orchestration + queue dispatch
│   │   ├── traffic_light_cv.py   # OpenCV traffic light classifier
│   │   ├── violation_logic.py    # Red-light stabilizer + tripwire
│   │   ├── geometry.py           # Tripwire geometry
│   │   ├── video_io.py           # Frame source (file/RTSP)
│   │   └── export.py             # JSON/CSV export
│   ├── worker/                   # Background tasks
│   │   ├── celery_app.py         # Celery + Redis config
│   │   └── tasks.py              # push_violation_to_server
│   ├── api/                      # Control plane
│   │   └── server.py             # FastAPI /health, /action/restart
│   ├── settings.py               # Env-var configuration
│   ├── main.py                   # CLI entry point
│   ├── models/                   # YOLO weights (.pt/.onnx/.engine)
│   ├── data/                     # Input videos/images
│   ├── Dockerfile                # Docker image
│   ├── docker-compose.yml        # Full stack (Redis + Worker + Pipeline)
│   └── requirements.txt          # Python dependencies
└── .gitignore
```

## Cài đặt

```bash
pip install -r edge_node/requirements.txt
```

## Sử dụng

### Chạy pipeline

```bash
# Với video file (dùng --loop để video lặp lại liên tục)
python -m edge_node --input edge_node/data/videos/aziz1.MP4 \
    --stop-line 100,400,800,400 \
    --direction negative_to_positive \
    --start-api \
    --loop

# Với RTSP camera
python -m edge_node --input rtsp://camera:554/stream \
    --stop-line 100,400,800,400 \
    --direction negative_to_positive \
    --start-api

# Offline mode (không cần Redis/Celery)
python -m edge_node --input edge_node/data/videos/traffic_video_modified.mp4 \
    --stop-line 100,400,800,400 \
    --no-queue
```

### Export model (tối ưu cho edge)

```bash
# Export sang ONNX (nhanh hơn ~2x)
python -m edge_node --export-onnx edge_node/models/yolov8s.pt

# Export sang TensorRT (nhanh hơn ~4x, cần NVIDIA GPU)
python -m edge_node --export-tensorrt edge_node/models/yolov8s.pt
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

```bash
# Health check
curl http://localhost:8080/health

# Restart services
curl -X POST http://localhost:8080/action/restart
```

## Cấu hình (Environment Variables)

| Variable | Default | Mô tả |
|----------|---------|-------|
| `VIDEO_INPUT` | _none_ | Đường dẫn video/RTSP cho docker |
| `VIDEO_LOOP` | `false` | Lặp lại video (cho debug) |
| `YOLO_MODEL_PATH` | `edge_node/models/yolov8s.pt` | Đường dẫn model |
| `YOLO_CONFIDENCE` | `0.35` | Ngưỡng confidence |
| `YOLO_IMG_SIZE` | `640` | Input image size |
| `YOLO_DEVICE` | _(auto)_ | `cuda`, `cpu`, hoặc `0` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker URL |
| `CENTRAL_SERVER_URL` | `http://central-server:8000/api/violations` | API nhận violations |
| `ENABLE_QUEUE` | `true` | Bật/tắt Celery queue |
| `NODE_ID` | `edge-node-01` | ID định danh node |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8080` | API bind port |

## Công nghệ

- **Detection**: YOLOv8s (Ultralytics) – hỗ trợ `.pt`, `.onnx`, `.engine`
- **Tracking**: ByteTrack (supervision) – fix lỗi nhảy ID xe
- **Queue**: Redis + Celery – tách biệt luồng xử lý ảnh và đẩy dữ liệu
- **API**: FastAPI + Uvicorn – control plane cho Central Server
- **Traffic Light**: OpenCV HSV – không cần model riêng
