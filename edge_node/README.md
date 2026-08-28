# Edge Node — Red-Light Violation Detection

Node xử lý tại chỗ (edge): đọc video/RTSP từ camera, chạy pipeline thị giác máy tính để phát hiện xe vượt đèn đỏ, đọc biển số, và đẩy hồ sơ vi phạm lên Central Server. Kèm control-plane API (FastAPI) để web dashboard kẻ vạch dừng / vùng đèn / hướng giám sát từ xa.

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────┐
│                         EDGE NODE                            │
│                                                              │
│  Camera/video ──▶ Pipeline ──────────────────▶ Durable Outbox│
│                   │ YOLO detect (xe)            (SQLite)     │
│                   │ ByteTrack (tracking)            │        │
│                   │ Đèn giao thông (YOLO-cls+HSV)   ▼        │
│                   │ Tripwire + direction ──▶ ViolationSender │──▶ Central
│                   │ Plate detect + OCR (biển số)      (batch)│    Server
│                   │                                          │
│  FastAPI control-plane :8080  ◀── web dashboard kẻ vạch/hướng│
│  HLS camera stream      ◀── ffmpeg, web xem live            │
└──────────────────────────────────────────────────────────────┘
```

## Luồng phát hiện vi phạm

1. **Detect + track** — YOLO26m phát hiện 4 lớp phương tiện (`car`, `bike`, `van/bus`, `truck`), ByteTrack giữ ID ổn định.
2. **Đèn giao thông** — classifier YOLO26n-cls (3 màu red/yellow/green) trên vùng đèn do operator kẻ, fusion với HSV + vị trí bóng đèn; trạng thái đỏ phải ổn định `RED_STABLE_FRAMES` frame mới được xét.
3. **Tripwire** — xe bị tính vi phạm khi **tâm bbox** cắt qua vạch dừng (điểm crossing = bbox center) trong lúc đèn đỏ đã ổn định.
4. **Hướng giám sát** — đường 2 chiều: chỉ tính xe đi đúng hướng đã calibration (`positive_to_negative` / `negative_to_positive`), bỏ qua xe chiều ngược. `any` = tính cả 2 hướng.
5. **Biển số** — YOLO detect biển + fast-plate-ocr đọc chữ, validate theo cấu trúc biển VN (`vn_plate.py`: mã tỉnh 11–99, seri, 4–5 số, chuẩn hoá `NN-XXXXXX`). Đọc hỏng tại frame vi phạm thì dùng biển tốt nhất đã nhớ trước đó của cùng xe (plate memory).
6. **Gửi đi** — hồ sơ vi phạm (ảnh toàn cảnh + ảnh crop biển + metadata) ghi vào outbox SQLite rồi batch-push lên Central. Sống sót khi mất mạng / restart; dedup theo `event_id`.

Chưa có vạch dừng → pipeline vẫn chạy detection/tracking nhưng **không xét vi phạm** cho tới khi operator kẻ vạch trên web.

## Cấu trúc

```
edge_node/
├── main.py                  # CLI entry point (python -m edge_node)
├── settings.py              # Toàn bộ cấu hình qua env var
├── central_client.py        # Đăng ký node + heartbeat lên Central
├── outbox.py                # Durable SQLite outbox
├── violation_sender.py      # Batch delivery + upload media
├── camera_stream.py         # ffmpeg → HLS để web xem live
├── api/server.py            # FastAPI control-plane
├── core/
│   ├── pipeline.py          # Điều phối toàn bộ pipeline
│   ├── detector.py          # YOLO detector (.pt/.onnx/.engine)
│   ├── byte_tracker.py      # ByteTrack (supervision)
│   ├── traffic_light_yolo.py# Classifier màu đèn + fusion HSV
│   ├── traffic_light_cv.py  # Fallback HSV OpenCV
│   ├── violation_logic.py   # Ổn định đèn đỏ + tripwire + direction
│   ├── geometry.py          # Hình học tripwire / hướng cắt
│   ├── plate_detector.py    # Detector biển số (YOLO fine-tuned)
│   ├── ocr_recognizer.py    # OCR biển số (fast-plate-ocr)
│   ├── vn_plate.py          # Validator + chuẩn hoá biển số VN
│   ├── plate_associator.py  # Gán biển số với track xe
│   ├── video_io.py          # Nguồn frame (file/RTSP)
│   ├── calibration.py       # Frame nền cho web calibration
│   ├── visualizer.py        # Overlay debug (box đỏ khi vi phạm, UI scale theo độ phân giải)
│   └── export.py            # Xuất JSON/CSV
├── Dockerfile
├── docker-compose.yml       # Chạy độc lập (pipeline + API)
└── requirements.txt
```

Weights và dữ liệu nằm ở **gốc project** (không nằm trong `edge_node/`): `models/` (weights) và `data/` (video, calibration, outbox, HLS).

## Cài đặt & chạy

### Chạy trực tiếp (không Docker)

```bash
# GPU CUDA (khuyến nghị): cài torch CUDA trước
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r edge_node/requirements.txt

# Chạy với video file + control API + loop
python -m edge_node --input data/videos/16h30.25.9.22.mp4 --start-api --loop

# RTSP camera
python -m edge_node --input rtsp://camera:554/stream --start-api

# Truyền thẳng calibration qua CLI (thay vì kẻ trên web)
python -m edge_node --input ... --stop-line 100,400,800,400 \
    --direction negative_to_positive --light-roi 620,80,60,160

# Offline (không gửi vi phạm) / tắt OCR
python -m edge_node --input ... --no-outbox
python -m edge_node --input ... --no-ocr
```

Không có GPU sẽ tự fallback CPU (YOLO device auto-detect, FP16 tự bật khi chạy GPU).

### Test nhanh không cần Docker: `run_pipeline.py` (gốc project)

Runner chỉ chạy detection → tracking → đèn → tripwire, **không** outbox/sender/central. Calibration bằng chuột trên frame đầu: click 2 điểm kẻ vạch dừng, vẽ mũi tên chọn hướng giám sát (đường 2 chiều), click 2 điểm khoanh vùng đèn; lưu vào `data/calibration.json` theo tên video, lần sau tự nạp.

```bash
python run_pipeline.py                          # video mặc định trong data/videos/
python run_pipeline.py data/videos/aziz1.MP4
python run_pipeline.py --recalibrate            # kẻ lại
```

Live view: box xe đang track vẽ màu **đỏ** khi bị tính vi phạm; độ dày nét, cỡ chữ, kích thước overlay tự scale theo độ phân giải video.

### Export model tối ưu

```bash
python -m edge_node --export-onnx models/yolo26m_vehicle.pt       # ~2x
python -m edge_node --export-tensorrt models/yolo26m_vehicle.pt   # ~4x, cần GPU NVIDIA
```

### Docker

```bash
# Độc lập (chỉ edge)
docker compose -f edge_node/docker-compose.yml up -d

# Full stack (postgres + minio + central + web + edge) — chạy từ gốc project
./start.sh
```

## Control-plane API (mặc định :8080)

Endpoint ghi (`POST`) yêu cầu header `X-Admin-Token` khớp `EDGE_API_TOKEN`.

| Endpoint | Method | Mô tả |
|---|---|---|
| `/health` | GET | Health check (camera, pipeline, outbox) |
| `/action/restart` | POST | Restart pipeline (SIGTERM chính nó, container tự restart) |
| `/action/stop-line` | GET/POST | Xem/đặt vạch dừng + `direction` |
| `/api/light-roi` | GET/POST | Xem/đặt vùng đèn tín hiệu |
| `/api/calibration` | GET | Trạng thái vạch + vùng đèn + hướng hiện tại |
| `/api/calibration/snapshot` | GET | Frame JPEG để kẻ trên web |
| `/api/light-state` | GET | Trạng thái đèn realtime (red/yellow/green) |
| `/api/cameras` | GET | Danh sách camera |
| `/api/cameras/{id}/stream` | GET | HLS playlist (web xem live) |
| `/api/cameras/{id}/snapshot` | GET | Snapshot JPEG |

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/action/stop-line \
  -H 'Content-Type: application/json' -H 'X-Admin-Token: ***' \
  -d '{"x1":0,"y1":420,"x2":1280,"y2":420,"direction":"positive_to_negative"}'
```

## Cấu hình (env var chính)

| Variable | Default | Mô tả |
|---|---|---|
| `VIDEO_INPUT` | – | Đường dẫn video/RTSP |
| `VIDEO_LOOP` / `VIDEO_REALTIME` | `false` | Lặp video / pace theo thời gian thực |
| `YOLO_MODEL_PATH` | `models/yolo26m_vehicle.pt` | Model phát hiện xe |
| `YOLO_CONFIDENCE` / `YOLO_IMG_SIZE` | `0.35` / `640` | Ngưỡng / kích thước input |
| `YOLO_DEVICE` / `YOLO_FP16` | auto | `cuda`/`cpu`/trống; FP16 tự bật trên GPU |
| `ENABLE_OCR` / `OCR_MODEL_NAME` / `OCR_DEVICE` | `true` / global-plates-mobile-vit-v2 / auto | OCR biển số |
| `TRAFFIC_LIGHT_MODEL_PATH` | `models/traffic_light_cls.pt` | Model phân loại màu đèn |
| `TRAFFIC_LIGHT_FUSION` | `true` | Đối chiếu YOLO + HSV + vị trí đèn |
| `RED_STABLE_FRAMES` / `RED_MIN_CONFIDENCE` | – | Số frame ổn định / ngưỡng tin đèn đỏ |
| `CENTRAL_SERVER_URL` | `http://central-server:8000/api/violations` | Nơi đẩy vi phạm |
| `NODE_REGISTER_URL` / `NODE_HEARTBEAT_INTERVAL` | – | Đăng ký + heartbeat lên Central |
| `NODE_ID` / `NODE_NAME` / `NODE_IP_ADDRESS` | `edge-node-01` | Định danh node |
| `OUTBOX_ENABLED` / `OUTBOX_DB_PATH` | `true` / `data/outbox/violations.db` | Durable outbox |
| `OUTBOX_BATCH_SIZE` / `OUTBOX_FLUSH_INTERVAL` | `20` / `5` | Batch push |
| `EDGE_API_TOKEN` | – | Token bảo vệ endpoint ghi của API |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8080` | Bind control-plane |
| `CAMERA_STREAM_ENABLED` / `CAMERA_STREAM_HLS_DIR` | – | Bật HLS + thư mục segment |

## Payload vi phạm gửi lên Central

```json
{
  "event_id": "rlv-...",
  "track_id": 12,
  "light_state": "red",
  "plate": { "text": "30-K12345", "confidence": 0.92 }
}
```

Kèm media: ảnh toàn cảnh (đã nén theo `EVIDENCE_IMAGE_MAX_WIDTH` / `EVIDENCE_IMAGE_QUALITY`) + ảnh crop biển số, upload qua `/api/v1/violations/{event_id}/media`.

## Công nghệ

- **Detection**: YOLO26m fine-tuned 4 lớp phương tiện — `.pt`/`.onnx`/`.engine`, CUDA + FP16
- **Tracking**: ByteTrack (supervision <0.29)
- **Đèn giao thông**: YOLO26n-cls fine-tuned (LISA dataset) + fusion HSV; fallback OpenCV HSV
- **OCR biển số**: fast-plate-ocr (ONNX) + validator biển VN
- **Delivery**: durable SQLite outbox + batch sender (không Redis/Celery)
- **API**: FastAPI + Uvicorn; HLS qua ffmpeg
