# ---- Edge Node Dockerfile ----
# Build context should be the project root (parent of edge_node/).
# Usage: docker build -f edge_node/Dockerfile .

FROM python:3.11-slim AS base

# System deps for OpenCV and video
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 support FIRST (before requirements.txt, which
# would otherwise pull the CPU-only torch from PyPI). The cu124 wheels bundle
# the CUDA runtime, so no separate CUDA base image is needed.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY edge_node/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY edge_node/ edge_node/

# Default env-vars (override via docker-compose or .env)
ENV CENTRAL_SERVER_URL=http://central-server:8000/api/violations \
    YOLO_MODEL_PATH=edge_node/models/yolo26m_vehicle.pt \
    TRAFFIC_LIGHT_MODEL_PATH=edge_node/models/traffic_light_cls.pt \
    NODE_ID=edge-node-01 \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    OUTBOX_ENABLED=true \
    ENABLE_OCR=true \
    TZ=Asia/Ho_Chi_Minh

EXPOSE 8080

# Default: start the pipeline with the API server
CMD ["python", "-m", "edge_node", "--start-api"]
