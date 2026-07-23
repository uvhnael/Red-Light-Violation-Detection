# ---- Edge Node Dockerfile ----
# Build context should be the project root (parent of edge_node/).
# Usage: docker build -f edge_node/Dockerfile .

FROM python:3.11-slim AS base

# System deps for OpenCV and video
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY edge_node/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY edge_node/ edge_node/

# Default env-vars (override via docker-compose or .env)
ENV REDIS_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/1 \
    CENTRAL_SERVER_URL=http://central-server:8000/api/violations \
    YOLO_MODEL_PATH=models/yolov8s.pt \
    NODE_ID=edge-node-01 \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    ENABLE_QUEUE=true

EXPOSE 8080

# Default: start the pipeline with the API server
CMD ["python", "-m", "edge_node", "--start-api"]
