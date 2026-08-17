#!/usr/bin/env bash
# Cài nvidia-container-toolkit cho Docker trên Ubuntu
# Chạy: sudo bash scripts/install_nvidia_container_toolkit.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Cần chạy bằng sudo: sudo bash $0" >&2
  exit 1
fi

echo "==> Thêm NVIDIA container repo..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "==> apt update + install nvidia-container-toolkit..."
apt-get update -qq
apt-get install -y nvidia-container-toolkit

echo "==> Cấu hình Docker runtime..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "==> Kiểm tra..."
docker info 2>/dev/null | grep -i "Runtimes" || true
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "==> XONG! Docker đã hỗ trợ GPU."
