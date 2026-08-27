from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import asdict
from typing import Any, Dict

import requests

from edge_node.settings import EdgeNodeSettings

LOGGER = logging.getLogger(__name__)


def _detect_ip_address(fallback: str = "") -> str:
    if fallback:
        return fallback
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return fallback or "unknown"


def build_registration_payload(settings: EdgeNodeSettings) -> Dict[str, Any]:
    values = asdict(settings)
    return {
        "node_id": settings.node_id,
        "name": settings.node_name or settings.node_id,
        "ip_address": _detect_ip_address(settings.node_ip_address),
        "status": settings.node_status,
        "settings": {
            "central_server_url": values["central_server_url"],
            "node_register_url": values["node_register_url"],
            "video_input": values["video_input"],
            "video_loop": values["video_loop"],
            "video_realtime": values["video_realtime"],
            "video_max_lag_ms": values["video_max_lag_ms"],
            "yolo_model_path": values["yolo_model_path"],
            "yolo_confidence": values["yolo_confidence"],
            "yolo_img_size": values["yolo_img_size"],
            "yolo_device": values["yolo_device"],
            "yolo_fp16": values["yolo_fp16"],
            "outbox_enabled": values["outbox_enabled"],
            "enable_ocr": values["enable_ocr"],
            "ocr_model_name": values["ocr_model_name"],
            "ocr_device": values["ocr_device"],
            "api_host": values["api_host"],
            "api_port": values["api_port"],
            "api_token": values["api_token"],
            "heartbeat_interval_seconds": values["node_heartbeat_interval_seconds"],
        },
    }


def register_node(settings: EdgeNodeSettings) -> None:
    payload = build_registration_payload(settings)
    try:
        response = requests.post(settings.node_register_url, json=payload, timeout=settings.push_timeout_seconds)
        response.raise_for_status()
        LOGGER.info("Registered edge node %s at %s", settings.node_id, settings.node_register_url)
    except Exception as exc:
        LOGGER.warning("Failed to register edge node %s: %s", settings.node_id, exc)


def start_registration_heartbeat(settings: EdgeNodeSettings) -> threading.Thread:
    def _heartbeat() -> None:
        while True:
            register_node(settings)
            time.sleep(max(10, settings.node_heartbeat_interval_seconds))

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    return thread