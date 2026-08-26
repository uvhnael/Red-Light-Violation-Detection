#!/usr/bin/env python3
"""Train a YOLO26-nano classifier on the 3-class LISA traffic-light dataset.

Classes: red / yellow / green (merged from the 7 original LISA labels by
prepare_data.py).  The crops are tiny (10-87 px tall), so we train at
imgsz=64 — upscaling to 224 adds no information and only costs time.

Usage:
    python3 train_traffic_light/train.py [--epochs 50] [--imgsz 64] [--batch 64]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "yolo26n-cls.pt"))
    parser.add_argument("--data", default=str(HERE / "dataset"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detect if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=10,
        project=str(HERE / "runs"),
        name="traffic_light_cls",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        # mild augmentation — crops are already tight, heavy geometric
        # augmentation would destroy the lamp layout
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.0,   # flipping a traffic light vertically changes its meaning
        flipud=0.0,
        erasing=0.2,
    )

    best = HERE / "runs" / "traffic_light_cls" / "weights" / "best.pt"
    print("\nTraining completed!")
    print(f"Best model: {best}")


if __name__ == "__main__":
    main()
