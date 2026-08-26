#!/usr/bin/env python3
"""
Train YOLO26m on vehicle detection dataset (Roboflow vehicle.v1i.yolo26).
4 classes: car, bike, van/bus, truck.

Chay bang myenv:
    cd /home/uvhnael/projects/Red-Light-Violation-Detection/train_model/vehical_detection
    /home/uvhnael/projects/Red-Light-Violation-Detection/myenv/bin/python train.py

Smoke test nhanh tren mot phan nho du lieu:
    /home/uvhnael/projects/Red-Light-Violation-Detection/myenv/bin/python train.py --epochs 1 --fraction 0.01
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "data.yaml"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0, help="Train tren mot phan dataset (0.01 = 1%%)")
    p.add_argument("--name", default="vehicle_yolo26m")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(HERE / "yolo26m.pt"))

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,       # Neu het VRAM tren 6GB thi giam xuong 8 hoac 4
        fraction=args.fraction,
        device=device,
        workers=4,
        patience=20,
        save=True,
        save_period=10,
        cache=False,            # dataset ~8k anh, de False cho an toan RAM/VRAM
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        nbs=64,
        # Augmentation phu hop camera giao thong
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        amp=True,               # mixed precision - tiet kiem VRAM tren 6GB
    )

    weights_dir = HERE / "runs" / args.name / "weights"
    print("Training completed!")
    print(f"Best model: {weights_dir / 'best.pt'}")
    print(f"Last model: {weights_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
