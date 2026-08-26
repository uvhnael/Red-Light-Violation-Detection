#!/usr/bin/env python3
"""
Validate model YOLO26m da train tren tap valid.
Chay sau khi train xong (hoac bat ky luc nao de do lai):

    cd /home/uvhnael/projects/Red-Light-Violation-Detection/train_model/vehical_detection
    /home/uvhnael/projects/Red-Light-Violation-Detection/myenv/bin/python validate.py
    # hoac chi dinh weights khac:
    /home/uvhnael/projects/Red-Light-Violation-Detection/myenv/bin/python validate.py --weights runs/vehicle_yolo26m/weights/last.pt
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(HERE / "runs" / "vehicle_yolo26m" / "weights" / "best.pt"),
        help="Duong dan den file .pt can validate",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(HERE / "data.yaml"),
        imgsz=640,
        batch=16,
        device=device,
        project=str(HERE / "runs" / "val"),
        name="vehicle_yolo26m",
        exist_ok=True,
    )

    box = metrics.box
    names = model.names
    print("\n===== Ket qua validation =====")
    print(f"mAP50-95: {box.map:.4f}")
    print(f"mAP50:    {box.map50:.4f}")
    print("mAP theo lop:")
    for i, cls_idx in enumerate(box.ap_class_index):
        print(f"  {names[int(cls_idx)]:>8}: mAP50 = {box.ap50[i].mean():.4f} | mAP50-95 = {box.ap[i].mean():.4f}")


if __name__ == "__main__":
    main()
