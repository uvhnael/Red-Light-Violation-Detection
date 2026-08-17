#!/usr/bin/env python3
"""
Train YOLO26 model on license plate dataset.
Detects two classes: BSD (biển số đen - black plate) and BSV (biển số vàng - yellow plate).
"""

from ultralytics import YOLO
import torch

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load the YOLO26 model
    model = YOLO("yolo26m.pt")
    
    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        save=True,
        save_period=10,
        project="runs/train",
        name="license_plate_yolo26",
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
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
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
        auto_augment="randaugment",
        erasing=0.4,
        crop_fraction=1.0,
    )
    
    print("Training completed!")
    print(f"Best model saved to: runs/train/license_plate_yolo26/weights/best.pt")
    print(f"Last model saved to: runs/train/license_plate_yolo26/weights/last.pt")

if __name__ == "__main__":
    main()