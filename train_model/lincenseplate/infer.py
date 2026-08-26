#!/usr/bin/env python3
"""
Quick inference script for license plate detection.
Usage: python3 infer.py <image_path> [model_path]
"""

import sys
from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 infer.py <image_path> [model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "runs/detect/runs/train/license_plate_yolo26/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,
        show_labels=True,
        show_conf=True,
    )
    
    print(f"\nDetections for {image_path}:")
    for r in results:
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                print(f"  {label}: {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print("  No detections")
    
    print(f"\nResult saved to: runs/detect/predict/")

if __name__ == "__main__":
    import os
    main()