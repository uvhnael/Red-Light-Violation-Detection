#!/usr/bin/env python3
"""
Validate trained license plate detection model.
Tests the model on validation set and visualizes predictions.
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def validate_model(model_path, data_yaml="data.yaml"):
    """Run validation on the trained model."""
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        split="val",
        imgsz=640,
        batch=16,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        verbose=True,
    )
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results

def test_on_images(model_path, image_dir, output_dir="test_results"):
    """Run inference on test images and save visualizations."""
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
    print(f"\nTesting on {len(image_files)} images from {image_dir}")
    
    for img_path in image_files[:10]:  # Test first 10 images
        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            save=True,
            save_dir=output_dir,
            show_labels=True,
            show_conf=True,
            line_width=2,
        )
        
        # Print detections
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    print(f"  {img_path.name}: {label} ({conf:.2f})")
            else:
                print(f"  {img_path.name}: No detections")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    # Default to best model
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/runs/train/license_plate_yolo26/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Usage: python3 validate.py [model_path]")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    
    # Run validation
    validate_model(model_path)
    
    # Test on validation images
    test_on_images(model_path, "licenseplates/images/val")