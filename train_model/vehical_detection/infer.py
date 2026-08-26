#!/usr/bin/env python3
"""
Chay thu model da train tren 1 anh / video / webcam.

    cd /home/uvhnael/projects/Red-Light-Violation-Detection/train_model/vehical_detection
    /home/uvhnael/projects/Red-Light-Violation-Detection/myenv/bin/python infer.py --source duong_dan_anh.jpg
    /home/uvhnael/projects/Red-Light-Violation-Detection/myenv/bin/python infer.py --source video.mp4 --save
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(HERE / "runs" / "vehicle_yolo26m" / "weights" / "best.pt"))
    parser.add_argument("--source", required=True, help="Anh, video, folder, or webcam (0)")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--save", action="store_true", help="Luu ket qua vao runs/predict/")
    parser.add_argument("--show", action="store_true", help="Hien cua so xem truc tiep")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=640,
        save=args.save,
        show=args.show,
        project=str(HERE / "runs" / "predict"),
        name="vehicle",
        exist_ok=True,
        device=0,
    )

    for r in results:
        if r.boxes is not None:
            names = model.names
            counts = {}
            for c in r.boxes.cls.tolist():
                label = names[int(c)]
                counts[label] = counts.get(label, 0) + 1
            print(counts)


if __name__ == "__main__":
    main()
