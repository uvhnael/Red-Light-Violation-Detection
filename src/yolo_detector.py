# yolo_detector.py
from src.model_cache import ModelCache
import cv2
import warnings
import torch

# Filter out torchvision deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')

class YOLODetector:
    def __init__(self, model_path="models/yolo11l.pt", confidence=0.5):
        self.model_cache = ModelCache()
        self.model = self.model_cache.get_yolo_detector(model_path)
        self.conf = confidence
        self.class_names = self.model.names

    def detect(self, image):
        results = self.model.predict(source=image, conf=self.conf, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                label = self.class_names[cls]
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls,
                    "label": label
                })
        return detections