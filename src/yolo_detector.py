# yolo_detector.py
from ultralytics import YOLO
import cv2
import warnings

# Filter out torchvision deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')

class YOLODetector:
    def __init__(self, model_path="models/yolo11l.pt", confidence=0.5):
        self.model = YOLO(model_path)
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

    def draw(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f'{det["label"]} {det["confidence"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
