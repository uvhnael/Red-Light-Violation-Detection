import torch
from ultralytics import YOLO
from src.ppocr_onnx import DetAndRecONNXPipeline as PLateReader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

class ModelCache:
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._initialize_cache()
        return cls._instance
    
    def _initialize_cache(self):
        """Initialize the model cache with empty dictionaries"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model cache initialized using device: {self.device}")
    
    def get_yolo_detector(self, model_path="models/yolo12l.pt"):
        """Get or load YOLO detector model"""
        cache_key = f"yolo_detector_{model_path}"
        if cache_key not in self._models:
            print(f"Loading YOLO detector from {model_path}")
            model = YOLO(model_path)
            model.to(self.device)
            self._models[cache_key] = model
        return self._models[cache_key]
    
    def get_plate_detector(self, model_path="models/plate_yolov8n_320_2024.pt"):
        """Get or load license plate detector model"""
        cache_key = f"plate_detector_{model_path}"
        if cache_key not in self._models:
            print(f"Loading license plate detector from {model_path}")
            model = YOLO(model_path)
            model.to(self.device)
            self._models[cache_key] = model
        return self._models[cache_key]
    
    def get_plate_reader(self, det_model="models/ch_PP-OCRv4_det_infer.onnx", 
                        rec_model="models/ch_PP-OCRv4_rec_infer.onnx"):
        """Get or load PaddleOCR plate reader model"""
        cache_key = f"plate_reader_{det_model}_{rec_model}"
        if cache_key not in self._models:
            print(f"Loading PaddleOCR plate reader")
            model = PLateReader(
                text_det_onnx_model=det_model,
                text_rec_onnx_model=rec_model,
                box_thresh=0.6
            )
            self._models[cache_key] = model
        return self._models[cache_key]
    
    def get_traffic_light_classifier(self, model_path="models/resnet18_traffic_light.pth"):
        """Get or load traffic light classifier model"""
        cache_key = f"traffic_light_{model_path}"
        if cache_key not in self._models:
            print(f"Loading traffic light classifier from {model_path}")
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_ftrs, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3)  # 3 classes: red, yellow, green
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self._models[cache_key] = model
        return self._models[cache_key]
    
    def clear_cache(self):
        """Clear all models from cache"""
        self._models.clear()
        print("Model cache cleared") 