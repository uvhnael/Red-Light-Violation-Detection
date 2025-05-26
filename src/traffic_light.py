import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from src.config import TRAFFIC_LIGHT_COLORS, IMAGE_QUALITY, MODEL_CONFIDENCE
from src.model_cache import ModelCache

class TrafficLightClassifier:
    """
    A class for classifying traffic light states using both deep learning and color-based approaches.
    
    This classifier combines a ResNet18-based deep learning model with traditional
    computer vision techniques using color thresholding in HSV color space.
    
    Attributes:
        image_size (tuple): Target size for input images (height, width)
        model: Loaded PyTorch model for traffic light classification
        class_mapping (dict): Mapping from model output indices to class names
    """
    
    def __init__(self, model_path="models/resnet18_traffic_light.pth", image_size=(96, 96)):
        """
        Initialize the TrafficLightClassifier.
        
        Args:
            model_path (str): Path to the trained ResNet18 model file
            image_size (tuple): Target size for input images (height, width)
        """
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define class mapping for model predictions
        self.class_mapping = {0: "green", 1: "red", 2: "yellow"}
        
        # Load the model from cache
        try:
            self.model_cache = ModelCache()
            self.model = self.model_cache.get_traffic_light_classifier(model_path)
            print(f"Loaded traffic light classification model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        # Define transform for preprocessing images
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
        # Color thresholds from config
        self.color_ranges = {
            color: [
                {'lower': np.array(range_dict['lower']), 
                 'upper': np.array(range_dict['upper'])}
                for range_dict in ranges
            ]
            for color, ranges in TRAFFIC_LIGHT_COLORS.items()
        }

    def preprocess(self, image):
        """
        Resize and preprocess image for model input.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            torch.Tensor: Preprocessed image ready for model input
        """
        # Check if model is loaded
        if self.model is None:
            return None
            
        # Convert BGR to RGB and create PIL Image
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = Image.fromarray(image)
            
        # Apply transformations
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def color_based_classification(self, image):
        """
        Classify traffic light based on color thresholding in HSV space.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            str: Predicted traffic light color ("red", "green", "yellow", or "unknown")
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get average saturation and value for quality check
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        avg_saturation = np.average(s_channel)
        avg_value = np.average(v_channel)
        
        # If image is too dark or unsaturated, return unknown
        if avg_value < IMAGE_QUALITY['MIN_VALUE'] or avg_saturation < IMAGE_QUALITY['MIN_SATURATION']:
            return "unknown"
            
        color_scores = {}
        
        # Process each color using HSV masks
        for color, ranges in self.color_ranges.items():
            max_ratio = 0
            for range_dict in ranges:
                # Create mask using HSV thresholds
                mask = cv2.inRange(hsv, range_dict['lower'], range_dict['upper'])
                
                # Count non-zero pixels and calculate ratio
                non_zero_count = np.count_nonzero(mask)
                ratio = non_zero_count / mask.size
                max_ratio = max(max_ratio, ratio)
                
            color_scores[color] = max_ratio
        
        # Get color with highest ratio if it exceeds threshold
        max_color = max(color_scores.items(), key=lambda x: x[1])
        if max_color[1] > IMAGE_QUALITY['MIN_COLOR_RATIO']:
            return max_color[0]
            
        return "unknown"

    def classify_with_model(self, rgb_image):
        """
        Classify traffic light state using the deep learning model.
        
        Args:
            rgb_image (numpy.ndarray): Input image in RGB format
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        if self.model is None:
            return "unknown", 0.0
            
        model_input = self.preprocess(rgb_image)
        
        with torch.no_grad():
            outputs = self.model(model_input)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_class_idx = torch.max(probabilities, 0)
        
        return self.class_mapping.get(predicted_class_idx.item()), confidence.item()

    def classify(self, rgb_image):
        """
        Hybrid classification combining both deep learning and color-based methods.
        
        This method uses both the deep learning model and color-based classification
        to make a final decision about the traffic light state. It combines the strengths
        of both approaches for more robust classification.
        
        Args:
            rgb_image (numpy.ndarray): Input image in RGB format
            
        Returns:
            str: Predicted traffic light color ("red", "green", "yellow", or "unknown")
        """
        # Get color-based prediction
        cv_prediction = self.color_based_classification(rgb_image)
        
        # Get deep learning prediction
        dl_prediction, dl_confidence = self.classify_with_model(rgb_image)
        
        # Decision logic
        if dl_prediction == cv_prediction:
            # Both methods agree
            return dl_prediction
        elif dl_confidence > MODEL_CONFIDENCE['HIGH_CONFIDENCE']:
            # Deep learning is very confident
            return dl_prediction
        elif cv_prediction != "unknown":
            # Trust color-based method if it has a clear detection
            return cv_prediction
        else:
            # Default to deep learning prediction
            return cv_prediction
        # return dl_prediction