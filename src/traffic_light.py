import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

class TrafficLightClassifier:
    def __init__(self, model_path="models/mobilenetv2_traffic_light.h5", image_size=(32, 32)):
        self.image_size = image_size
        
        # Load the MobileNetV2 model
        try:
            self.model = load_model(model_path)
            # Compile the model to avoid the warning
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # Run a dummy prediction to build the metrics
            dummy_input = np.zeros((1, *self.model.input_shape[1:]))
            self.model.predict(dummy_input, verbose=0)
            print(f"Loaded traffic light classification model from {model_path}")
            # Define class mapping for model predictions
            self.class_mapping = {0: "green", 1: "red", 2: "yellow"}
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        # Color thresholds in HSV space
        self.color_ranges = {
            'red': [
                {'lower': np.array([0, 120, 70]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([170, 120, 70]), 'upper': np.array([180, 255, 255])}
            ],
            'green': [
                {'lower': np.array([40, 100, 70]), 'upper': np.array([80, 255, 255])}
            ],
            'yellow': [
                {'lower': np.array([15, 150, 150]), 'upper': np.array([35, 255, 255])}
            ]
        }

    def preprocess(self, image):
        """Resize and preprocess image for model input"""
        # For neural network - resize to expected input size
        model_input_shape = self.model.input_shape[1:3]
        model_input = cv2.resize(image, model_input_shape)
            
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
            
        # Apply MobileNetV2 preprocessing
        model_input = preprocess_input(model_input.astype(np.float32))
        
        return model_input

    def color_based_classification(self, image):
        """Classify traffic light based on color thresholding in HSV space"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get average saturation and value for quality check
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        avg_saturation = np.average(s_channel)
        avg_value = np.average(v_channel)
        
        # If image is too dark or unsaturated, return unknown
        if avg_value < 70 or avg_saturation < 70:
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
        if max_color[1] > 0.05:  # At least 5% of pixels should be of the color
            return max_color[0]
            
        return "unknown"

    def classify_with_model(self, rgb_image):
        """Classify using deep learning model"""
        model_input = self.preprocess(rgb_image)
        model_input = np.expand_dims(model_input, axis=0)
        predictions = self.model.predict(model_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        return self.class_mapping.get(predicted_class_idx, "unknown"), confidence

    def classify(self, rgb_image):
        """Hybrid classification combining both methods"""
        # Get predictions from both methods
        dl_prediction, dl_confidence = self.classify_with_model(rgb_image)
        cv_prediction = self.color_based_classification(rgb_image)
        
        # Decision logic
        if dl_prediction == cv_prediction:
            # Both methods agree
            return dl_prediction
        elif dl_confidence > 0.9:
            # Deep learning is very confident
            return dl_prediction
        elif cv_prediction != "unknown":
            # Trust color-based method if it has a clear detection
            return cv_prediction
        else:
            # Default to deep learning prediction
            return dl_prediction
