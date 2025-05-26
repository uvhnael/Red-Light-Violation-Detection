"""Configuration constants for traffic monitoring system"""

# Traffic light detection constants
import cv2


TRAFFIC_LIGHT_DETECTION = {
    'FRAME_SKIP': 5,
    'MAX_FRAMES': 100,
    'MIN_DETECTIONS': 10,
    'CONFIDENCE_THRESHOLD': 0.1,
    'TRAFFIC_LIGHT_CLASS_ID': 9
}

# Traffic light color thresholds in HSV space
TRAFFIC_LIGHT_COLORS = {
    'red': [
        {'lower': [0, 120, 70], 'upper': [10, 255, 255]},
        {'lower': [170, 120, 70], 'upper': [180, 255, 255]}
    ],
    'green': [
        {'lower': [40, 100, 70], 'upper': [80, 255, 255]}
    ],
    'yellow': [
        {'lower': [15, 150, 150], 'upper': [35, 255, 255]}
    ]
}

# Image quality thresholds
IMAGE_QUALITY = {
    'MIN_SATURATION': 70,
    'MIN_VALUE': 70,
    'MIN_COLOR_RATIO': 0.05
}

# Model confidence thresholds
MODEL_CONFIDENCE = {
    'HIGH_CONFIDENCE': 0.8,
    'DEFAULT_CONFIDENCE': 0.3
}

# Video processing
VIDEO_PROCESSING = {
    'RESIZE_WIDTH': 1000,
    'RESIZE_HEIGHT': 750,
    'DEFAULT_THRESHOLD': 0.3
}

# YOLO detection
YOLO_DETECTION = {
    'DEFAULT_CONFIDENCE': 0.5,
    'VEHICLE_CLASSES': [2, 3, 5, 7]  # car, motorcycle, truck
}

# Stop line detection
STOP_LINE_DETECTION = {
    'MIN_CONTOUR_AREA': 800,
    'MAX_CONTOUR_POINTS': 100,
    'APPROX_POLY_EPSILON': 0.04,
    'ADAPTIVE_THRESH_BLOCK_SIZE': 115,
    'ADAPTIVE_THRESH_C': 1,
    'ERODE_ITERATIONS': 1,
    'DILATE_ITERATIONS': 2
}

# UI Display
UI_DISPLAY = {
    'VIOLATION_ALERT_DURATION': 10,
    'FONT_SCALE': 1.5,
    'FONT': cv2.FONT_HERSHEY_SIMPLEX, 
    'LINE_THICKNESS': 4,
    'BOX_THICKNESS': 1,
    'COLORS': {
        'RED': (0, 0, 255),
        'GREEN': (0, 255, 0),
        'YELLOW': (0, 255, 255),
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0)
    }
}

# Progress logging
LOGGING = {
    'PROGRESS_INTERVAL': 100  # Log progress every N frames
} 