# Traffic Light Violation Detection System

This system detects and records vehicles that violate traffic light signals. It uses computer vision techniques to identify traffic lights, classify their state (red, yellow, green), detect vehicles, and monitor if they cross the stop line during a red light.

## Features

- Traffic light detection and state classification
- Vehicle detection and tracking
- Red light violation detection
- Exports violation data to JSON or CSV
- Logs process information
- Creates annotated output video showing violations

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLO model (if not included) and place it in the `models` directory

## Usage

Run the main script with appropriate arguments:

```
python main.py --input videos/traffic.mp4 --output output/result.avi --model models/yolo12l.pt
```

### Command Line Arguments

- `--input`: Path to input video file (required)
- `--output`: Path to output video file (required)
- `--model`: Path to YOLO model file (default: models/yolo12l.pt)
- `--confidence`: Minimum confidence threshold for detections (default: 0.5)
- `--threshold`: Non-maxima suppression threshold (default: 0.3)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--export`: Format for exporting violation data (json or csv)

## Project Structure

- `main.py`: Entry point for the application
- `traffic_monitor.py`: Main class for traffic monitoring and violation detection
- `traffic_light.py`: Traffic light classification
- `yolo_detector.py`: YOLO-based object detection
- `trackers.py`: Vehicle tracking using OpenCV's MultiTracker
- `utils.py`: Utility functions

## How it Works

1. The system first detects the traffic light in the video
2. It identifies the stop line position
3. For each frame, it:
   - Tracks previously detected vehicles
   - Periodically detects new vehicles
   - Classifies the traffic light state
   - Monitors vehicles crossing the stop line during red lights
   - Records violations
4. After processing, it generates an output video with annotations
5. It exports violation data to JSON or CSV format #