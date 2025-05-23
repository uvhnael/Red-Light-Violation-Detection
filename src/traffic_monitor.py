import cv2
import numpy as np
import os
import time
import json
import csv
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from src.yolo_detector import YOLODetector
from src.traffic_light import TrafficLightClassifier
from src.trackers import VehicleTracker

@dataclass
class VehicleRecord:
    """Class to store vehicle tracking information and violations"""
    vehicle_id: int
    position_history: List[Tuple[Tuple[int, int, int, int], int]] = field(default_factory=list)
    is_violation: bool = False
    violation_frames: List[Tuple[Any, int]] = field(default_factory=list)

class TrafficMonitor:
    
    def __init__(self, 
                 input_path: str, 
                 output_path: str, 
                 model_path: str = "models/yolo12l.pt",
                 confidence: float = 0.5,
                 threshold: float = 0.3,
                 log_level: int = logging.INFO,
                 export_format: str = "json"):
        """
        Initialize the TrafficMonitor
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            model_path: Path to YOLO model file
            confidence: Minimum probability to filter weak detections
            threshold: Threshold for non-maxima suppression
            log_level: Logging level
            export_format: Format to export violation data ("json" or "csv")
        """
        # Setup logging
        self.setup_logging(log_level)
        
        # Save parameters
        self.input_path = input_path
        self.output_path = output_path
        self.confidence = confidence
        self.threshold = threshold
        self.export_format = export_format
        
        # Initialize video capture and properties
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_writer = None
        
        # Initialize models
        self.logger.info("Loading YOLO detector...")
        self.detector = YOLODetector(model_path=model_path, confidence=confidence)
        self.traffic_light_classifier = TrafficLightClassifier()
        
        # Initialize trackers
        self.vehicle_tracker = VehicleTracker()
        self.violation_tracker = VehicleTracker()
        
        # Initialize counters and data structures
        self.frame_count = 0
        self.display_violation_counter = 0
        self.violation_count = 0
        self.vehicle_records = []
        self.id_counter = 0
        self.processed_frames = []
        
        # Traffic light coordinates and stop line position
        self.traffic_light_coords = None
        self.stop_line_y = None
        
    def setup_logging(self, log_level):
        """Setup logging configuration"""
        # Get logger without any configuration - rely on root logger's configuration
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
    
    def detect_traffic_light(self, frame_skip=5, max_frames=100, min_detections=10):
        """
        Detect and return traffic light coordinates in the video
        
        Args:
            frame_skip: Number of frames to skip (process every nth frame)
            max_frames: Maximum number of frames to process
            min_detections: Minimum number of detections before early stopping
        """
        self.logger.info("Detecting traffic light position...")
        cap = cv2.VideoCapture(self.input_path)
        
        # list of the xyxy coordinates of the traffic light
        traffic_light_coords = []
        frame_count = 0
        processed_frames = 0
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Skip frames to improve speed
            if frame_count % frame_skip != 0:
                continue
                
            processed_frames += 1
            frame = cv2.resize(frame, (1000, 750))
            
            # Use YOLODetector to detect traffic lights (class ID 9)
            detections = self.detector.detect(frame)
            
            for det in detections:
                class_id = det["class_id"]
                confidence = det["confidence"]
                box = det["box"]
                # If detected traffic light with confidence > 0.1
                if class_id == 9 and confidence > 0.1:
                    x, y, x2, y2 = box
                    # Save the coordinates of the traffic light
                    traffic_light_coords.append((x, y, x2, y2))
            
            # Early stopping if we have enough detections
            if len(traffic_light_coords) >= min_detections:
                self.logger.info(f"Found {len(traffic_light_coords)} traffic light detections, stopping early")
                break
                    
        cap.release()
        
        # Process detected traffic lights
        if traffic_light_coords:
            self.logger.info(f"Processing {len(traffic_light_coords)} traffic light detections")
            
            # Cluster detections to find the most common coordinates
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            # Convert to numpy array for clustering
            coords_array = np.array(traffic_light_coords)
            
            # Use DBSCAN to cluster close detections
            if len(coords_array) >= 5:  # Need sufficient points for clustering
                clustering = DBSCAN(eps=50, min_samples=3).fit(coords_array)
                labels = clustering.labels_
                
                # Find the largest cluster
                unique_labels = set(labels)
                max_cluster_size = 0
                max_cluster_label = -1
                
                for label in unique_labels:
                    if label == -1:  # Noise points
                        continue
                    cluster_size = np.sum(labels == label)
                    if cluster_size > max_cluster_size:
                        max_cluster_size = cluster_size
                        max_cluster_label = label
                
                if max_cluster_label != -1:
                    # Get coordinates from the largest cluster
                    cluster_coords = coords_array[labels == max_cluster_label]
                    
                    # Calculate bounding box that contains all points in the cluster
                    x1 = np.min(cluster_coords[:, 0])
                    y1 = np.min(cluster_coords[:, 1])
                    x2 = np.max(cluster_coords[:, 2])
                    y2 = np.max(cluster_coords[:, 3])
                    
                    w = (x2 - x1)
                    h = (y2 - y1)
                    
                    self.logger.info(f"Traffic light detected at: ({x1}, {y1}, {w}, {h})")
                    return (int(x1), int(y1), int(w), int(h))
            
            # Fallback to original method if clustering doesn't work
            x1, y1, x2, y2 = traffic_light_coords[0]
            for coords in traffic_light_coords[1:]:
                x, y, x2_new, y2_new = coords
                x1 = min(x1, x)
                y1 = min(y1, y)
                x2 = max(x2, x2_new)
                y2 = max(y2, y2_new)
                
            w = (x2 - x1)
            h = (y2 - y1)
            self.logger.info(f"Traffic light detected at: ({x1}, {y1}, {w}, {h})")
            return (x1, y1, w, h)
            
        self.logger.warning("No traffic light detected")
        return None, None, None, None
        
    def detect_stop_line(self, show_stages=False):
        """Detect and return the y-coordinate of the stop line"""
        self.logger.info("Detecting stop line position...")
        
        if self.traffic_light_coords is None:
            self.logger.error("Traffic light coordinates not detected")
            return 0
            
        xlight, ylight, wlight, hlight = self.traffic_light_coords
        
        cap = cv2.VideoCapture(self.input_path)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            self.logger.error("Could not read frame for stop line detection")
            return 0
            
        frame = cv2.resize(frame, (1000, 750))
        if show_stages:
            cv2.imshow('Original Frame', frame)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        temp = frame.copy()
        temp2 = frame.copy()
        
        # Convert image to grayscale
        grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        th = cv2.adaptiveThreshold(grayscaled, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 115, 1)
        kernel = np.ones((3, 3), np.uint8)
        
        # Erode and dilate to filter noise
        th = cv2.erode(th, kernel, iterations=1)
        th = cv2.dilate(th, kernel, iterations=2)
        
        if show_stages:
            cv2.imshow('Threshold Image', th)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        # Find contours in the processed image
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        
        all_contours = []
        
        # Filter contours to find suitable rectangles
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 800 and len(contour) < 100:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                if len(approx) == 4:  # Find contours with 4 sides (rectangles)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.drawContours(frame, contours, i, (0, 255, 0), 3)
                    cv2.rectangle(temp, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    all_contours.append((x, y, w, h))
        
        cv2.drawContours(temp2, contours, -1, (0, 255, 0), 3)
        
        if show_stages:
            cv2.imshow('All Contours', temp2)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('Detected Rectangle Contours', frame)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('Bounding Boxes', temp)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        min_index = 0
        min_distance = float('inf')
        
        # delete the rectangle that upper than the traffic light
        all_contours = [rect for rect in all_contours if rect[1] > ylight + hlight]
        if not all_contours:
            self.logger.warning("No suitable rectangles found above the traffic light")
            cap.release()
            return 0
        
        # Find the rectangle closest to the traffic light (possible stop line)
        for i, rect in enumerate(all_contours):
            x, y, w, h = rect
            if ylight + wlight < y:
                cv2.line(temp, (xlight, ylight), (x, y), (0, 0, 255), 2)
                distance = ((x-xlight)**2 + (y-ylight)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
        
        if all_contours:
            x, y, w, h = all_contours[min_index]
            if show_stages:
                cv2.imshow('Distance Visualization', temp)
                cv2.waitKey()
                cv2.destroyAllWindows()
                cv2.line(temp, (0, y), (1300, y), (0, 0, 0), 4, cv2.LINE_AA)
                cv2.imshow('Stop Line', temp)
                cv2.waitKey()
                cv2.destroyAllWindows()
            cap.release()
            self.logger.info(f"Stop line detected at y={y}")
            return y  # Return y-coordinate of the stop line
        
        cap.release()
        self.logger.warning("No stop line detected")
        return 0  # Return 0 if no stop line found
    
    def initialize(self):
        """Initialize traffic light coords and stop line position"""
        self.traffic_light_coords = self.detect_traffic_light()
        self.stop_line_y = self.detect_stop_line(show_stages=True)
        self.logger.info(f"Initialization complete. Traffic light: {self.traffic_light_coords}, Stop line: {self.stop_line_y}")
    
    def _detect_vehicles(self, frame):
        """Detect vehicles in the frame"""
        detections = self.detector.detect(frame)
        vehicle_detections = []
        for det in detections:
            class_id = det["class_id"]
            confidence = det["confidence"]
            box = det["box"]
            if class_id in [2, 3, 7] and confidence > self.confidence:
                x1, y1, x2, y2 = box
                conf = confidence
                vehicle_detections.append([x1, y1, x2, y2, conf, class_id])
        return vehicle_detections
    
    def _track_vehicles(self, frame, vehicle_detections):
        """Track detected vehicles and update vehicle records"""
        active_boxes, track_ids = self.vehicle_tracker.update(frame, vehicle_detections, self.stop_line_y)
        
        # Update vehicle records
        for box, track_id in zip(active_boxes, track_ids):
            found = False
            for record in self.vehicle_records:
                if record.vehicle_id == track_id:
                    record.position_history.append((box, self.frame_count))
                    found = True
                    break
            if not found:
                new_record = VehicleRecord(
                    vehicle_id=track_id,
                    position_history=[(box, self.frame_count)]
                )
                self.vehicle_records.append(new_record)
                
        return active_boxes, track_ids
    
    def _classify_traffic_light(self, frame):
        """Get and classify the traffic light color"""
        light_color = "unknown"
        if self.traffic_light_coords is not None:
            x, y, w, h = self.traffic_light_coords
            if y is not None and h is not None and x is not None and w is not None:
                if (y >= 0 and y + h <= frame.shape[0] and 
                    x >= 0 and x + w <= frame.shape[1]):
                    light_region = frame[y:y+h, x:x+w]
                    if light_region.size > 0:
                        # b, g, r = cv2.split(light_region)
                        # rgb_light = cv2.merge([r, g, b])
                        light_color = self.traffic_light_classifier.classify(light_region)
        return light_color
    
    def _check_violations(self, active_boxes, track_ids, light_color):
        """Check for red light violations"""
        violation_boxes = []
        violation_ids = []
        
        if light_color == "red":
            for box, track_id in zip(active_boxes, track_ids):
                x, y, w, h = box
                y_mid = y + h/2
                if y_mid < self.stop_line_y:
                    # Find vehicle record
                    for record in self.vehicle_records:
                        if record.vehicle_id == track_id:
                            # Only count as violation if this is first time for this ID
                            if not record.is_violation:
                                self.violation_count += 1
                                self.display_violation_counter = 10
                                record.is_violation = True
                                # Inform the vehicle tracker that this ID is a violator
                                self.vehicle_tracker.add_violation_id(track_id)
                            
                            # Always add the frame to violation frames
                            record.violation_frames.append((box, self.frame_count))
                            violation_boxes.append(box)
                            violation_ids.append(track_id)
                            break
        
        # Always add existing violators to the violation boxes
        # regardless of traffic light color
        for box, track_id in zip(active_boxes, track_ids):
            if track_id not in violation_ids:  # Avoid duplicates
                for record in self.vehicle_records:
                    if record.vehicle_id == track_id and record.is_violation:
                        violation_boxes.append(box)
                        violation_ids.append(track_id)
                        # Ensure the tracker knows about this violation ID
                        self.vehicle_tracker.add_violation_id(track_id)
                        break
                               
        return violation_boxes
    
    def _draw_annotations(self, frame, active_boxes, violation_boxes, light_color):
        """Draw bounding boxes, traffic light status, and violation count"""
        # Draw stop line
        if self.stop_line_y:
            cv2.line(frame, (0, self.stop_line_y), (1300, self.stop_line_y), 
                    (0, 0, 0), 4, cv2.LINE_AA)
        
        # Display traffic light status
        if self.traffic_light_coords is not None:
            x, y, w, h = self.traffic_light_coords
            if y is not None:
            
                color_map = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
                cv2.putText(frame, light_color, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_map.get(light_color, (255, 255, 255)), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_map.get(light_color, (255, 255, 255)), 1)

        # Display violation counter
        cv2.putText(frame, f'Violation Counter: {self.violation_count}', (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)
        
        # Display violation alert
        if self.display_violation_counter > 0:
            cv2.putText(frame, 'Violation', (30, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            self.display_violation_counter -= 1
        
        # Get list of boxes that are violations
        violation_boxes_set = set((box[0], box[1], box[2], box[3]) for box in violation_boxes)
        
        # Draw boxes for tracked vehicles (green for non-violators, red for violators)
        for box in active_boxes:
            x, y, w, h = box
            box_tuple = (x, y, w, h)
            
            # If this box is in violation_boxes, it will be drawn as red later
            if box_tuple not in violation_boxes_set:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw boxes for violations (red)
        for box in violation_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
    def process_frame(self, frame):
        """Process a single frame for vehicle detection and tracking"""
        frame = cv2.resize(frame, (1000, 750))
        frame_with_annotations = frame.copy()

        # Detect vehicles
        vehicle_detections = self._detect_vehicles(frame)
        
        # Track vehicles
        active_boxes, track_ids = self._track_vehicles(frame, vehicle_detections)
        
        # Classify traffic light
        light_color = self._classify_traffic_light(frame)
        
        # Check for violations
        violation_boxes = self._check_violations(active_boxes, track_ids, light_color)

        # Draw annotations on frame
        self._draw_annotations(
            frame_with_annotations, 
            active_boxes, 
            violation_boxes, 
            light_color
        )

        return frame_with_annotations
    
    def process(self):
        """Process the entire video"""
        self.logger.info(f"Processing video with {self.total_frames} frames")
        start_time = time.time()
        
        # Initialize traffic light and stop line
        self.initialize()
        
        # Get dimensions from resized frame (1000x750)
        output_size = (1000, 750)
        
        # Use mp4v codec for MP4 files, XVID for AVI
        output_ext = os.path.splitext(self.output_path)[1].lower()
        if output_ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, output_size
        )

        
        # Main processing loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Store the processed frame
            self.video_writer.write(processed_frame)

            
            # Display frame
            cv2.imshow('Traffic Monitoring', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.frame_count += 1
            
            # Log progress every 100 frames
            if self.frame_count % 100 == 0:
                progress = (self.frame_count / self.total_frames) * 100
                self.logger.info(f"Processing progress: {progress:.2f}% ({self.frame_count}/{self.total_frames})")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.video_writer:
            self.video_writer.release()
        
        # Write violations to file
        self.export_violations()
        
        end_time = time.time()
        self.logger.info(f"Processing complete. Total time: {end_time - start_time:.2f} seconds")
        
    def export_violations(self):
        """Export violation data to JSON or CSV file"""
        if not self.vehicle_records:
            self.logger.warning("No violation data to export")
            return
            
        violations = [
            {
                "vehicle_id": record.vehicle_id,
                "violation_frames": [frame for _, frame in record.violation_frames],
                "first_detected_frame": record.position_history[0][1] if record.position_history else None,
                "total_tracked_frames": len(record.position_history)
            }
            for record in self.vehicle_records if record.is_violation
        ]
        
        if not violations:
            self.logger.info("No violations detected")
            return
            
        base_name = os.path.splitext(self.output_path)[0]
        
        if self.export_format.lower() == 'json':
            export_path = f"{base_name}_violations.json"
            with open(export_path, 'w') as f:
                json.dump(violations, f, indent=4)
            self.logger.info(f"Exported {len(violations)} violations to {export_path}")
            
        elif self.export_format.lower() == 'csv':
            export_path = f"{base_name}_violations.csv"
            with open(export_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["vehicle_id", "violation_frames", "first_detected_frame", "total_tracked_frames"])
                for v in violations:
                    writer.writerow([
                        v["vehicle_id"],
                        ",".join(map(str, v["violation_frames"])),
                        v["first_detected_frame"],
                        v["total_tracked_frames"]
                    ])
            self.logger.info(f"Exported {len(violations)} violations to {export_path}")
            
        else:
            self.logger.warning(f"Unsupported export format: {self.export_format}")
    
    def is_violator_id(self, track_id):
        """Check if a track ID belongs to a vehicle that has committed a violation"""
        for record in self.vehicle_records:
            if record.vehicle_id == track_id and record.is_violation:
                return True
        return False
