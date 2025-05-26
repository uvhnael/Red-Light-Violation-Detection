import numpy as np
import torch
import re
from src.model_cache import ModelCache


class LicensePlateRecognizer:
    def __init__(self):
        """
        Initializes the LicensePlateRecognizer with a specified model path.
        """
        self.model_cache = ModelCache()
        self.plate_detector = self.model_cache.get_plate_detector()
        self.plate_reader = self.model_cache.get_plate_reader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    def detect_and_recognize_plates(self, image, conf_threshold=0.25):
        """Detect and recognize the first valid license plate in the cropped vehicle image"""
        if image is None:
            print("Error: Invalid image")
            return None, None
        
        # Detect license plates
        results = self.plate_detector.predict(source=image, conf=conf_threshold, verbose=False, device=self.device)
        
        # Process each detected plate
        for result in results:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detection_conf = float(box.conf[0])
                
                # Crop the plate region with some padding
                try:
                    plate_img = self.crop_expanded_plate([x1, y1, x2, y2], image, 0.15)
                except:
                    continue
                    
                # Recognize text on the plate
                ocr_results = self.plate_reader.detect_and_ocr(plate_img)
                
                
                if len(ocr_results) > 0:
                    plate_text = ''
                    
                    # Combine all detected text
                    for result in ocr_results:
                        plate_text += result.text + ' '
                    
                    # Clean the text
                    plate_text = re.sub(r'[^A-Za-z0-9\-.]', '', plate_text)
                    
                    print(f"Detected plate text: {plate_text}")
                    
                    # Return first valid plate
                    # if self.check_legit_plate(plate_text):
                    #     return plate_text, plate_img
                    
                    return plate_text, plate_img
            
        # No valid plate found
        return None, None

    def crop_expanded_plate(self, plate_xyxy, cropped_vehicle, expand_ratio=0.1):
        # Original coordinates
        x_min, y_min, x_max, y_max = plate_xyxy

        # Calculate the width and height of the original cropping area
        width = x_max - x_min
        height = y_max - y_min

        # Calculate the expansion amount (10% of the width and height by default)
        expand_x = int(expand_ratio * width)
        expand_y = int(expand_ratio * height)

        # Calculate the new coordinates with expansion
        new_x_min = max(x_min - expand_x, 0)
        new_y_min = max(y_min - expand_y, 0)
        new_x_max = min(x_max + expand_x, cropped_vehicle.shape[1])
        new_y_max = min(y_max + expand_y, cropped_vehicle.shape[0])

        # Crop the expanded area
        cropped_plate = cropped_vehicle[new_y_min:new_y_max, new_x_min:new_x_max, :]

        return cropped_plate

    def check_legit_plate(self, s):
        # Remove unwanted characters
        s_cleaned = re.sub(r'[.\-\s]', '', s)

        # Regular expressions for different cases
        pattern1 = r'^[A-Za-z]{2}[0-9]{4}$'  # Matches exactly 2 letters followed by exactly 4 digits
        pattern2 = r'[A-Za-z][0-9]{4,}'      # Matches an alphabet character followed by at least 4 digits

        # Check if the cleaned string matches either pattern
        if re.search(pattern1, s_cleaned) or (re.search(pattern2, s_cleaned) and not re.match(r'^[A-Za-z]{2}', s_cleaned)):
            return True
        else:
            return False
        