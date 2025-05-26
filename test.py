from src.license_plate import LicensePlateRecognizer
import cv2


license_plate_recognizer = LicensePlateRecognizer()

img_path = "data/imgs/1.jpg"
image = cv2.imread(img_path)
if image is None:
    print(f"Error: Could not load image from {img_path}")
    exit()
    
print(license_plate_recognizer.plate_detector.names)

plate_text, plate_img = license_plate_recognizer.detect_and_recognize_plates(image, conf_threshold=0.25)

if plate_text:
    print(f"Detected plate text: {plate_text}")
else:
    print("No valid plate detected.")