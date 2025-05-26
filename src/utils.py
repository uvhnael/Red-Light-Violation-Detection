import cv2
import numpy as np

def draw_vehicle_box(image, box, color, labels=None):
    """
    Draws a bounding box on the image.

    Args:
        image (np.ndarray): The image on which to draw the box.
        box (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
        color (tuple): Color for the box in BGR format.
        labels (list of str, optional): List of labels corresponding to each box. Defaults to None.

    Returns:
        np.ndarray: The image with drawn bounding box.
    """
    overlay = image.copy()
    alpha = 0.2  # Transparency factor: 0 is transparent, 1 is opaque

    x, y, w, h = box
    # Fill the rectangle with color
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    # Apply the overlay with transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    # Draw the border separately (solid, not transparent)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Optionally draw label
    if labels is not None and len(labels) > 0:
        label = labels[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), (0, 0, 0), -1)
        cv2.putText(image, label, (x, y - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return image