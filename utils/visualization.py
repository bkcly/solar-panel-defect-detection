import cv2
import numpy as np
from PIL import Image
import os
import json

# Import from local files
from useful_tools.highlight_defects import draw_defects, load_defect_colors, load_defect_names
from defect_detection.draw_bbox import draw_bbox_from_mask

# Load defect configurations
defect_colors_path = os.path.join('useful_tools', 'defect_colors.json')
defect_names_path = os.path.join('useful_tools', 'defect_name.json')

DEFECT_COLORS = load_defect_colors(defect_colors_path)
DEFECT_NAMES = load_defect_names(defect_names_path)

def draw_defect_boxes(image, mask):
    """
    Draw bounding boxes around detected defects using existing utilities
    Args:
        image: PIL Image or numpy array
        mask: Predicted mask from model (numpy array)
    Returns:
        Image with drawn bounding boxes and labels
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw defects using existing utility
    output = draw_defects(image.copy(), mask, DEFECT_COLORS)
    
    # Add bounding boxes
    output = draw_bbox_from_mask(output, mask, DEFECT_NAMES)
    
    return output

def get_defect_statistics(mask):
    """
    Get statistics of detected defects
    Args:
        mask: Predicted mask from model
    Returns:
        Dictionary with defect counts
    """
    stats = {}
    for class_idx, name in DEFECT_NAMES.items():
        if class_idx == 0:  # Skip background
            continue
        class_mask = (mask == class_idx)
        contours, _ = cv2.findContours(class_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        stats[name] = len(contours)
    
    return stats