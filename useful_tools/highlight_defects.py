import cv2 as cv
import json
import os
import numpy as np
from defect_detection.draw_bbox import draw_bbox_from_mask

def load_defect_colors(json_path):
    """Load defect colors from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_defect_names(json_path):
    """Load defect names from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def draw_defects(image, mask, colors, thickness=2):
    """
    Draw defects on the image using the segmentation mask
    Args:
        image: Input image (numpy array)
        mask: Segmentation mask (numpy array)
        colors: Dictionary mapping class indices to colors
        thickness: Line thickness for drawing
    Returns:
        Image with drawn defects
    """
    output = image.copy()
    
    # Draw each defect class
    for class_idx, color in colors.items():
        if isinstance(class_idx, str):
            class_idx = int(class_idx)
        
        # Get binary mask for current class
        class_mask = (mask == class_idx).astype(np.uint8)
        
        if class_mask.sum() > 0:
            # Find contours
            contours, _ = cv.findContours(class_mask, 
                                        cv.RETR_EXTERNAL, 
                                        cv.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            cv.drawContours(output, contours, -1, color, thickness)
    
    return output
