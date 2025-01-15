import cv2 as cv
import numpy as np

def draw_bbox_from_mask(image, mask, defect_names, thickness=2):
    """
    Draw bounding boxes on image from segmentation mask
    Args:
        image: Input image (numpy array)
        mask: Segmentation mask (numpy array)
        defect_names: Dictionary mapping class indices to names
        thickness: Line thickness for drawing
    Returns:
        Image with drawn bounding boxes and labels
    """
    output = image.copy()
    
    # Process each defect class
    for class_idx, name in defect_names.items():
        if isinstance(class_idx, str):
            class_idx = int(class_idx)
            
        if class_idx == 0:  # Skip background
            continue
            
        # Get binary mask for current class
        class_mask = (mask == class_idx).astype(np.uint8)
        
        if class_mask.sum() > 0:
            # Find contours
            contours, _ = cv.findContours(class_mask, 
                                        cv.RETR_EXTERNAL, 
                                        cv.CHAIN_APPROX_SIMPLE)
            
            # Draw bounding box for each contour
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), thickness)
                
                # Add label
                label = f"{name}"
                (text_width, text_height), _ = cv.getTextSize(label, 
                                                            cv.FONT_HERSHEY_SIMPLEX,
                                                            0.5, thickness)
                cv.rectangle(output, 
                           (x, y - text_height - 5),
                           (x + text_width, y),
                           (0, 255, 0), -1)
                cv.putText(output, label,
                          (x, y - 5),
                          cv.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 0, 0), thickness)
    
    return output