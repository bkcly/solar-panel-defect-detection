import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class FixResize:
    """UNet requires input size to be multiple of 16"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return F.resize(image, (self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR)

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        FixResize(256),  # Match the size used in training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB if image is from OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Add batch dimension
    return transform(image).unsqueeze(0)

def analyze_defects(mask):
    """Analyze defects in the segmentation mask
    
    Args:
        mask (np.ndarray): Segmentation mask with class indices
        
    Returns:
        dict: Dictionary containing defect statistics
            - counts: Number of instances per defect type
            - areas: Area of each defect instance
            - severity: Severity score based on defect size and type
    """
    defect_stats = {
        'counts': {
            'Crack': 0,
            'Dark Area': 0,
            'Cross': 0,
            'Busbar': 0
        },
        'areas': {
            'Crack': [],
            'Dark Area': [],
            'Cross': [],
            'Busbar': []
        },
        'severity': 0.0
    }
    
    # Class index to name mapping
    idx_to_name = {
        1: 'Crack',
        2: 'Dark Area',
        3: 'Cross',
        4: 'Busbar'
    }
    
    # Severity weights for each defect type
    severity_weights = {
        'Crack': 1.0,
        'Dark Area': 0.8,
        'Cross': 0.6,
        'Busbar': 0.4
    }
    
    for class_idx in range(1, 5):  # Skip background (0)
        # Get binary mask for current class
        class_mask = (mask == class_idx).astype(np.uint8)
        
        if class_mask.sum() > 0:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask)
            
            defect_name = idx_to_name[class_idx]
            defect_stats['counts'][defect_name] = num_labels - 1  # Subtract 1 to exclude background
            
            # Calculate areas and severity
            for i in range(1, num_labels):  # Skip background label
                area = stats[i, cv2.CC_STAT_AREA]
                defect_stats['areas'][defect_name].append(area)
                
                # Calculate severity score based on area and defect type
                severity = area * severity_weights[defect_name]
                defect_stats['severity'] += severity
    
    # Normalize severity score
    total_pixels = mask.shape[0] * mask.shape[1]
    defect_stats['severity'] = min(1.0, defect_stats['severity'] / total_pixels)
    
    return defect_stats