import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from useful_tools.seg_cnn import SegmentationCNN

class DefectDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        print(f"Loading weights from: {model_path}")
        self.model = SegmentationCNN(num_classes=5)
        
        # Load state dict and remove 'module.' prefix
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[name] = v
        print("State dict keys:", new_state_dict.keys())
        
        # Load weights with strict=True to ensure all layers are properly mapped
        self.model.load_state_dict(new_state_dict, strict=True)
        print("Model structure:")
        print(self.model)
        self.model.to(self.device)
        self.model.eval()
        
        # Class mapping from config.json
        self.class_map = ['bg', 'busbar', 'crack', 'cross', 'dark']
        
        # Color mapping from config.json
        self.colors = {
            'bg': [0, 0, 0],
            'busbar': [138, 100, 15],    # From config.json #8A640F
            'crack': [138, 15, 89],      # From config.json #8A0F59  
            'cross': [15, 138, 119],     # From config.json #0F8A77
            'dark': [68, 138, 15]        # From config.json #448A0F
        }
        
        # Class-specific thresholds based on training data statistics
        self.thresholds = {
            'busbar': {
                'prob_thresh': 0.3,    # Relative threshold above background
                'min_area': 200,       # Minimum area in pixels
                'max_coverage': 0.4    # Maximum coverage as fraction
            },
            'crack': {
                'prob_thresh': 0.2,    # Relative threshold above background
                'min_area': 100,       # Minimum area in pixels
                'max_coverage': 0.2    # Maximum coverage as fraction
            },
            'cross': {
                'prob_thresh': 0.25,   # Relative threshold above background
                'min_area': 50,        # Minimum area in pixels
                'max_coverage': 0.1    # Maximum coverage as fraction
            },
            'dark': {
                'prob_thresh': 0.25,   # Relative threshold above background
                'min_area': 100,       # Minimum area in pixels
                'max_coverage': 0.2    # Maximum coverage as fraction
            }
        }
        
        # Set up preprocessing to match training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Only normalize first channel
        ])
        
    def predict(self, image):
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Ensure image is in range [0, 255]
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Preprocess
        x = self.transform(image)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.repeat(1, 3, 1, 1)  # Repeat grayscale channel to match model input
        
        print("Input tensor shape:", x.shape)
        print("Input tensor range:", [f"{x.min():.2f}", f"{x.max():.2f}"])
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            print("Raw output shape:", logits.shape)
            print("Raw output range:", [f"{logits.min():.2f}", f"{logits.max():.2f}"])
            
            # Get probabilities with temperature scaling
            temperature = 0.5  # Sharper predictions
            probs = F.softmax(logits / temperature, dim=1)
            predictions = probs[0].cpu().numpy()
            
            # Print per-class probabilities
            print("Class probabilities:", {self.class_map[i]: f"{predictions[i].mean():.3f}" 
                                         for i in range(len(self.class_map))})
            
            # Create visualization image - maintain original size
            vis_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            h, w = vis_image.shape[:2]
            found_defects = []
            
            # Create segmentation mask at model output size
            mask = np.zeros((*predictions.shape[1:], 3), dtype=np.uint8)
            
            # Get background probability for relative thresholding
            bg_prob = predictions[0]
            
            # Process each class (skip background)
            for i in range(1, len(self.class_map)):
                class_name = self.class_map[i]
                thresh = self.thresholds[class_name]
                
                # Get class mask where probability exceeds background by threshold
                class_prob = predictions[i]
                prob_diff = class_prob - bg_prob
                class_mask = prob_diff > thresh['prob_thresh']
                
                if np.any(class_mask):
                    # Apply morphological operations to clean up the mask
                    kernel = np.ones((3,3), np.uint8)
                    class_mask = cv2.morphologyEx(class_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                    class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
                    
                    # Calculate area and coverage
                    area = np.sum(class_mask)
                    total_pixels = predictions.shape[1] * predictions.shape[2]
                    coverage = float(area) / total_pixels
                    
                    # Check if area and coverage are within acceptable ranges
                    if area > thresh['min_area'] and coverage < thresh['max_coverage']:
                        print(f"Found {class_name} defect with area: {area:.0f} pixels ({coverage*100:.1f}%)")
                        print(f"Max probability: {class_prob[class_mask].max():.3f}")
                        
                        found_defects.append({
                            'type': class_name,
                            'area': int(area),
                            'coverage': coverage * 100
                        })
                        
                        # Add to visualization mask
                        color = np.array(self.colors[class_name], dtype=np.uint8)
                        mask[class_mask > 0] = color
                        
                        # Scale mask to original image size for visualization
                        class_mask_resized = cv2.resize(class_mask.astype(np.uint8), (w, h), 
                                                      interpolation=cv2.INTER_NEAREST)
                        
                        # Draw contours on original size image
                        contours, _ = cv2.findContours(class_mask_resized,
                                                     cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Draw filled contours with alpha blending
                        overlay = vis_image.copy()
                        cv2.drawContours(overlay, contours, -1, tuple(map(int, color)), -1)
                        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
                        
                        # Draw contour edges
                        cv2.drawContours(vis_image, contours, -1, tuple(map(int, color)), 2)
                        
                        # Add labels at contour centers
                        for contour in contours:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                # Draw text with black outline for better visibility
                                font_size = 0.7
                                thickness = 2
                                # Draw black outline
                                cv2.putText(vis_image, class_name, (cX-30, cY), 
                                          cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness + 2)
                                # Draw colored text
                                cv2.putText(vis_image, class_name, (cX-30, cY), 
                                          cv2.FONT_HERSHEY_SIMPLEX, font_size, tuple(map(int, color)), thickness)
                                      
        # Return both the defects and visualization
        return found_defects, vis_image