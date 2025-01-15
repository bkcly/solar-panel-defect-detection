from .visualization import draw_defect_boxes, DEFECT_COLORS
from .preprocessing import preprocess_image

# Constants
IMAGE_SIZE = (512, 640)
NUM_CLASSES = 5  # background + 4 defect types

# Defect types mapping
DEFECT_TYPES = {
    0: 'background',
    1: 'busbar',
    2: 'crack',
    3: 'cross',
    4: 'dark'
}