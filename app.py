import streamlit as st
import cv2
import numpy as np
from models.model import DefectDetector
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Solar Panel Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Solar Panel Defect Detection")
st.write("""
This app detects defects in solar panel electroluminescence (EL) images.
The model can identify:
- Busbars (horizontal lines)
- Cracks
- Cross defects
- Dark areas
""")

# Initialize model
@st.cache_resource
def load_model():
    return DefectDetector("models/saved_model/model.pt")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Create columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Get predictions
    defects, vis_mask = model.predict(image)
    
    with col2:
        st.subheader("Detected Defects")
        # Blend the visualization with the original image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_rgb = cv2.resize(image_rgb, (vis_mask.shape[1], vis_mask.shape[0]))  # Match sizes
        blended = cv2.addWeighted(image_rgb, 0.7, vis_mask, 0.3, 0)
        st.image(blended, use_container_width=True)
    
    # Display defect analysis
    if defects:
        st.subheader("Defect Analysis")
        df = pd.DataFrame(defects)
        df = df.round({'coverage': 2})
        st.dataframe(df)
    else:
        st.info("No defects detected in the image.")