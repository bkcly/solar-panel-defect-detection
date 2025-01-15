import streamlit as st
import cv2
import numpy as np
from models.model import DefectDetector
import pandas as pd
import os
import requests

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
    """Download and load the model from Hugging Face if it doesn't exist locally"""
    os.makedirs("models/saved_model", exist_ok=True)
    model_path = "models/saved_model/model.pt"
    
    if not os.path.exists(model_path):
        # Get token from Streamlit secrets
        token = st.secrets["HUGGINGFACE_TOKEN"]
        
        # Your Hugging Face model repository URL
        repo_id = "malianboy/solar-panel-defect-detection"
        filename = "model.pt"
        
        # Construct the URL for the model file
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        
        # Set up headers with token
        headers = {"Authorization": f"Bearer {token}"}
        
        # Download with progress bar
        with st.spinner("Downloading model weights... This may take a few minutes."):
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Download and save the file
            with open(model_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Update progress bar
                        if total_size:
                            progress = int((downloaded_size / total_size) * 100)
                            progress_bar.progress(progress)
            
            st.success("Model weights downloaded successfully!")
    
    return DefectDetector(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, width=400)
        
        # Get predictions
        defects, vis_mask = model.predict(image)
        
        with col2:
            st.subheader("Detected Defects")
            # Blend the visualization with the original image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_rgb = cv2.resize(image_rgb, (vis_mask.shape[1], vis_mask.shape[0]))  # Match sizes
            blended = cv2.addWeighted(image_rgb, 0.7, vis_mask, 0.3, 0)
            st.image(blended, width=400)
        
        # Display defect analysis
        if defects:
            st.subheader("Defect Analysis")
            # Convert the defects list to DataFrame and add descriptions
            df = pd.DataFrame(defects)
            df = df.round({'coverage': 2})
            # Rename columns and add units
            df.columns = ['Defect Type', 'Area (pixels)', 'Coverage (%)']
            
            # Create color-coded defect type descriptions
            st.markdown("""
            **Analysis Details:**
            - Area: Number of pixels affected by the defect
            - Coverage: Percentage of total image area affected
            
            **Defect Types and Colors:**
            - 🟫 Busbar: Horizontal conductive lines (brown)
            - 🟪 Crack: Cell fractures or breaks (pink)
            - 🟦 Cross: Cross-shaped defects (blue)
            - 🟩 Dark: Areas with reduced luminescence (green)
            """)
            
            # Display the dataframe
            st.dataframe(
                df,
                column_config={
                    "Defect Type": st.column_config.Column(
                        width="medium",
                    ),
                    "Area (pixels)": st.column_config.NumberColumn(
                        width="medium",
                        format="%d",
                    ),
                    "Coverage (%)": st.column_config.NumberColumn(
                        width="medium",
                        format="%.2f",
                    ),
                },
                hide_index=True,
            )
        else:
            st.info("No defects detected in the image.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again with a different image or contact support if the issue persists.")