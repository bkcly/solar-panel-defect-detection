# Create a script called download_weights.py
import requests
import os

def download_weights():
    # Create models directory if it doesn't exist
    os.makedirs('models/saved_model', exist_ok=True)
    
    # Download weights
    url = 'https://datahub.duramat.org/dataset/a1417b84-3724-47bc-90b8-b34660e462bb/resource/45da3b55-fa96-471d-a231-07b98ec5dd8e/download/crack_segmentation.zip'
    response = requests.get(url)
    
    # Save zip file
    with open('models/saved_model/model_weights.zip', 'wb') as f:
        f.write(response.content)
    
    # Unzip
    import zipfile
    with zipfile.ZipFile('models/saved_model/model_weights.zip', 'r') as zip_ref:
        zip_ref.extractall('models/saved_model')
    
    # Move the model file to the correct location
    os.rename(
        'models/saved_model/unet_oversample_low_final_model_for_paper/model.pt',
        'models/saved_model/model.pt'
    )

if __name__ == '__main__':
    download_weights()