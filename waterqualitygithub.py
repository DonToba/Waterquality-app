import streamlit as st
import rasterio
import numpy as np
import joblib
import os
import requests
from tempfile import NamedTemporaryFile
import pickle

# Expected file names sequence
expected_sequence = [
    "SR_B1.tif", "SR_B2.tif", "SR_B3.tif", "SR_B4.tif", "SR_B5.tif", 
    "SR_B6.tif", "SR_B7.tif", "B2_B3.tif", "B2_B4.tif", "B3_B4.tif",
    "B4_B5.tif", "B1_B7.tif", "B2_B7.tif", "B3_B7.tif", "B4_B7.tif", 
    "B5_B7.tif", "B6_B7.tif", "B2_B5.tif", "B2_B6.tif", "B3_B5.tif", 
    "B3_B6.tif"
]

def load_model():
    # URL to the pre-trained model file on GitHub
    model_url = 'https://raw.githubusercontent.com/DonToba/Waterquality-app/main/random_forest_model.pkl'
    
    # Download the model file
    response = requests.get(model_url)
    if response.status_code == 200:
        with NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load the model using pickle
        with open(tmp_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error("Failed to download the model from GitHub.")
        return None

def preprocess_images(files):
    bands = []
    tmp_files = []
    try:
        for file in files:
            # Save the uploaded file temporarily to disk
            with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
                tmp_files.append(tmp_path)
                with rasterio.open(tmp_path) as src:
                    bands.append(src.read(1))
        stacked_array = np.stack(bands)
        cleaned_array = np.nan_to_num(stacked_array)
        reshaped_array = cleaned_array.reshape(cleaned_array.shape[0], -1).T
        return reshaped_array, src
    except rasterio.errors.RasterioIOError as e:
        st.error(f"Error reading file: {e}")
        return None, None
    finally:
        # Clean up the temporary files
        for tmp_file in tmp_files:
            os.remove(tmp_file)

def predict_water_quality(model, data, original_shape, crs, transform):
    predicted_labels = model.predict(data)
    predicted_image = predicted_labels.reshape(original_shape)
    return predicted_image

def save_tiff(output_path, predicted_image, crs, transform):
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=predicted_image.shape[0],
        width=predicted_image.shape[1],
        count=1,
        dtype=predicted_image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(predicted_image, 1)

# Streamlit Interface
st.title("Water Quality Prediction App")

uploaded_files = st.file_uploader("Upload 21 Raster Images", accept_multiple_files=True, type=['tif'])

# Ensure files are uploaded in the expected sequence
if uploaded_files:
    # Reorder files to match the expected sequence
    uploaded_files_dict = {file.name: file for file in uploaded_files}
    reordered_files = [uploaded_files_dict.get(name) for name in expected_sequence]
    
    # Check if any files are missing
    if None in reordered_files:
        st.error(f"Uploaded files do not match the expected sequence. Missing files: {', '.join([name for name, file in zip(expected_sequence, reordered_files) if file is None])}")
        st.stop()

if st.button("Predict Water Quality") and len(uploaded_files) == 21:
    with st.spinner("Processing..."):
        model = load_model()
        if model is not None:
            data, src = preprocess_images(reordered_files)
            predicted_image = predict_water_quality(model, data, src.shape, src.crs, src.transform)
            output_path = "predicted_water_quality.tif"
            save_tiff(output_path, predicted_image, src.crs, src.transform)
            st.success("Prediction completed and saved as 'predicted_water_quality.tif'.")
            st.download_button("Download Prediction", data=open(output_path, "rb").read(), file_name=output_path)
