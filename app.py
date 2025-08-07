import os
import json
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import gdown
from PIL import Image

# Constants
MODEL_URL = "https://drive.google.com/uc?export=download&id=1XwhGkn_C_AkA8W_pAuPBjpVJQp8ZjRKT"
MODEL_PATH = "crop_classifier_model.h5"
CROP_INFO_FILE = "crop_info.json"
CLASS_NAMES = ['Apple', 'Banana', 'Cotton', 'Grapes', 'Jute', 'Maize',
               'Mango', 'Millets', 'Orange', 'Paddy', 'Papaya',
               'Sugarcane', 'Tea', 'Tomato', 'Wheat']

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# Load crop info
try:
    with open(CROP_INFO_FILE, 'r') as f:
        crop_info = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading crop_info.json: {e}")
    crop_info = {"crops": []}

# Streamlit UI
st.title("🌾 Ecofind - Crop Identifier")
st.write("Upload an image of a crop leaf, fruit, or field to identify the crop.")

uploaded_file = st.file_uploader("📸 Upload Crop Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_crop = CLASS_NAMES[predicted_index]

    st.success(f"🌱 Predicted Crop: **{predicted_crop}**")

    # Find crop info by name
    found_crop = next((c for c in crop_info["crops"] if c["name"].lower() == predicted_crop.lower()), None)

    if found_crop:
        st.subheader("📄 Crop Information:")
        for key, value in found_crop.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
    else:
        st.warning("ℹ️ No additional information found for this crop.")
elif not model:
    st.error("❌ Model not loaded. Please check if the `.h5` file is available.")
