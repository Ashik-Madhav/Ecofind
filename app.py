import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import json

# ---------------------------
# Constants
# ---------------------------
MODEL_PATH = "crop_classifier_mobilenet.h5"   # keep the file in same folder as app.py
CROP_INFO_FILE = "crop_info.json"
CLASS_NAMES = ['Apple', 'Banana', 'Cotton', 'Grapes', 'Jute', 'Maize',
               'Mango', 'Millets', 'Orange', 'Paddy', 'Papaya',
               'Sugarcane', 'Tea', 'Tomato', 'Wheat']

# ---------------------------
# Load model
# ---------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# ---------------------------
# Load crop info
# ---------------------------
try:
    with open(CROP_INFO_FILE, 'r') as f:
        crop_info = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading crop_info.json: {e}")
    crop_info = {"crops": []}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Ecofind 🌾", page_icon="🌱")
st.title("🌾 Ecofind – Crop Identifier")
st.write("Upload an image of a crop's leaf, fruit, or field to identify it using AI.")

uploaded_file = st.file_uploader("📸 Upload Crop Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_crop = CLASS_NAMES[predicted_index]

    st.success(f"🌱 **Predicted Crop: {predicted_crop}**")

    # Lookup crop info in JSON
    found_crop = next((c for c in crop_info.get("crops", []) 
                       if c["name"].lower() == predicted_crop.lower()), None)

    if found_crop:
        st.subheader("📄 Crop Information:")
        for key, value in found_crop.items():
            if key != "name":
                st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
    else:
        st.warning("ℹ️ No additional information found for this crop.")

elif not model:
    st.error("❌ Model not loaded. Please check if the `.h5` file is available.")
