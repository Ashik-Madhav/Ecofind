import streamlit as st
import numpy as np
import cv2
import json
import os
from PIL import Image
import gdown
from tensorflow.keras.models import load_model

# -------------------- CONFIGURATION --------------------
# Set page title and icon
st.set_page_config(page_title="Ecofind 🌾", page_icon="🌱")

# -------------------- MODEL LOADING --------------------
MODEL_URL = "https://drive.google.com/uc?id=1XwhGkn_C_AkA8W_pAuPBjpVJQp8ZjRKT"
MODEL_PATH = "crop_classifier_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    st.error(f"❌ Error loading model: {e}")

# -------------------- LOAD CROP INFO --------------------
with open("crop_info.json", "r") as f:
    crop_info = json.load(f)

# 15 classes (must match model training)
CLASS_NAMES = [
    "Apple", "Banana", "Cotton", "Grapes", "Jute",
    "Maize", "Mango", "Millets", "Orange", "Paddy",
    "Papaya", "Sugarcane", "Tea", "Tomato", "Wheat"
]

# -------------------- STREAMLIT UI --------------------
# Header and instructions
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 3em;'>🌾 <span style='color: green;'>Ecofind</span> – Crop Identifier</h1>
        <p style='font-size: 1.2em;'>Empowering Farmers & Agri-Researchers with AI 🌱</p>
        <hr style='border: 1px solid #ccc;' />
        <p style='font-size: 1.1em;'>Upload an image of a crop's <b>leaf</b>, <b>fruit</b>, or <b>field</b> to identify the crop using AI.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🌐 About Ecofind")
st.sidebar.info("""
Ecofind is a lightweight AI-powered crop identification tool built for students, farmers, and researchers.

👨‍🌾 Trained on 15 major crops  
📸 Accepts leaf, fruit, or field images  
🔍 Returns crop info instantly  
📱 Deployable on web & mobile
""")
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by the Ecofind team.")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("📸 Upload Crop Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # Load and display image
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

    # Display crop info
    found_crop = next((c for c in crop_info["crops"] if c["name"].lower() == predicted_crop.lower()), None)

    if found_crop:
        st.subheader("📄 Crop Information:")
        for key, value in found_crop.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
    else:
        st.warning("ℹ️ No additional information found for this crop.")

elif not model:
    st.error("❌ Model not loaded. Please check if the `.h5` file is available.")
