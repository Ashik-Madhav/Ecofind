import os
import streamlit as st
import gdown
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import json

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1MVPWJK71yKIdM9xZDTMtp_Oo9pYQfSL5"  # replace with your model's URL
MODEL_PATH = "crop_classification_model.h5"
CROP_INFO_FILE = "crop_info.json"
CLASS_NAMES = ['Apple', 'Banana', 'Cotton', 'Grapes', 'Jute', 'Maize',
               'Mango', 'Millets', 'Orange', 'Paddy', 'Papaya',
               'Sugarcane', 'Tea', 'Tomato', 'Wheat']

# 1. Download model if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# 2. Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# 3. Load crop info
try:
    with open(CROP_INFO_FILE, 'r') as f:
        crop_info = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading crop_info.json: {e}")
    crop_info = {"crops": []}

# 4. Streamlit UI & homepage
st.set_page_config(page_title="Ecofind 🌾", page_icon="🌱")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 3em;'>🌾 <span style='color: green;'>Ecofind</span> – Crop Identifier</h1>
        <p style='font-size: 1.2em;'>Empowering Farmers & Agri-Researchers with Technology 🌱</p>
        <hr style='border: 1px solid #ccc;' />
        <p style='font-size: 1.1em;'>Upload an image of a crop's <b>leaf</b>, <b>fruit</b>, or <b>field</b> to identify the crop using AI.</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("🌐 About Ecofind")
st.sidebar.info("""
Ecofind is a lightweight AI-powered crop identification tool built for students, farmers, and researchers.

👨‍🌾 Trained on all major crops  
📸 Accepts leaf, fruit, or field images  
🔍 Returns crop info instantly  
📱 Deployable on web & mobile
""")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by Ashik Madhav.")

# 5. Upload & prediction section
uploaded_file = st.file_uploader("📸 Upload Crop Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess for model
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_crop = CLASS_NAMES[predicted_index]

    st.success(f"🌱 **Predicted Crop: {predicted_crop}**")

    # Show crop info if available
    found_crop = next((c for c in crop_info.get("crops", []) if c["name"].lower() == predicted_crop.lower()), None)

    if found_crop:
        st.subheader("📄 Crop Information:")
        for key, value in found_crop.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
    else:
        st.warning("ℹ️ No additional information found for this crop.")

elif not model:
    st.error("❌ Model not loaded. Please check if the `.h5` file is available.")
