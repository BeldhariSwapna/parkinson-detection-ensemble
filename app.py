import streamlit as st
import sys
import os
from PIL import Image

# Fix import path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import image_model
import voice_model
import fusion_model

st.title("🧠 Parkinson Detection System")

option = st.selectbox("Choose Mode", ["Image", "Voice", "Fusion"])

# IMAGE
if option == "Image":
    file = st.file_uploader("Upload Handwriting Image")

    if file:
        image = Image.open(file)
        st.image(image)

        pred = image_model.predict_image(image)
        st.success(f"Prediction: {pred}")

# VOICE
elif option == "Voice":
    file = st.file_uploader("Upload Voice CSV")

    if file:
        pred = voice_model.predict_voice(file)
        st.success(f"Prediction: {pred}")

# FUSION
elif option == "Fusion":
    img = st.file_uploader("Upload Image")
    voice = st.file_uploader("Upload Voice CSV")

    if img and voice:
        image = Image.open(img)

        img_pred = image_model.predict_image(image)
        voice_pred = voice_model.predict_voice(voice)

        final = fusion_model.final_prediction(img_pred, voice_pred)

        st.success(f"Final Result: {final}")