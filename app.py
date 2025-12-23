import os
import json
import io

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------------------------------------------
# MUST be first Streamlit command
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Multiclass Fish Image Classification",
    layout="centered"
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Your actual files
BEST_MODEL_PATH = r"D:\yenaval\fish_learn\fish_classification_inceptionv3_finetuned_model.h5"
CLASS_LABELS_PATH = r"D:\yenaval\fish_learn\class_labels.json"

# -------------------------------------------------------------------
# SAFE MODEL + LABEL LOADING
# -------------------------------------------------------------------
@st.cache_resource
def load_best_model():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {BEST_MODEL_PATH}")
    model = load_model(BEST_MODEL_PATH)
    return model

@st.cache_resource
def load_class_labels():
    if not os.path.exists(CLASS_LABELS_PATH):
        raise FileNotFoundError(f"Class labels file not found: {CLASS_LABELS_PATH}")
    with open(CLASS_LABELS_PATH, "r") as f:
        labels = json.load(f)
    # Expecting { "0": "class_name", ... } or {0: "class_name", ... }
    index_to_class = {int(k): v for k, v in labels.items()}
    num_classes = len(index_to_class)
    ordered_classes = [index_to_class[i] for i in range(num_classes)]
    return ordered_classes

# Try to load; if anything fails, show error on the page and stop.
try:
    model = load_best_model()
    class_names = load_class_labels()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# -------------------------------------------------------------------
# IMAGE PREPROCESSING + PREDICTION
# -------------------------------------------------------------------
def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image: Image.Image):
    preprocessed = preprocess_image(image)
    preds = model.predict(preprocessed)
    probs = preds[0]
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    top_conf = float(probs[top_idx])

    all_probs = list(zip(class_names, probs))
    all_probs.sort(key=lambda x: x[1], reverse=True)
    return top_class, top_conf, all_probs

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("Multiclass Fish Image Classification")
st.write(
    """
Upload a fish image and this app will **predict the species**
using your fine-tuned InceptionV3 model.
"""
)

st.sidebar.header("Project Info")
st.sidebar.markdown(
    """
**Project:** Multiclass Fish Image Classification   
"""
)

st.sidebar.subheader("How to use")
st.sidebar.markdown(
    """
1. Upload a fish image (JPG/PNG).  
2. Click **Predict**.  
3. See predicted class and confidence scores.  
"""
)

uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                top_class, top_conf, all_probs = predict_image(image)
                st.success(f"Predicted Species: **{top_class}**")
                st.write(f"Confidence: **{top_conf * 100:.2f}%**")

                st.subheader("All Class Probabilities")
                for cls, prob in all_probs:
                    st.write(f"- {cls}: {prob * 100:.2f}%")
            except Exception as e:
                st.error(f"Prediction error: {e}")
else:
    st.info("Please upload a fish image to get started.")
