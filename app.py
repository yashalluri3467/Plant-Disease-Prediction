import os
from datetime import datetime

import streamlit as st
import tensorflow as tf
from PIL import Image

from config import MODEL_PATH
from src.class_utils import load_class_names
from src.openrouter_integration import get_disease_advice
from src.predict import get_top_predictions
from src.preprocess import preprocess_image
from src.remedy_knowledge import get_local_remedy

st.set_page_config(page_title="Plant Disease Predictor", layout="centered")
st.title("Plant Disease Prediction and Remedy Assistant")


@st.cache_resource
def load_model(model_path: str, model_mtime: float):
    _ = model_mtime  # cache key so new model file invalidates cache
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train first with: python src/train.py")
    return tf.keras.models.load_model(model_path)


@st.cache_data
def get_class_names():
    return load_class_names()


try:
    model_mtime = os.path.getmtime(MODEL_PATH)
    model = load_model(MODEL_PATH, model_mtime)
    class_names = get_class_names()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Loaded model updated: {datetime.fromtimestamp(model_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
use_ai_advice = st.checkbox("Fetch additional AI remedy advice (requires OpenRouter API key)")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(uploaded_file)
    predictions = model.predict(processed_image, verbose=0)
    predicted_index = int(predictions.argmax())
    confidence = float(predictions.max() * 100)

    if predicted_index >= len(class_names):
        st.error("Model output classes do not match class names. Re-train the model.")
        st.stop()

    predicted_class = class_names[predicted_index]
    remedy = get_local_remedy(predicted_class)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    if confidence < 40:
        st.warning(
            "Low confidence prediction. The current model may be under-trained or this image may be out-of-distribution."
        )

    top3 = get_top_predictions(predictions, class_names, top_k=3)
    st.subheader("Top Predictions")
    st.table(
        {
            "Class": [item["class"] for item in top3],
            "Confidence (%)": [f"{item['confidence']:.2f}" for item in top3],
        }
    )

    st.subheader("Remedy")
    st.write(remedy["summary"])

    st.markdown("**Immediate Actions**")
    for action in remedy["actions"]:
        st.write(f"- {action}")

    st.markdown("**Prevention**")
    for item in remedy["prevention"]:
        st.write(f"- {item}")

    st.caption(remedy["safety_note"])

    if use_ai_advice:
        with st.spinner("Fetching AI advice..."):
            ai_advice = get_disease_advice(
                disease_name=remedy["disease"],
                crop_name=remedy["crop"],
            )
        st.subheader("AI Guidance")
        st.write(ai_advice)
