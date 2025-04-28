import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# --- Load Models ---
@st.cache_resource

def load_rnn_model():
    return tf.keras.models.load_model('RNN_model.h5')

@st.cache_resource

def load_cnn_model():
    return YOLO('CNN_model.pt')

@st.cache_resource

def load_feature_extractor():
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        Dense(128, activation='relu')  # Reducing feature size to 128
    ])
    return model

# --- Prediction Functions ---
def extract_features(image, feature_extractor):
    img_resized = cv2.resize(image, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    features = np.expand_dims(features, axis=1)  # (batch, 1, features)
    return features

def predict_image_rnn(image, model, feature_extractor):
    features = extract_features(image, feature_extractor)
    preds = model.predict(features)
    class_idx = np.argmax(preds)
    return 'Fire' if class_idx == 0 else 'Smoke'

def predict_image_cnn(image, model):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
    cv2.imwrite(temp_path, image)
    results = model(temp_path)
    os.remove(temp_path)
    result_frame = results[0].plot()
    return result_frame

# --- Streamlit UI ---
st.title("🔥 D-Fire Detection App")

# Sidebar
model_choice = st.sidebar.selectbox("Choose a Model", ("RNN_model.h5", "CNN_model.pt"))
input_mode = st.selectbox("Select Input Mode", ("Upload Image", "Use Webcam"))

# Load models based on selection
if model_choice == "RNN_model.h5":
    model = load_rnn_model()
    feature_extractor = load_feature_extractor()
else:
    model = load_cnn_model()

# Image Processing
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    process_image = st.button("Process Image")

    if uploaded_file is not None and process_image:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if model_choice == "RNN_model.h5":
            prediction = predict_image_rnn(image, model, feature_extractor)
            color = (0, 0, 255) if prediction == 'Fire' else (0, 255, 0)
            cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Prediction: {prediction}")
        else:
            result_img = predict_image_cnn(image, model)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="YOLOv8 Detection")

# Webcam Processing
elif input_mode == "Use Webcam":
    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")

    if start_webcam and not stop_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            frame = cv2.flip(frame, 1)

            if model_choice == "RNN_model.h5":
                prediction = predict_image_rnn(frame, model, feature_extractor)
                color = (0, 0, 255) if prediction == 'Fire' else (0, 255, 0)
                cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                frame = predict_image_cnn(frame, model)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if stop_webcam:
                break

        cap.release()

st.sidebar.markdown("---")
st.sidebar.write("Made by MuhamMad Hamza ")
