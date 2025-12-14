import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Build absolute path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pneumo_model.h5")

# Load model safely
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(path):
    # Load and preprocess the image
    img = Image.open(path).convert("L").resize((150, 150))
    img = np.array(img) / 255.0
    img = img.reshape(1, 150, 150, 1)

    # Make prediction
    prediction = model.predict(img)[0][0]

    # If model output is 0-1 probability
    has_pneumonia = prediction > 0.5
    confidence = round(float(prediction * 100), 2)

    return has_pneumonia, confidence
