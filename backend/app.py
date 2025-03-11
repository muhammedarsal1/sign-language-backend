from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Allow Netlify requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

app = Flask(__name__)
CORS(app)  # ✅ Enable Cross-Origin Requests

@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    image = cv2.resize(image, (28, 28)).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(image)
    predicted_label = int(np.argmax(prediction))

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
