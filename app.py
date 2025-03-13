from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
from keras.models import load_model

app = Flask(__name__)

# Load the trained model and labels
model = load_model("model/sign_model.h5")
with open("labels.json", "r") as f:
    labels = json.load(f)

@app.route("/")
def home():
    return "Sign Language Translator API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from request
        file = request.files["image"]
        img = tf.keras.preprocessing.image.load_img(file, target_size=(64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_label = labels[str(np.argmax(prediction))]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
