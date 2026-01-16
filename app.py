from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your model
MODEL_PATH = "tomato_model.h5"
model = load_model(MODEL_PATH)

# Example route to test
@app.route('/')
def home():
    return "Tomato Disease Detection API is running!"

# Example inference route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    img_file = request.files['file']
    img_path = os.path.join("temp.jpg")
    img_file.save(img_path)
    
    # Preprocess image
    img = image.load_img(img_path, target_size=(128,128))  # adjust size to your model
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    predicted_class = str(np.argmax(prediction, axis=1)[0])
    
    os.remove(img_path)
    return jsonify({'predicted_class': predicted_class})

if __name__ == "__main__":
    # IMPORTANT: host=0.0.0.0 so Docker exposes it
    app.run(host="0.0.0.0", port=5000, debug=True)
