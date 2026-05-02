import numpy as np
import cv2
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from base64 import b64decode
import tensorflow as tf

app = Flask(__name__)
CORS(app) # Allows the frontend to communicate with the backend

# Load TFLite model and allocate tensors
MODEL_PATH = "../artifact/model/cancer_screen_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ['Type 1', 'Type 2', 'Type 3']

# function to preprocess input image
def preprocess_image(base64_img):
    # Decode base64
    data = base64_img.split(',')[1] if ',' in base64_img else base64_img
    nparr = np.frombuffer(b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize and normalize
    img = cv2.resize(img, (180, 180))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# /health page
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "live"}), 200

# classify image page
@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        if 'image_data' not in data:
            return jsonify({"error": "No image data provided"}), 400

        input_data = preprocess_image(data['image_data'])

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get results
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        prediction_idx = np.argmax(output_data)
        
        return jsonify({
            'class': LABELS[prediction_idx],
            'probability': output_data.tolist(),
            'confidence': float(output_data[prediction_idx])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)