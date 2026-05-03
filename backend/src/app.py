import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from base64 import b64decode

app = Flask(__name__)
CORS(app)

# Load TFLite model using the standard TF Lite Interpreter
# This is much faster than loading a full .h5 model
MODEL_PATH = os.path.join("backend", "artifact", "model", "cancer_screen_model.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = ['Type 1', 'Type 2', 'Type 3']

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

# /home page
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to CervixScan API",
        "endpoints": {
            "/health": "GET - Health check",
            "/classify": "POST - Classify cervix type from image"
        }
    }), 200

# /health page
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "live"}), 200

# classify image page
app.route('/classify', methods=['POST'])
def classify_img():
    try:
        data = request.json
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        input_data = preprocess(image_data)

        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Extract the results
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        prediction_idx = np.argmax(output_data)
        
        return jsonify({
            'class': labels[prediction_idx],
            'confidence': float(output_data[prediction_idx]),
            'all_probabilities': output_data.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)