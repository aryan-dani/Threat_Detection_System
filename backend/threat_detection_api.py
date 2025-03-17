from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import cv2
import os
import base64
import time

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

print("Loading EfficientNetV2S X-ray classifier model...")
# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'efficientnetv2s_xray_classifier_best.h5')
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Define class names
CLASS_NAMES = ['gun', 'knife', 'no_threat']

@app.route('/api/detect', methods=['POST'])
def detect_threat():
    start_time = time.time()
    try:
        print("Received detection request")
        # Get image from request
        if 'image' not in request.files and 'image' not in request.json:
            print("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
            
        if 'image' in request.files:
            # Handle file upload
            print("Processing file upload")
            image_file = request.files['image']
            img_bytes = image_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Handle base64 image
            print("Processing base64 image")
            base64_image = request.json.get('image')
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            img_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if img is None or img.size == 0:
            print("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # Preprocess the image
        print("Preprocessing image")
        img = cv2.resize(img, (224, 224))  # Resize to match model's expected input
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make prediction with error handling
        print("Running prediction")
        try:
            prediction = model.predict(img, verbose=0)  # Set verbose=0 to reduce logging
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index])
            
            # Create response
            response = {
                'threat_detected': predicted_class != 'no_threat',
                'threat_type': predicted_class if predicted_class != 'no_threat' else None,
                'confidence': confidence,
                'timestamp': time.time(),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
            
            print(f"Prediction complete: {predicted_class} with {confidence:.2f} confidence")
            return jsonify(response)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Request processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)