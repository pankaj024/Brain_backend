from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from werkzeug.utils import secure_filename

# Import models
from models.brain_tumor import predict_image as predict_brain_tumor
from models.bitcoin import get_prediction_data as predict_bitcoin

app = Flask(__name__)
# Allow CORS for Vercel frontend
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "ML API is running. Access the endpoints at /api/", "endpoints": ["/api/health", "/api/predict/brain_tumor", "/api/predict/bitcoin"]})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "ML API is running"})

@app.route('/api/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the ML Predictive Analytics API. Use /api/predict/brain_tumor or /api/predict/bitcoin endpoints."})

@app.route('/api/predict/brain_tumor', methods=['POST'])
def brain_tumor_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = predict_brain_tumor(filepath)
            # Clean up
            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/predict/bitcoin', methods=['GET']) # Changed to GET as we don't need input for latest prediction
def bitcoin_endpoint():
    try:
        result = predict_bitcoin()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
