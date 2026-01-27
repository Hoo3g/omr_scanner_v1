from flask import Flask, request, jsonify, render_template
import os
import uuid
import base64
from omr_pipeline_v1 import OMRPipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Pipeline
pipeline = OMRPipeline(
    region_model_path="models/region_model.pt",
    bubble_model_path="models/bubble_model.pt"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        results = pipeline.process_image(filepath, visualize=True, collect_data=True)
        # Clean up
        os.remove(filepath)
        return jsonify(results)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Listen on all interfaces so phone can connect
    app.run(host='0.0.0.0', port=5000, debug=True)
