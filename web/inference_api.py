from flask import Flask, request, send_file
import torch
from ultralytics import YOLO
from PIL import Image
import io
import os
import numpy as np

app = Flask(__name__)

# Define the path to the model
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')

# Load the YOLOv5 model
model = YOLO(model_path)

@app.route('/detect', methods=['POST'])
def detect():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    img = Image.open(file.stream)

    # Perform object detection
    results = model(img)
    annotated_img_np = results[0].plot()  # Get the annotated image as a NumPy array

    # Convert the NumPy array to a PIL image
    annotated_img = Image.fromarray(annotated_img_np)

    # Convert annotated image to a format suitable for sending over HTTP
    img_io = io.BytesIO()
    annotated_img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)