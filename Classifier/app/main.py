# main.py
from typing import Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

app = FastAPI()

# Class names
class_names = [
    "Fruit healthy",
    "Fruit with Rugose",
    "Leaf healthy",
    "Leaf with Rugose",
    "Stem healthy",
    "Stem with Rugose"
]

# Paths
THIS_DIR = Path(__file__).resolve().parent 
onnx_path = THIS_DIR / "models" / "best_model.onnx"

# Load ONNX model
model = onnx.load(onnx_path)
session = ort.InferenceSession(
    onnx_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# Helper: transform image
def preprocess_image(img: Image.Image, target_size=(128, 128)):
    """
    Resize, normalize and convert PIL Image to numpy array suitable for ONNX.
    Output shape: [1, C, H, W], dtype float32
    """
    # Resize
    img = img.resize(target_size)

    # Convert to numpy array (H, W, C) and scale to [0,1]
    img_np = np.array(img).astype(np.float32) / 255.0

    # Convert H W C -> C H W
    img_np = np.transpose(img_np, (2, 0, 1))

    # Add batch dimension -> [1, C, H, W]
    img_np = np.expand_dims(img_np, axis=0)

    return img_np


# Routes
@app.get("/")
def read_root():
    return {"message": "API running correctly"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image (multipart/form-data),
    processes it with the ONNX model and returns the predicted class.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    # Read file bytes
    image_bytes = await file.read()

    # Open image with PIL
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess -> numpy
    img_np = preprocess_image(img)  # shape: [1, C, H, W], float32

    # ONNX inference
    outputs = session.run([output_name], {input_name: img_np})
    logits = outputs[0]  
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) 
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  

    # Predicted class
    pred_idx = int(np.argmax(logits, axis=1)[0])
    print(logits)
    pred_class = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
    # Map probabilities to class names
    prob_per_class = {name: float(prob) for name, prob in zip(class_names, probs[0])}
    return {
        "predicted_class_index": pred_idx,
        "predicted_class_name": pred_class,
        "probabilities": prob_per_class
    }