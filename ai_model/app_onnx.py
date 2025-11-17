"""
FastAPI app using ONNX Runtime for inference.

This is a lightweight alternative to PyTorch that works well on Render
and doesn't have Python version compatibility issues.
"""
import io
import os
import json
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np


def load_labels(classes_path: str) -> List[str]:
    """Load class names from JSON file."""
    if not os.path.exists(classes_path):
        return []
    with open(classes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("classes", []))


def try_import_onnxruntime():
    """Import onnxruntime, providing helpful error if not available."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is not installed. "
            "Install it with: pip install onnxruntime"
        ) from e


MODEL_DIR = os.path.dirname(__file__)
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
CLASSES_PATH = os.path.join(MODEL_DIR, "model.classes.json")

ort = try_import_onnxruntime()

app = FastAPI(title="Paddy Disease (ONNX Runtime)")

session = None
input_name = None
output_name = None
class_names: List[str] = []


def init_model():
    """Initialize ONNX Runtime session."""
    global session, input_name, output_name, class_names
    
    if not os.path.exists(ONNX_PATH):
        raise RuntimeError(f"ONNX model not found at {ONNX_PATH}")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Load class names
    class_names = load_labels(CLASSES_PATH)
    
    print(f"Model loaded successfully with {len(class_names)} classes")


def preprocess(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for ONNX model.
    
    Expected format: [1, 3, 224, 224] with values in [0, 1]
    """
    # Resize to 224x224
    img = image.resize((224, 224)).convert("RGB")
    
    # Convert to numpy array and normalize to [0, 1]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    
    # Convert from HWC to CHW format (channels first)
    arr = np.transpose(arr, (2, 0, 1))
    
    # Add batch dimension
    arr = arr[None, ...]
    
    return arr.astype(np.float32)


def predict(image: Image.Image) -> str:
    """Run inference on image and return predicted class name."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Preprocess image
    x = preprocess(image)
    
    # Run inference
    outputs = session.run([output_name], {input_name: x})
    logits = outputs[0]  # Shape: [1, num_classes]
    
    # Get predicted class index
    idx = int(np.argmax(logits, axis=1)[0])
    
    # Return class name
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return str(idx)


@app.on_event("startup")
def _startup():
    """Initialize model on startup."""
    init_model()


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": "Paddy Disease API (ONNX Runtime)",
        "classes": class_names,
        "num_classes": len(class_names)
    }


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict disease class from uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with prediction
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    cls = predict(image)
    return {"prediction": cls}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": session is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_onnx:app", host="0.0.0.0", port=8000, log_level="info")
