import io
import os
import sys
import time
from fastapi.concurrency import run_in_threadpool
# We'll treat fastapi, PIL (Pillow) and torch as optional at module import so
# running `python3 app.py` with system Python that doesn't have the project's
# venv packages doesn't immediately raise ModuleNotFoundError. When imported
# in the venv (or when uvicorn runs with the venv Python) these packages will
# be present and the API app will be created.

# optional dependencies
FASTAPI_AVAILABLE = True
"""Guarded FastAPI app for paddy disease classification.

This file defers importing optional dependencies (fastapi, Pillow, torch)
so that running `python3 app.py` with a system Python that lacks the project's
venv packages will print helpful instructions instead of crashing.
"""

import io
import os
import sys
import importlib.util


# Check optional dependencies
_fastapi_spec = importlib.util.find_spec("fastapi")
_pil_spec = importlib.util.find_spec("PIL")
_torch_spec = importlib.util.find_spec("torch")

FASTAPI_AVAILABLE = _fastapi_spec is not None
PIL_AVAILABLE = _pil_spec is not None
TORCH_AVAILABLE = _torch_spec is not None

_fastapi_import_error = None
_pil_import_error = None
_torch_import_error = None

if FASTAPI_AVAILABLE:
    try:
        from fastapi import FastAPI, File, UploadFile, HTTPException
    except Exception as e:
        FASTAPI_AVAILABLE = False
        _fastapi_import_error = e

if PIL_AVAILABLE:
    try:
        from PIL import Image
    except Exception as e:
        PIL_AVAILABLE = False
        _pil_import_error = e

if TORCH_AVAILABLE:
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        from torchvision import models
    except Exception as e:
        TORCH_AVAILABLE = False
        _torch_import_error = e


if not FASTAPI_AVAILABLE or not PIL_AVAILABLE:
    print("This project requires FastAPI and Pillow (PIL) to run the API.")
    print()
    if not FASTAPI_AVAILABLE:
        print("- fastapi is missing:", _fastapi_import_error or "not installed")
    if not PIL_AVAILABLE:
        print("- pillow (PIL) is missing:", _pil_import_error or "not installed")
    if not TORCH_AVAILABLE:
        print("- torch is missing:", _torch_import_error or "not installed")
    print()
    print("Quick start (from project root):")
    print("  source venv/bin/activate")
    print("  pip install -r requirements.txt")
    print("  # then run with the venv python: ./venv/bin/python app.py")
    sys.exit(1)


# Now import FastAPI and PIL (we verified availability above)
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image


app = FastAPI()

# runtime placeholders
transform = None
model = None
class_names = []
_model_load_error = None

if TORCH_AVAILABLE:
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            class_names = checkpoint.get("classes", [])

            model = models.resnet18(weights=None)

            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, max(1, len(class_names)))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
        else:
            _model_load_error = FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            model = None
    except Exception as e:
        _model_load_error = e
        model = None


def _service_unavailable(msg: str):
    raise HTTPException(status_code=503, detail=msg)


def predict_image(img: Image.Image):
    if not TORCH_AVAILABLE:
        _service_unavailable(
            "PyTorch is not installed in this Python environment. Activate the project venv or run with the venv python."
        )
    if model is None:
        _service_unavailable(f"Model not loaded: {_model_load_error}")
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(t)
        _, predicted = torch.max(outputs, 1)
    idx = int(predicted.item())
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return str(idx)


def _collect_metrics():
    if not TORCH_AVAILABLE or model is None:
        return {
            "torch_available": TORCH_AVAILABLE,
            "model_loaded": model is not None,
            "error": str(_model_load_error),
        }
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # dummy inference timing
    dummy = torch.randn(1, 3, 224, 224)
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "torch_available": True,
        "model_loaded": True,
        "classes": class_names,
        "num_classes": len(class_names),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "dummy_inference_ms": round(elapsed_ms, 3),
    }


@app.get("/")
def home():
    if not TORCH_AVAILABLE:
        return {"message": "Paddy Disease Classification API (torch missing)", "error": str(_torch_import_error)}
    if model is None:
        return {"message": "Paddy Disease Classification API (model not loaded)", "error": str(_model_load_error)}
    return {"message": "Paddy Disease Classification API is running"}




@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Debug log
        try:
            print("Received file:", getattr(file, "filename", None), "size:", len(contents))
        except Exception:
            pass

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run heavy inference in background thread pool
        class_name = await run_in_threadpool(predict_image, image)

        return {"prediction": class_name}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()

        print("Prediction error:", e)
        print(tb)

        return {"error": str(e), "traceback": tb}



@app.get("/metrics")
def metrics():
    return _collect_metrics()


if __name__ == "__main__":
    # If torch is missing, print guidance and exit instead of crashing.
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed in this Python interpreter:")
        print(_torch_import_error)
        print()
        print("Run with the project's virtualenv activated:")
        print("  source venv/bin/activate")
        print("  python app.py")
        sys.exit(1)

    try:
        import uvicorn

        uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        print("Failed to start uvicorn:", e)
        sys.exit(1)





