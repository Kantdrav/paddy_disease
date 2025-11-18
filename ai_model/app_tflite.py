import io
import os
import json
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np


def load_labels(classes_path: str) -> List[str]:
    if not os.path.exists(classes_path):
        return []
    with open(classes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("classes", []))


def try_import_tflite():
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
        return tflite
    except Exception:
        # fallback to full TF if available
        try:
            import tensorflow as tf  # type: ignore
            class _Lite:
                Interpreter = tf.lite.Interpreter
            return _Lite
        except Exception as e:
            raise RuntimeError(
                "Neither tflite-runtime nor tensorflow is installed. "
                "Install tflite-runtime for deployment: pip install tflite-runtime==2.14.0"
            ) from e


MODEL_DIR = os.path.dirname(__file__)
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")
CLASSES_PATH = os.path.join(MODEL_DIR, "model.classes.json")

tflite = try_import_tflite()

app = FastAPI(title="Paddy Disease (TFLite)")

interpreter = None
input_details = None
output_details = None
class_names: List[str] = []
model_loaded: bool = False


def init_model():
    """Attempt to load model at startup. If missing, don't crash the app.
    We'll serve health info and return 503 on /predict until the model is present.
    """
    global interpreter, input_details, output_details, class_names, model_loaded
    if not os.path.exists(TFLITE_PATH):
        # Defer model loading; allow service to boot
        interpreter = None
        input_details = None
        output_details = None
        class_names = load_labels(CLASSES_PATH)
        model_loaded = False
        print(f"[warn] TFLite model not found at {TFLITE_PATH}. Service will start; /predict returns 503.")
        return

    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    class_names = load_labels(CLASSES_PATH)
    model_loaded = True
    print(f"[info] Loaded TFLite model: {TFLITE_PATH}")


def preprocess(image: Image.Image) -> np.ndarray:
    # Determine expected input shape
    shape = input_details[0]["shape"]  # e.g., [1,224,224,3] or [1,3,224,224]
    h = int(shape[-3]) if shape[-1] == 3 else int(shape[-1])
    w = int(shape[-2]) if shape[-1] == 3 else int(shape[-1])
    img = image.resize((w, h)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if shape[-1] == 3:  # NHWC
        arr = arr[None, ...]
    else:  # NCHW -> transpose
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr.astype(np.float32)


def predict(image: Image.Image) -> str:
    if interpreter is None or input_details is None or output_details is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server. Upload/commit model.tflite and redeploy.")
    x = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])  # [1, C]
    idx = int(np.argmax(y, axis=1)[0])
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return str(idx)


@app.on_event("startup")
def _startup():
    init_model()


@app.get("/")
def home():
    return {
        "message": "Paddy Disease API (TFLite)",
        "classes": class_names,
        "model_loaded": model_loaded,
        "model_path": TFLITE_PATH,
    }


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Debug log for Render
        try:
            print("Received file:", getattr(file, "filename", None), "size:", len(contents))
        except Exception:
            pass

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        cls = predict(image)
        return {"prediction": cls}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Prediction error:", e)
        print(tb)
        return {"error": str(e), "traceback": tb}
