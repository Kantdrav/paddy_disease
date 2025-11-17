# Paddy Disease Classification API - Deployment Guide

This guide explains how to deploy the Paddy Disease Classification API on Render or other platforms.

## Deployment Options

We provide three deployment options, listed from lightest to heaviest:

### 1. **ONNX Runtime** (Recommended for Production) ⭐

**File:** `app_onnx.py`  
**Requirements:** `requirements.onnx.txt`

**Pros:**
- ✅ Lightest weight (~150 MB total)
- ✅ Fast inference
- ✅ Works with Python 3.12+
- ✅ No GPU dependencies
- ✅ Best for cloud deployment (Render, Heroku, etc.)

**Cons:**
- ❌ Requires ONNX model (already provided: `model.onnx`)

**Deploy command:**
```bash
pip install -r requirements.onnx.txt
python3 -m uvicorn app_onnx:app --host 0.0.0.0 --port 8000
```

### 2. **TFLite Runtime** (Alternative Lightweight Option)

**File:** `app_tflite.py`  
**Requirements:** `requirements.tflite.txt`

**Pros:**
- ✅ Very lightweight (~80 MB)
- ✅ Fast inference
- ✅ Works with Python 3.12+

**Cons:**
- ❌ Requires TFLite model conversion (not yet available)
- ❌ TFLite conversion can be complex

### 3. **PyTorch** (Development/Local Only)

**File:** `app.py`  
**Requirements:** `requirements.txt`

**Pros:**
- ✅ Full PyTorch ecosystem
- ✅ Easy model development and training

**Cons:**
- ❌ Very heavy (~1.5 GB+)
- ❌ Doesn't work with Python 3.13+
- ❌ Not recommended for cloud deployment
- ❌ CPU-only builds require special installation

## Render Deployment

### Quick Setup

1. **Create a new Web Service** on Render
2. **Connect your GitHub repository**
3. **Configure the service:**
   - **Build Command:** `pip install -r ai_model/requirements.onnx.txt`
   - **Start Command:** `cd ai_model && bash start.sh`
   - **Python Version:** 3.12.3 (set via `.python-version` file)

### Environment Variables

No environment variables are required for basic deployment. The service will automatically:
- Use `PORT` environment variable provided by Render
- Start on `0.0.0.0` to accept external connections
- Use the ONNX Runtime app by default

### Advanced Configuration

To use a different app module, set the `APP_MODULE` environment variable:

- ONNX Runtime (default): `APP_MODULE=app_onnx:app`
- TFLite Runtime: `APP_MODULE=app_tflite:app`
- PyTorch (not recommended): `APP_MODULE=app:app`

### Troubleshooting

#### Error: "torch==2.9.1+cpu not found"

**Solution:** Use ONNX Runtime instead. The PyTorch version is incompatible with Render's Python environment. Make sure your build command uses `requirements.onnx.txt`.

#### Error: "Python version 3.13 not supported"

**Solution:** Add a `.python-version` file with `3.12.3` to your `ai_model` directory. This file is already included.

#### Error: "Model not found"

**Solution:** Ensure these files are committed to your repository:
- `ai_model/model.onnx` - ONNX model file
- `ai_model/model.onnx.data` - ONNX model weights
- `ai_model/model.classes.json` - Class names

## API Endpoints

All three deployment options provide the same API:

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Paddy Disease API (ONNX Runtime)",
  "classes": ["Bacterial Leaf Blight", "Brown Spot"],
  "num_classes": 2
}
```

### `POST /predict/`
Predict disease class from an uploaded image.

**Request:**
- Multipart form data
- Field name: `file`
- Content: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "prediction": "Bacterial Leaf Blight"
}
```

### `GET /health`
Health check endpoint (ONNX Runtime only).

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Local Development

### Using ONNX Runtime (Recommended)

```bash
# Install dependencies
pip install -r requirements.onnx.txt

# Run the server
python3 -m uvicorn app_onnx:app --host 127.0.0.1 --port 8000 --reload
```

### Using PyTorch (Development)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Run the server
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

## Testing the API

```bash
# Test health endpoint
curl http://localhost:8000/

# Test prediction
curl -X POST "http://localhost:8000/predict/" \
  -F "file=@/path/to/your/image.jpg"
```

## Model Information

- **Architecture:** ResNet-18
- **Input Size:** 224x224 RGB
- **Output Classes:** 2 (Bacterial Leaf Blight, Brown Spot)
- **Model Format:** ONNX (optimized for inference)

## File Structure

```
ai_model/
├── app.py                    # PyTorch app (development)
├── app_onnx.py              # ONNX Runtime app (production) ⭐
├── app_tflite.py            # TFLite app (alternative)
├── requirements.txt         # PyTorch dependencies
├── requirements.onnx.txt    # ONNX Runtime dependencies ⭐
├── requirements.tflite.txt  # TFLite dependencies
├── start.sh                 # Render startup script
├── .python-version          # Python version for Render
├── model.pth                # PyTorch checkpoint
├── model.onnx               # ONNX model ⭐
├── model.onnx.data          # ONNX model weights ⭐
└── model.classes.json       # Class names ⭐
```

## License

MIT License - See repository root for details.
