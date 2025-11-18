# Paddy Disease Classification API

A FastAPI-based service for classifying paddy (rice) disease from images.

## 🚀 Quick Start

### For Production/Deployment (Recommended)

Use **ONNX Runtime** for lightweight deployment:

```bash
# Install dependencies
pip install -r requirements.onnx.txt

# Run the server
python3 -m uvicorn app_onnx:app --host 0.0.0.0 --port 8000
```

Or use the startup script:

```bash
./start.sh
```

### For Local Development

If you want to use PyTorch for development:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Or use the helper script:

```bash
./run.sh
```

## 📋 API Endpoints

- **GET /** - Health check and model info
- **POST /predict/** - Upload an image and get disease prediction
  - Accepts: multipart form data with `file` field
  - Returns: `{"prediction": "Disease Name"}`
- **GET /health** - Service health status (ONNX Runtime only)

## 🌐 Deployment

**For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**

The deployment guide covers:
- Deploying to Render (recommended)
- Choosing between ONNX Runtime, TFLite, and PyTorch
- Troubleshooting common deployment issues
- Environment configuration

## 📦 Available Deployment Options

1. **ONNX Runtime** (`app_onnx.py`) - ⭐ Recommended for production
   - Lightweight (~150 MB)
   - Fast inference
   - Works with Python 3.12+
   
2. **TFLite Runtime** (`app_tflite.py`) - Alternative lightweight option
   - Very lightweight (~80 MB)
   - Requires model conversion
   
3. **PyTorch** (`app.py`) - Development only
   - Full PyTorch ecosystem
   - Heavy (~1.5 GB+)
   - Not recommended for deployment

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

This means you haven't activated the virtual environment. Either:
- Activate the venv: `source venv/bin/activate`
- Use the venv python: `venv/bin/python app.py`
- Or switch to ONNX Runtime: `python3 -m uvicorn app_onnx:app`

### "torch==2.9.1+cpu not found" on Render

Use ONNX Runtime instead of PyTorch. See [DEPLOYMENT.md](DEPLOYMENT.md) for details.

## 📄 License

See repository root for license information.
