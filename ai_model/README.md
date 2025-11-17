# AI Model - Run instructions

This project uses a Python virtual environment located at `venv/`.

Quick start:

1. Activate the venv:

```bash
source venv/bin/activate
```

2. Install dependencies (if not already installed):

```bash
pip install -r requirements.txt
pip install uvicorn fastapi python-multipart
```

3. Run the API:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Or use the helper script:

```bash
./run.sh
```

API endpoints:
- GET / -> health
- POST /predict/ -> multipart upload of an image field `file`, returns JSON {"prediction": <class>}

If you run `python3 app.py` with your system Python and see "ModuleNotFoundError: No module named 'torch'", that means you haven't activated the virtualenv. Either activate the venv or run the app with the venv python: `venv/bin/python app.py`.
