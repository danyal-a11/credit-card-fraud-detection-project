# Flask Web Application

Real-time fraud detection web app serving the XGBoost deployment model.

## Run

From the project root:

```bash
conda activate fraud_detection
python app/app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Files

- `app.py` — Flask backend with prediction, health, and model-info endpoints
- `templates/index.html` — Single-page frontend (HTML, CSS, JavaScript)
- `static/css/` — Static assets

See the main project [README](../README.md) for full details.