requirements.txt — dependencies
train.py — trains models, evaluates, picks thresholds, saves artifacts
serve.py — FastAPI app that serves the trained model



Run:

Install deps: pip install -r requirements.txt
Train and evaluate: /usr/bin/python3 train.py
Serve API: uvicorn serve:app --host 0.0.0.0 --port 8080
Test prediction:
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.25, -0.87, 1.02, 0.33, -0.44, 0.56, 0.78, -0.12, 0.99, -0.65,
                    0.11, 0.22, -0.33, 1, 0.55, -0.66, 0.77, 0.88, -0.99, 0.12,
                    0.23, -0.34, 0.45, 0.56, -0.67, 0.78, 0.89, -0.90, 0.91, 0.12]}'
