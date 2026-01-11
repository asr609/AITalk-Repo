# serve.py
import json
import pickle
import numpy as np
from fastapi import FastAPI

ARTIFACTS_DIR = "artifacts"

with open(f"{ARTIFACTS_DIR}/model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(f"{ARTIFACTS_DIR}/config.json", "r") as f:
    CFG = json.load(f)

THRESHOLD = CFG["threshold"]

app = FastAPI(title="FraudDemo API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    """
    Payload format:
    {
      "features": [f1, f2, ..., f30]  # 30 features expected (match training)
    }
    """
    x = np.array(payload["features"], dtype=np.float64).reshape(1, -1)
    # Pipeline handles scaling internally
    prob = float(MODEL.predict_proba(x)[:, 1][0])
    decision = int(prob >= THRESHOLD)
    return {
        "probability": prob,
        "decision": decision,
        "threshold": THRESHOLD
    }
