import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException

app = FastAPI()

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
model = None

class PredictRequest(BaseModel):
    features: list[float]

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

N_FEATURES = 30

@app.post("/predict")
def predict(request: PredictRequest):
    if len(request.features) != N_FEATURES:
        raise HTTPException(status_code=400, detail="Number of features does not match.")

    x = np.array(request.features).reshape(1, -1)

    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0].max())

    return {"prediction": pred, "probability": proba}


