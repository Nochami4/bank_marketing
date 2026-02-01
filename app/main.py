from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


FEATURE_COLUMNS = [
    "age",
    "balance",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

app = FastAPI(title="Bank Marketing API")
model = None


class PredictRequest(BaseModel):
    age: float
    balance: float
    day_of_week: float
    campaign: float
    pdays: float
    previous: float
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    poutcome: str


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    data = pd.DataFrame([payload], columns=FEATURE_COLUMNS)

    proba = float(model.predict_proba(data)[0][1])
    label = int(proba >= 0.5)

    return {"proba": proba, "label": label}
