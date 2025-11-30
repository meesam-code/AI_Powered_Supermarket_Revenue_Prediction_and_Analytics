# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# ---------- FASTAPI APP ----------
app = FastAPI(
    title="Supermarket Revenue Prediction API",
    description="Predict revenue from store/item features",
    version="1.0"
)

# ---------- LOAD MODEL ----------
MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ---------- REQUEST SCHEMA ----------
class PredictRequest(BaseModel):
    Item_Category: str
    Store_Type: str
    Outlet_Location_Type: str
    Outlet_Size: str
    Item_Weight: float
    Item_MRP: float
    Outlet_Sales: float

# ---------- HOME ROUTE ----------
@app.get("/")
def home():
    return {"message": "Welcome to the Supermarket Revenue Prediction API! Visit /docs for API usage."}

# ---------- PREDICT ROUTE ----------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        input_df = pd.DataFrame([req.dict()])
        revenue = model.predict(input_df)[0]
        return {"Predicted_Revenue": round(float(revenue), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
