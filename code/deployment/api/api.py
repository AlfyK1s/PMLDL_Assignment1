from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("models/house_price_model.pkl")

app = FastAPI()

class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    YearBuilt: int
    Neighborhood: str

@app.post("/predict")
def predict(features: dict):
    data = pd.DataFrame([features])
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
