from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import SessionLocal, AirQualityData
from app.core.model import model
from app.core.data_processing import normalization, create_dataset
import numpy as np

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/ingest")
async def ingest_data(pm10: float, timestamp: str, db: Session = Depends(get_db)):
    normalized_data = normalization(np.array([pm10]))
    new_data = AirQualityData(pm10=pm10, timestamp=timestamp, prediction=None)
    db.add(new_data)
    db.commit()
    return {"status": "success"}

@router.post("/predict")
async def predict_data(pm10: float):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    normalized_data = normalization(np.array([pm10]))
    X, _ = create_dataset(normalized_data, lookback=3)
    X = X.to(device)

    with torch.no_grad():
        prediction = model(X)
    return {"prediction": prediction.cpu().numpy()}