import random
import time
from fastapi import BackgroundTasks, WebSocket, APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import torch
import json
import numpy as np
# from app.db.database import SessionLocal, AirQualityData
from app.core.model import ModelInstance
from app.core.train_model import build_and_train
from app.core.data_processing import normalization, create_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType


router = APIRouter()


# Tworzymy sesję Spark
spark = SparkSession.builder.appName("AirQualityApp").getOrCreate()

# Definiujemy schemat dla danych PM10
schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("pm10", FloatType(), True)
])


@router.post("/build-and-train")
async def build_train(background_tasks: BackgroundTasks):
    start_time = time.time()  
    await build_and_train()  
    end_time = time.time()  
    elapsed_time = end_time - start_time

    return {
        "status": "Model building and training completed",
        "time_taken": elapsed_time
    }
    

@router.post("/evaluate")
async def evaluate():
    model_instance = ModelInstance()
    metrics = model_instance.evaluate_model()
    return metrics

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    data = []
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            msg_json = json.loads(msg)
            data.append(msg_json['pm10'])
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Otrzymane dane PM10:", data)
        model_instance = ModelInstance()
        normalized_data = normalization(np.array([data]))
        X, _ = create_dataset(normalized_data, lookback=3)
        X = X.to(model_instance.device)
        
        with torch.no_grad():
            prediction = model_instance.model(X)
        return {"prediction": prediction.cpu().numpy()}
