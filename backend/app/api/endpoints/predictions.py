import random
import time
from fastapi import BackgroundTasks, FastAPI, WebSocket, APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import torch
import json
import numpy as np
# from app.db.database import SessionLocal, AirQualityData
from app.core.model import ModelInstance
from app.core.train_model import build_and_train
from app.core.data_processing import Normalizer, create_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from torch.utils.data import DataLoader, TensorDataset

app = FastAPI()
router = APIRouter()


predictions= []

# Tworzymy sesję Spark
spark = SparkSession.builder.appName("AirQualityApp").getOrCreate()

# Definiujemy schemat dla danych PM10
schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("pm10", FloatType(), True)
])

data_storage = []

@router.post("/get-predictions")
async def get_predictions():
    return {"data": data_storage}


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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    data = []
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            msg_json = json.loads(msg)
            data.append(msg_json['pm10'])
    except Exception as e:
        pass
    finally:
        print("Otrzymane dane PM10:", data)
        model_instance = ModelInstance()
        normalizer = Normalizer()
        normalizer.fit(np.array(data))
        normalized_data = normalizer.transform(np.array(data))

        lookback = 3
        X, y = create_dataset(normalized_data, lookback)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                X_batch = batch[0].to(model_instance.device)
                y_pred = model_instance.model(X_batch)
                predictions.extend(y_pred.cpu().numpy())

        predictions = normalizer.inverse_transform(np.array(predictions)).flatten()        
        print("Predictions for the next 5 days:", predictions[:5])
        
