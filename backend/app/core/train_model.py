import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col

from tqdm import tqdm
from app.core.model import AirModelGRU
from app.core.data_processing import Normalizer, create_dataset

spark = SparkSession.builder.appName("AirQualityApp").getOrCreate()


async def build_and_train():
    df = spark.read.csv('data/Alaska_PM10_one_site.csv', header=True, inferSchema=True)
    data = np.array(df.select(col("PM10").cast(FloatType())).rdd.flatMap(lambda x: x).collect())

    train_size = int(len(data) * 0.8)
    normalizer = Normalizer()
    normalizer.fit(data[:train_size])
    data_normalized = normalizer.transform(data[:train_size])

    lookback = 3
    epochs = 10

    X, y = create_dataset(data_normalized, lookback)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AirModelGRU(hidden_size=100, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train()
        train_loss = 0
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
    
    train_loss /= len(dataloader.dataset)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    
    torch.save(model.state_dict(), os.environ['MODEL_PATH'])

    return train_loss