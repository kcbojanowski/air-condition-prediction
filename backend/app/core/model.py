import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from app.core.data_processing import create_dataset, Normalizer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col
import os

class AirModelGRU(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x


class ModelInstance():
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.spark = SparkSession.builder.appName("AirQualityApp").getOrCreate()


    def load_model(self):
        hidden_size = 100
        num_layers = 2
        model = AirModelGRU(hidden_size, num_layers).to(self.device)
        model.load_state_dict(torch.load(os.environ['MODEL_PATH'], map_location=self.device))
        model.eval()
        return model


    def evaluate_model(self):
        df = self.spark.read.csv('data/Alaska_PM10_one_site.csv', header=True, inferSchema=True)
        data = np.array(df.select(col("PM10").cast(FloatType())).rdd.flatMap(lambda x: x).collect())
        

        train_size = int(len(data) * 0.8)
        test_data = data[train_size:]

        normalizer = Normalizer()
        normalizer.fit(test_data)
        data_normalized = normalizer.transform(test_data)

        lookback = 3
        X, y = create_dataset(data_normalized, lookback)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        predictions = []
        targets = []
        with torch.no_grad():
            for batch in dataloader:
                X_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
                y_pred = self.model(X_batch)
                predictions.extend(y_pred.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)[: , -1].reshape(-1, 1)

        print(predictions.shape)
        print(targets.shape)

        mae = mean_absolute_error(targets, predictions).item()
        mse = mean_squared_error(targets, predictions).item()
        rmse = np.sqrt(mse).item()

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }

        return metrics
