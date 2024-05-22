import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
from pyspark.streaming import StreamingContext


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network for air quality prediction.')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train for')
    parser.add_argument('--hidden_size', type=int, default=100, help='The size of the hidden layer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_layers', type=int, default=1, help='The number of layers in the model')
    parser.add_argument('--data_path', type=str, default='/app/data/Alaska_PM10_one_site.csv',
                        help='Path to the CSV data file')
    parser.add_argument('--streaming_host', type=str, default='localhost', help='Host for streaming data')
    parser.add_argument('--streaming_port', type=int, default=9999, help='Port for streaming data')
    return parser.parse_args()


def load_data(spark, filepath):
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    timeseries = df.select(col("PM10").cast("float")).toPandas().values
    train_size = int(len(timeseries) * 0.67)
    return normalization(timeseries[:train_size]), normalization(timeseries[train_size:])


def normalization(data_array):
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    scaled = (data_array - min_val) / (max_val - min_val)
    return scaled


def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


class AirModel_GRU(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x


def train_model(args, X_train, y_train, X_test, y_test, device):
    model = AirModel_GRU(args.hidden_size, args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    n_epochs = args.epochs

    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        model.train()
        for batch in loader:
            X, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
    return model


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    spark = SparkSession.builder.appName("AirQualityPrediction").getOrCreate()

    train, test = load_data(spark, args.data_path)
    X_train, y_train = create_dataset(train, lookback=3)
    X_test, y_test = create_dataset(test, lookback=3)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model = train_model(args, X_train, y_train, X_test, y_test, device)
    torch.save(model.state_dict(), "model.pth")

    ssc = StreamingContext(spark.sparkContext, 1)
    lines = ssc.socketTextStream(args.streaming_host, args.streaming_port)

    def process_stream(rdd):
        if not rdd.isEmpty():
            df = rdd.toDF(["PM10"])
            timeseries = df.select(col("PM10").cast("float")).toPandas().values
            timeseries = normalization(timeseries)
            X, _ = create_dataset(timeseries, lookback=3)
            X = X.to(device)

            model.eval()
            with torch.no_grad():
                predictions = model(X)
                print(predictions.cpu().numpy())

    lines.foreachRDD(process_stream)

    ssc.start()
    ssc.awaitTermination()

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()