import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from app.core.data_processing import create_dataset, normalization
from app.core.config import settings
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score


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
        

    def load_model(self):
        hidden_size = 100
        num_layers = 2
        model = AirModelGRU(hidden_size, num_layers).to(self.device)
        model.load_state_dict(torch.load(settings.model_path, map_location=self.device))
        model.eval()
        return model


    def evaluate_model(self, X, y):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X.to(device))
        
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        
        accuracy = accuracy_score(y, predictions.round())
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        auc = roc_auc_score(y, predictions)
        
        metrics = {
            "accuracy": accuracy,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "auc": auc
        }

        return metrics
