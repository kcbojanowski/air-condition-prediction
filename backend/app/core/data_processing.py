import numpy as np
import torch


class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, data_array):
        self.min_val = np.min(data_array)
        self.max_val = np.max(data_array)

    def transform(self, data_array):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before calling transform.")
        scaled = (data_array - self.min_val) / (self.max_val - self.min_val)
        return scaled

    def inverse_transform(self, normalized_array):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before calling inverse_transform.")
        return normalized_array * (self.max_val - self.min_val) + self.min_val

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)