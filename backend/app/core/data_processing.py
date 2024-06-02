import numpy as np
import torch


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
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
