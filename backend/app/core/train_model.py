import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from app.core.model import AirModelGRU
from app.core.data_processing import normalization, create_dataset
from app.core.config import settings

def train_model():
    data = pd.read_csv('data/Alaska_PM10_one_site.csv')
    data_normalized = normalization(data['pm10'].values)
    lookback = 3
    X, y = create_dataset(data_normalized, lookback)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AirModelGRU(hidden_size=100, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    model.train()
    epochs = 10
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), settings.model_path)