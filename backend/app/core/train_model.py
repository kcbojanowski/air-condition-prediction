import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from app.core.model import AirModelGRU
from app.core.data_processing import normalization, create_dataset
from app.core.config import settings

async def build_and_train():
    data = pd.read_csv('data/Alaska_PM10_one_site.csv')
    data_normalized = normalization(data['PM10'].astype(float).values)
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
    
    torch.save(model.state_dict(), settings.model_path)