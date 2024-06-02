import torch
import torch.nn as nn
from config import settings

class AirModel_GRU(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x

def load_model():
    hidden_size = 100
    num_layers = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AirModel_GRU(hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(settings.model_path, map_location=device))
    model.eval()
    return model

model = load_model()