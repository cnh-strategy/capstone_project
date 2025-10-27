import torch
import torch.nn as nn

class StockSentimentLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, input_size=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).unsqueeze(-1)  # 차원 (batch_size, 1) 으로 변경
