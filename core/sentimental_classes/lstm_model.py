# core/sentimental_classes/lstm_model.py
import torch
import torch.nn as nn

class SentimentalLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]   # (batch, hidden_dim)
        y = self.fc(last_hidden)      # (batch, 1)
        return y.squeeze(-1)          # (batch,)
