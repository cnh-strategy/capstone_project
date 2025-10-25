import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings("ignore")

# 감정 분석 에이전트 클래스
class SentimentalAgent(nn.Module):
    # Sentimental Analysis Agent - Transformer based
    def __init__(self, input_dim=8, d_model=64, nhead=4, num_layers=2, output_dim=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)  # 필수 - Stage 3 불확실성 추정용
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: (batch, time, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Use the last time step output
        last_output = x[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output