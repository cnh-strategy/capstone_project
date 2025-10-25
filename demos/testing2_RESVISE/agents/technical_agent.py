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

# 기술적 분석 에이전트 클래스
class TechnicalAgent(nn.Module):
    # Technical Analysis Agent - TCN based
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Temporal Convolutional Network
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # 필수 - Stage 3 불확실성 추정용
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim//2, output_dim)
        
    def forward(self, x):
        # x shape: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x