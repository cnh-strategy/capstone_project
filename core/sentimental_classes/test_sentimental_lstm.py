# test_sentimental_lstm.py

import pandas as pd
import numpy as np
import torch
import joblib

from core.sentimental_classes.train_sentimental import (
    LSTMModel,
    FEATURE_COLS,
    WINDOW_SIZE,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
)

TICKER = "NVDA"

data_csv = f"data/datasets/{TICKER}_sentimental_dataset.csv"
scaler_path = f"models/scalers/{TICKER}_SentimentalAgent.pkl"
model_path = f"models/{TICKER}_SentimentalAgent.pt"

# 1) 데이터 로드
df = pd.read_csv(data_csv)
print("df shape:", df.shape)

# 2) 마지막 윈도우 하나 만들기
values = df[FEATURE_COLS].values
last_seq = values[-WINDOW_SIZE:]           # (40, 5)

# 3) 스케일러 로드 + 스케일링
meta = joblib.load(scaler_path)
scaler = meta["scaler"]

last_seq_scaled = scaler.transform(last_seq)          # (40, 5)
X = last_seq_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))  # (1, 40, 5)

# 4) 모델 로드
model = LSTMModel(
    input_dim=len(FEATURE_COLS),
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    out = model(torch.tensor(X, dtype=torch.float32))
    next_return = float(out[0, 0])

print(f"📈 {TICKER} 예측 next-day return ≈ {next_return:.4f}")
