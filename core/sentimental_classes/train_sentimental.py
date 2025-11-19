# core/sentimental_classes/train_sentimental.py

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import joblib  # pip install joblib 필요할 수 있음

# 한 번에 학습할 티커들
TICKERS = ["NVDA", "MSFT", "AAPL"]

BASE_DATA_DIR = "data/datasets"
BASE_MODEL_DIR = "models"
BASE_SCALER_DIR = os.path.join("models", "scalers")

FEATURE_COLS = [
    "return_1d",
    "hl_range",
    "Volume",
    "news_count_1d",
    "news_count_7d",
    "sentiment_mean_1d",
    "sentiment_mean_7d",
    "sentiment_vol_7d",
]

WINDOW_SIZE = 40
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 30
LR = 1e-3


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)  # next-day return 예측

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # 시퀀스 마지막 타임스텝
        y = self.fc(last)
        return y


def build_sequences(df: pd.DataFrame):
    """
    df: Date + FEATURE_COLS
    target: 다음날 return_1d
    """
    df = df.copy()
    df["target"] = df["return_1d"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    X_list = []
    y_list = []

    values = df[FEATURE_COLS].values
    targets = df["target"].values

    for i in range(len(df) - WINDOW_SIZE + 1):
        X_list.append(values[i : i + WINDOW_SIZE])
        y_list.append(targets[i + WINDOW_SIZE - 1])

    X = np.stack(X_list)  # (N, T, F)
    y = np.array(y_list).reshape(-1, 1)
    return X, y


def train_for_ticker(ticker: str):
    data_csv = os.path.join(BASE_DATA_DIR, f"{ticker}_sentimental_dataset.csv")
    if not os.path.exists(data_csv):
        print(f"⚠ {ticker}: {data_csv} 가 없습니다. 건너뜁니다.")
        return

    print(f"\n===== 🔧 {ticker} 학습 시작 =====")
    df = pd.read_csv(data_csv)
    print(f"[{ticker}] 데이터 shape:", df.shape)

    X, y = build_sequences(df)
    print(f"[{ticker}] X: {X.shape}, y: {y.shape}")

    # 스케일링
    N, T, F = X.shape
    scaler = StandardScaler()
    X_2d = X.reshape(-1, F)
    X_scaled_2d = scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(N, T, F)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel(
        input_dim=len(FEATURE_COLS),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{ticker}] Epoch {epoch}/{EPOCHS}  loss={loss.item():.6f}")

    # 저장 경로 준비
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    os.makedirs(BASE_SCALER_DIR, exist_ok=True)

    model_path = os.path.join(BASE_MODEL_DIR, f"{ticker}_SentimentalAgent.pt")
    scaler_path = os.path.join(BASE_SCALER_DIR, f"{ticker}_SentimentalAgent.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(
        {
            "feature_cols": FEATURE_COLS,
            "window_size": WINDOW_SIZE,
            "scaler": scaler,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        scaler_path,
    )

    print(f"[{ticker}] ✅ 모델 저장: {model_path}")
    print(f"[{ticker}] ✅ 스케일러 메타 저장: {scaler_path}")
    print(f"===== ✅ {ticker} 학습 완료 =====\n")


def main():
    for t in TICKERS:
        train_for_ticker(t)


if __name__ == "__main__":
    main()
