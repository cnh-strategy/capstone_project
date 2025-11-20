# core/sentimental_classes/train_sentimental.py

import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import joblib
import yfinance as yf

from core.sentimental_classes.news import merge_price_with_news_features

TICKERS = ["NVDA", "MSFT", "AAPL"]

BASE_DATA_DIR = Path("data") / "datasets"
BASE_MODEL_DIR = Path("models")
BASE_SCALER_DIR = BASE_MODEL_DIR / "scalers"

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
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


def build_dataset_with_news(
    ticker: str,
    start: str,
    end: str,
    window_size: int = WINDOW_SIZE,
    feature_cols: List[str] = FEATURE_COLS,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    가격 + 뉴스 피처를 merge한 df로부터
    LSTM 학습용 (X, y) 시퀀스 만들기.
    """
    # 1) 가격 + 뉴스 merge (여기서 내부적으로 FinBERT까지 돌았다고 가정)
    df_merged = merge_price_with_news_features(
        ticker=ticker,
        start=start,
        end=end,
        base_dir_news=os.path.join("data", "raw", "news"),
    )

    # 2) feature, target 구성
    df_merged = df_merged.sort_index()
    X_df = df_merged[feature_cols].copy()
    # 타겟: 다음날 종가 (또는 수익률 등, 기존 정의에 맞춰 조정)
    y_series = df_merged["next_close"].copy()  # 이미 next_close 컬럼이 있다고 가정

    # 마지막 window를 만들 수 없는 뒷부분 잘라내기
    valid_len = len(X_df) - window_size
    if valid_len <= 0:
        raise ValueError(f"[build_dataset_with_news] not enough data for {ticker}")

    X_values = X_df.values.astype(float)
    y_values = y_series.values.astype(float)

    # 3) 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)

    # 4) 시퀀스 생성
    X_list = []
    y_list = []
    for i in range(valid_len):
        X_window = X_scaled[i : i + window_size]
        y_target = y_values[i + window_size - 1]  # window 마지막 시점 기준 타겟
        X_list.append(X_window)
        y_list.append(y_target)

    X = np.stack(X_list, axis=0)  # (N, T, F)
    y = np.array(y_list)          # (N,)

    return X, y, scaler, feature_cols


def pretrain_sentimental_single(
    ticker: str,
    years: int = 5,
    device: str = "cpu",
) -> None:
    """
    개별 티커에 대해 가격 + 뉴스 기반 LSTM 사전학습 수행.
    """
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)

    X, y, scaler, feature_cols = build_dataset_with_news(
        ticker=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        window_size=WINDOW_SIZE,
        feature_cols=FEATURE_COLS,
    )

    # 저장 경로
    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BASE_SCALER_DIR.mkdir(parents=True, exist_ok=True)

    ds_path = BASE_DATA_DIR / f"{ticker}_SentimentalAgent.npz"
    scaler_path = BASE_SCALER_DIR / f"{ticker}_SentimentalAgent_scaler.pkl"
    model_path = BASE_MODEL_DIR / f"{ticker}_SentimentalAgent.pt"

    # 데이터셋 저장 (원하면 meta도 같이)
    np.savez_compressed(
        ds_path,
        X=X,
        y=y,
        feature_cols=np.array(feature_cols),
        window_size=np.array([WINDOW_SIZE]),
    )

    # 스케일러 저장
    joblib.dump(scaler, scaler_path)

    # 모델 학습
    device = torch.device(device)
    model = LSTMModel(
        input_dim=X.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = loss_fn(out, y_tensor)
        loss.backward()
        optimizer.step()
        # 필요하면 print(f"[{ticker}] epoch {epoch+1}/{EPOCHS}, loss={loss.item():.4f}")

    # 모델 저장
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "window_size": WINDOW_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        model_path,
    )
