import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib  # pip install joblib


# ===============================
# 기본 설정
# ===============================

SYMBOLS = ["NVDA", "MSFT", "AAPL"]  # 학습할 티커들

NEWS_CSV = "news_data.csv"          # 앞에서 수집한 뉴스 CSV
STOCK_CSV = "stock_data.csv"        # 앞에서 수집한 주가 CSV

WINDOW_SIZE = 40
TARGET_COL = "Close"                # 다음날 Close 예측

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2

MODELS_DIR = "models"
SCALERS_DIR = os.path.join(MODELS_DIR, "scalers")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# LSTM 모델 정의
# ===============================

class SentimentalLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
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
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]  # 마지막 타임스텝 hidden
        y = self.fc(last_hidden)     # (batch, 1)
        return y.squeeze(-1)         # (batch,)


# ===============================
# Dataset
# ===============================

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===============================
# 유틸: CSV 로드
# ===============================

def load_news_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_stock_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


# ===============================
# 유틸: 뉴스 + 주가 merge & feature 생성
# ===============================

def merge_and_create_features(
    symbol: str, news_df: pd.DataFrame, stock_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    1) 심볼별로 주가/뉴스 필터링
    2) 일자 기준 뉴스 감성 집계
    3) 주가와 merge
    4) 감성/가격 기반 feature 8개 생성
    """
    # --- 주가 ---
    s_stock = stock_df[stock_df["Symbol"] == symbol].copy()
    s_stock = s_stock.sort_values("Date")
    s_stock.set_index("Date", inplace=True)

    # 1일 수익률
    s_stock["return_1d"] = s_stock["Close"].pct_change()

    # --- 뉴스 ---
    s_news = news_df[news_df["ticker"] == symbol].copy()
    s_news = s_news.rename(columns={"date": "Date"})
    s_news["Date"] = pd.to_datetime(s_news["Date"]).dt.date
    s_news = s_news.sort_values("Date")

    # sentiment_score 숫자형 변환
    s_news["sentiment_score"] = pd.to_numeric(
        s_news["sentiment_score"], errors="coerce"
    )

    grouped = s_news.groupby("Date")

    daily_sent = grouped["sentiment_score"].mean().rename("sentiment_daily")
    daily_pos_ratio = (
        (grouped["sentiment_score"].apply(lambda x: (x > 0).sum()) / grouped["sentiment_score"].count())
        .rename("pos_ratio")
    )
    daily_neg_ratio = (
        (grouped["sentiment_score"].apply(lambda x: (x < 0).sum()) / grouped["sentiment_score"].count())
        .rename("neg_ratio")
    )
    daily_news_count = grouped["sentiment_score"].count().rename("news_count_1d")

    daily_df = pd.concat(
        [daily_sent, daily_pos_ratio, daily_neg_ratio, daily_news_count], axis=1
    )

    # --- 주가와 뉴스 merge (left join: 주가 기준) ---
    merged = s_stock.join(daily_df, how="left")

    # 뉴스 없는 날: 0으로
    merged["sentiment_daily"] = merged["sentiment_daily"].fillna(0.0)
    merged["pos_ratio"] = merged["pos_ratio"].fillna(0.0)
    merged["neg_ratio"] = merged["neg_ratio"].fillna(0.0)
    merged["news_count_1d"] = merged["news_count_1d"].fillna(0.0)

    # --- 롤링 피처 ---
    merged["sentiment_mean_7d"] = merged["sentiment_daily"].rolling(window=7, min_periods=1).mean()
    merged["sentiment_vol_7d"] = merged["sentiment_daily"].rolling(window=7, min_periods=1).std().fillna(0.0)

    # 간단한 trend_7d: 최근값 - 7일 전
    merged["trend_7d"] = merged["sentiment_daily"] - merged["sentiment_daily"].shift(7)
    merged["trend_7d"] = merged["trend_7d"].fillna(0.0)

    # 최근 7일 뉴스 개수
    merged["news_count_7d"] = merged["news_count_1d"].rolling(window=7, min_periods=1).sum()

    # --- 타겟: 다음날 Close ---
    merged["target_next_close"] = merged["Close"].shift(-1)

    merged = merged.dropna(subset=["return_1d", "target_next_close"])

    # feature 8개 (input_dim = 8)
    feature_cols = [
        "return_1d",
        "sentiment_daily",
        "sentiment_mean_7d",
        "sentiment_vol_7d",
        "trend_7d",
        "pos_ratio",
        "neg_ratio",
        "news_count_7d",
    ]

    merged = merged.reset_index()  # Date 다시 컬럼으로
    return merged, feature_cols


# ===============================
# 유틸: 시퀀스(X_seq) & y 생성
# ===============================

def build_sequences(
    df: pd.DataFrame, feature_cols: List[str], window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    df: merge_and_create_features 결과
    """
    values = df[feature_cols].values
    targets = df["target_next_close"].values

    X_list, y_list = [], []

    for i in range(len(df) - window_size):
        X_list.append(values[i : i + window_size])
        # window 마지막 날짜 기준 '다음날' 종가(=target_next_close)를 예측
        y_list.append(targets[i + window_size - 1])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


# ===============================
# 학습 루프 (티커 하나)
# ===============================

def train_one_symbol(
    symbol: str, news_df: pd.DataFrame, stock_df: pd.DataFrame
):
    print(f"\n========== [{symbol}] 학습 시작 ==========")

    merged, feature_cols = merge_and_create_features(symbol, news_df, stock_df)

    print(f"[{symbol}] 총 행 개수: {len(merged)}")
    print(f"[{symbol}] feature_cols: {feature_cols}")

    if len(merged) <= WINDOW_SIZE + 10:
        print(f"[{symbol}] 데이터가 너무 적어 학습을 건너뜁니다.")
        return

    X, y = build_sequences(merged, feature_cols, WINDOW_SIZE)
    print(f"[{symbol}] X shape: {X.shape}, y shape: {y.shape}")

    # --- train/valid 시간 순 split ---
    n = len(X)
    n_train = int(n * 0.8)

    X_train, X_valid = X[:n_train], X[n_train:]
    y_train, y_valid = y[:n_train], y[n_train:]

    # --- 스케일링 ---
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)

    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_valid_scaled = scaler.transform(
        X_valid.reshape(-1, X_valid.shape[-1])
    ).reshape(X_valid.shape)

    train_ds = SequenceDataset(X_train_scaled, y_train)
    valid_ds = SequenceDataset(X_valid_scaled, y_valid)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 모델 ---
    input_dim = X_train.shape[-1]
    model = SentimentalLSTM(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else None

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        print(
            f"[{symbol}] Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    # --- best state 로드 후 저장 ---
    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = os.path.join(MODELS_DIR, f"{symbol}_SentimentalAgent.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[{symbol}] 모델 저장 완료: {model_path}")

    # --- scaler & 메타데이터 저장 ---
    scaler_path = os.path.join(
        SCALERS_DIR, f"{symbol}_SentimentalAgent.pkl"
    )
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "scaler": scaler,
            "window_size": WINDOW_SIZE,
        },
        scaler_path,
    )
    print(f"[{symbol}] 스케일러/메타데이터 저장 완료: {scaler_path}")


# ===============================
# 메인
# ===============================

def main():
    if not os.path.exists(NEWS_CSV) or not os.path.exists(STOCK_CSV):
        print("news_data.csv 또는 stock_data.csv가 없습니다. 먼저 수집 스크립트를 실행하세요.")
        return

    news_df = load_news_df(NEWS_CSV)
    stock_df = load_stock_df(STOCK_CSV)

    print("뉴스/주가 CSV 로드 완료.")

    for symbol in SYMBOLS:
        train_one_symbol(symbol, news_df, stock_df)


if __name__ == "__main__":
    main()
