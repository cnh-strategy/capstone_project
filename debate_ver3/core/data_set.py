# debate_ver3\core\data_set.py
# ===============================================================


import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from debate_ver3.config.agents import agents_info, dir_info


# --------------------------------------------
# 1. Raw Dataset 생성
# --------------------------------------------
def fetch_ticker_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """yfinance로 주가 데이터 다운로드"""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)

    # 컬럼명 정리 (튜플 형태를 단순 문자열로 변환)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 기본 기술적 지표
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["sma_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["sma_20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)

    # Fundamental Agent용 추가 데이터
    try:
        usd_krw = yf.download("USDKRW=X", period=period, interval=interval, auto_adjust=True)
        nasdaq  = yf.download("^IXIC",      period=period, interval=interval, auto_adjust=True)
        vix     = yf.download("^VIX",       period=period, interval=interval, auto_adjust=True)

        df["USD_KRW"] = (usd_krw["Close"] if not usd_krw.empty else 1300.0)
        df["NASDAQ"]  = (nasdaq["Close"]  if not nasdaq.empty  else 15000.0)
        df["VIX"]     = (vix["Close"]     if not vix.empty     else 20.0)

        # 인덱스 정렬/정합
        for c in ["USD_KRW", "NASDAQ", "VIX"]:
            if isinstance(df[c], pd.Series):
                df[c] = df[c].reindex(df.index, method="ffill")
    except Exception as e:
        print(f"⚠️ 추가 지표 다운로드 실패: {e}")
        df["USD_KRW"] = 1300.0
        df["NASDAQ"]  = 15000.0
        df["VIX"]     = 20.0

    # Sentimental Agent용 감성 proxy 지표
    df["sentiment_mean"] = df["returns"].rolling(3, min_periods=1).mean().fillna(0)
    df["sentiment_vol"]  = df["returns"].rolling(3, min_periods=1).std().fillna(0)

    df.dropna(inplace=True)
    return df

# --------------------------------------------
# 2️⃣ 시퀀스 생성
# --------------------------------------------
def create_sequences(features, target, window_size=14):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

# --------------------------------------------
# 4️⃣ 통합 함수
# --------------------------------------------
def build_dataset(ticker: str = "TSLA", save_dir=dir_info["data_dir"]):
    os.makedirs(save_dir, exist_ok=True)

    # 공통 raw
    raw_df = fetch_ticker_data(ticker, period="2y", interval="1d")
    raw_df.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True)

    for agent_id, cfg in agents_info.items():
        period   = cfg.get("period", "2y")
        interval = cfg.get("interval", "1d")

        df = fetch_ticker_data(ticker, period=period, interval=interval)

        col = cfg["data_cols"]
        # 차원 정합 체크
        assert len(col) == cfg.get("input_dim", len(col)), \
            f"[{agent_id}] input_dim({cfg.get('input_dim')}) != data_cols({len(col)})"

        X = df[col]
        # 다음날 수익률 타깃
        y = df["Close"].pct_change().shift(-1).fillna(0).values.reshape(-1, 1)

        X_seq, y_seq = create_sequences(X, y, window_size=cfg["window_size"])
        samples, time_steps, features = X_seq.shape
        print(f"[{agent_id}] X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")

        # 평면화 저장
        flattened_data = []
        for sample_idx in range(samples):
            for time_idx in range(time_steps):
                row = {
                    'sample_id': sample_idx,
                    'time_step': time_idx,
                    'target': y_seq[sample_idx, 0] if time_idx == time_steps - 1 else np.nan,
                }
                for feat_idx, feat_name in enumerate(col):
                    row[feat_name] = X_seq[sample_idx, time_idx, feat_idx]
                flattened_data.append(row)

        agent_df = pd.DataFrame(flattened_data)
        csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
        agent_df.to_csv(csv_path, index=False)
        print(f"✅ {ticker} {agent_id} dataset saved to CSV ({len(X_seq)} samples, {len(col)} features)")

# --------------------------------------------
# 5️⃣ CSV 데이터 로드 함수들
# --------------------------------------------
def load_dataset(ticker, agent_id=None, save_dir=dir_info["data_dir"]):
    csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in ["sample_id", "time_step", "target"]]
    unique_samples = df["sample_id"].nunique()
    time_steps     = df["time_step"].nunique()
    n_features     = len(feature_cols)

    X = np.zeros((unique_samples, time_steps, n_features), dtype=np.float32)
    y = np.zeros((unique_samples, 1), dtype=np.float32)

    for i, sid in enumerate(df["sample_id"].unique()):
        sdata = df[df["sample_id"] == sid].sort_values("time_step")
        X[i] = sdata[feature_cols].values
        y[i, 0] = sdata["target"].iloc[-1] if not np.isnan(sdata["target"].iloc[-1]) else 0.0

    return X, y, feature_cols

def get_latest_close_price(ticker, save_dir=dir_info["data_dir"]):
    raw_data_path = os.path.join(save_dir, f"{ticker}_raw_data.csv")
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, index_col=0)
        return float(df["Close"].iloc[-1])
    data = yf.download(ticker, period="1d", interval="1d")
    return float(data["Close"].iloc[-1])

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))