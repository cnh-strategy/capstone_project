# core/data_set.py
# ===============================================================
# 멀티에이전트용 데이터셋 빌더 및 로더
#  - if / elif 분기: 매크로 / 센티멘탈 / 테크니컬
#  - 센티멘탈 분기에서는 "네가 맨 처음 썼던 코드" 기반 로직 사용
#  - 타깃: 다음날 수익률 (Close_{t+1}/Close_t - 1)
#  - 저장: {save_dir}/{ticker}_{agent_id}_dataset.csv
# ===============================================================

from __future__ import annotations
import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from config.agents import agents_info, dir_info

# ---- optional macro import (지연) : 파일 상단에서 '한 번만' 처리 ----
_HAS_MACRO = False
_MACRO_IMPORT_ERROR = ""
try:
    # 1) 패키지 내부 상대 임포트
    from .macro_classes.macro_funcs import macro_dataset as _macro_dataset
    macro_dataset = _macro_dataset
    _HAS_MACRO = True
    _MACRO_SRC = "relative: .macro_classes.macro_funcs"
except Exception as e1:
    try:
        # 2) 절대 임포트 (환경에 따라 상대가 막힐 때)
        from core.macro_classes.macro_funcs import macro_dataset as _macro_dataset
        macro_dataset = _macro_dataset
        _HAS_MACRO = True
        _MACRO_SRC = "absolute: core.macro_classes.macro_funcs"
    except Exception as e2:
        macro_dataset = None
        _HAS_MACRO = False
        _MACRO_IMPORT_ERROR = f"{type(e1).__name__}: {e1} | {type(e2).__name__}: {e2}"
        _MACRO_SRC = "unavailable"


# -----------------------------
# 공용 유틸
# -----------------------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def create_sequences(features: pd.DataFrame, target: np.ndarray, window_size: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features.iloc[i:i + window_size].to_numpy())
        y.append(target[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def _save_agent_csv(flattened_rows: List[dict], csv_path: str) -> None:
    agent_df = pd.DataFrame(flattened_rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    agent_df.to_csv(csv_path, index=False, encoding="utf-8")


# -----------------------------
# 센티멘탈 전용(네 초기 코드 기반)
# -----------------------------
def _fetch_ticker_data_for_sentimental(ticker: str, period: Optional[str], interval: Optional[str]) -> pd.DataFrame:
    period = period or "2y"
    interval = interval or "1d"

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    df.dropna(inplace=True)

    # 멀티인덱스 컬럼 방어
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 기본 기술 지표
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)

    # Fundamental 보조(USD/KRW, NASDAQ, VIX)
    try:
        usd_krw = yf.download("USDKRW=X", period=period, interval=interval, auto_adjust=True, progress=False)
        df["USD_KRW"] = (usd_krw["Close"].reindex(df.index, method="ffill") if not usd_krw.empty else 1300.0)

        nasdaq = yf.download("^IXIC", period=period, interval=interval, auto_adjust=True, progress=False)
        df["NASDAQ"] = (nasdaq["Close"].reindex(df.index, method="ffill") if not nasdaq.empty else 15000.0)

        vix = yf.download("^VIX", period=period, interval=interval, auto_adjust=True, progress=False)
        df["VIX"] = (vix["Close"].reindex(df.index, method="ffill") if not vix.empty else 20.0)
    except Exception as e:
        print(f"⚠️ 추가 지표 다운로드 실패: {e}")
        df["USD_KRW"] = 1300.0
        df["NASDAQ"] = 15000.0
        df["VIX"] = 20.0

    # 감성(placeholder): 초기 코드 그대로
    df["sentiment_mean"] = df["returns"].rolling(3).mean().fillna(0)
    df["sentiment_vol"] = df["returns"].rolling(3).std().fillna(0)

    df.dropna(inplace=True)
    return df


# -----------------------------
# 공개 API
# -----------------------------
def build_dataset(
    ticker: str,
    save_dir: str = dir_info["data_dir"],
    agent_id: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> None:
    """
    debate_agent.py에서 agent_id를 넘겨주면, 여기서 분기 처리합니다.
    - agent_id == 'MacroSentiAgent' / '매크로' / 'macro'
    - agent_id == 'SentimentalAgent' / '센티멘탈' / 'sentimental'
    - agent_id == 'TechnicalAgent' / '테크니컬' / 'technical' (추후)
    """
    os.makedirs(save_dir, exist_ok=True)
    aid = (agent_id or "").strip().lower()

    # ---------- macro_agent ----------
    if aid in {"macrosentiagent", "macro", "매크로"}:
        if not _HAS_MACRO or macro_dataset is None:
            raise ImportError(
                "macro_dataset 모듈을 찾을 수 없습니다. core/macro_classes 확인 필요. "
                f"details={_MACRO_IMPORT_ERROR}"
            )
        # 팀 구현에 맞게 호출
        macro_dataset(ticker_name=ticker)
        print(f"✅ {ticker} MacroSentiAgent dataset saved (macro_dataset 호출 via {_MACRO_SRC})")

    # ---------- sentimental_agent ----------
    elif aid in {"sentimentalagent", "sentimental", "센티멘탈"}:
        df = _fetch_ticker_data_for_sentimental(ticker, period, interval)

        # 원본 CSV 저장(후속처리 참고용)
        df.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True, encoding="utf-8")

        # 사용할 피처 컬럼
        if agent_id in agents_info and "data_cols" in agents_info[agent_id]:
            feature_cols = agents_info[agent_id]["data_cols"]
            window_size = agents_info[agent_id].get("window_size", 14)
        else:
            # fallback: 네 초기 코드 기반으로 합리적 기본 피처
            feature_cols = [
                "returns", "sma_5", "sma_20", "rsi", "volume_z",
                "USD_KRW", "NASDAQ", "VIX",
                "sentiment_mean", "sentiment_vol",
                "Open", "High", "Low", "Close", "Volume",
            ]
            window_size = 14

        # 타깃: 다음날 수익률
        returns = df["Close"].pct_change().shift(-1)
        valid_mask = ~returns.isna()
        y = returns[valid_mask].to_numpy().reshape(-1, 1)
        X = df.loc[valid_mask, feature_cols]

        # 시퀀스 생성
        X_seq, y_seq = create_sequences(X, y, window_size=window_size)
        samples, time_steps, n_feats = X_seq.shape
        print(f"[SentimentalAgent] X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")

        # 플랫 CSV
        flattened = []
        for sample_idx in range(samples):
            for time_idx in range(time_steps):
                row = {
                    "sample_id": sample_idx,
                    "time_step": time_idx,
                    "target": float(y_seq[sample_idx, 0]) if time_idx == time_steps - 1 else np.nan,
                }
                for feat_idx, feat_name in enumerate(feature_cols):
                    row[feat_name] = float(X_seq[sample_idx, time_idx, feat_idx])
                flattened.append(row)

        csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
        _save_agent_csv(flattened, csv_path)
        print(f"✅ {ticker} {agent_id} dataset saved to CSV ({samples} samples, {len(feature_cols)} features)")

    # ---------- technical_agent ----------
    elif aid in {"technicalagent", "technical", "테크니컬"}:
        raise NotImplementedError("TechnicalAgent 데이터셋 빌더는 추후 연결 예정입니다.")

    else:
        raise ValueError(f"지원하지 않는 agent_id: {agent_id}")


def load_dataset(ticker: str, agent_id: str, save_dir: str = dir_info["data_dir"]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    위에서 저장한 CSV({ticker}_{agent_id}_dataset.csv)를 다시 시퀀스로 복원.
    """
    csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ["sample_id", "time_step", "target"]]

    unique_samples = df["sample_id"].nunique()
    time_steps = df["time_step"].nunique()
    n_features = len(feature_cols)

    X = np.zeros((unique_samples, time_steps, n_features), dtype=np.float32)
    y = np.zeros((unique_samples, 1), dtype=np.float32)

    for i, sample_id in enumerate(sorted(df["sample_id"].unique())):
        block = df[df["sample_id"] == sample_id].sort_values("time_step")
        X[i] = block[feature_cols].to_numpy(dtype=np.float32)
        # 마지막 타임스텝에만 target 값이 들어가 있으므로 그 값을 사용
        y[i, 0] = block["target"].dropna().iloc[-1] if block["target"].notna().any() else np.nan

    return X, y, feature_cols


def get_latest_close_price(ticker: str, save_dir: str = dir_info["data_dir"]) -> float:
    raw_data_path = os.path.join(save_dir, f"{ticker}_raw_data.csv")
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, index_col=0)
        return float(df["Close"].iloc[-1])
    data = yf.download(ticker, period="1d", interval="1d", progress=False)
    return float(data["Close"].iloc[-1])
