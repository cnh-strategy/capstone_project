# core/data_set.py
# ===============================================================
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
from .technical_classes import technical_data_set as _techds
from .sentimental_classes.news import merge_price_with_news_features  
from dataclasses import dataclass
from typing import Optional, Dict, Any

SENTI_FEATURE_COLS = [
    "return_1d",
    "hl_range",
    "Volume",
    "news_count_1d",
    "news_count_7d",
    "sentiment_mean_1d",
    "sentiment_mean_7d",
    "sentiment_vol_7d",
]

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

@dataclass
class StockData:
    """
    에이전트 입력 원천 데이터(필요 시 자유 확장)
    - sentimental: 심리/커뮤니티/뉴스 스냅샷
    - fundamental: 재무/밸류에이션 요약
    - technical  : 가격/지표 스냅샷
    - last_price : 최신 종가
    - currency   : 통화 단위 (예: 'USD')
    """
    ticker: str
    sentimental: Optional[Dict[str, Any]] = None
    fundamental: Optional[Dict[str, Any]] = None
    technical: Optional[Dict[str, Any]] = None
    last_price: Optional[float] = None
    currency: str = "USD"


@dataclass
class Target:
    """
    에이전트가 예측한 결과를 담는 객체.
    - current_price: 현재(기준) 종가
    - next_close   : 예측한 다음날 종가
    - change_ratio : (next_close / current_price - 1)
    - uncertainty  : 예측 불확실도(표준편차 등), 없으면 None
    - confidence   : 신뢰도(0~1), 없으면 None
    - agent_id     : 이 타겟을 만든 에이전트 ID
    """
    ticker: str
    current_price: float
    next_close: float
    change_ratio: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    agent_id: Optional[str] = None


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


# Sentimental
def _fetch_ticker_data_for_sentimental(ticker: str, period: Optional[str], interval: Optional[str]) -> pd.DataFrame:
    period = period or "5y"
    interval = interval or "1d"

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    df.dropna(inplace=True)

    # 멀티인덱스 방어
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 기본 기술 지표
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)

    # USD/KRW, NASDAQ, VIX
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

    # === 가격 + 뉴스 피처 병합 ===
    try:
        df_reset = df.reset_index()
        if "Date" not in df_reset.columns:
            idx_name = df_reset.columns[0]
            df_reset = df_reset.rename(columns={idx_name: "Date"})

        merged = merge_price_with_news_features(
            df_price=df_reset,
            ticker=ticker,
            window=7,
            show_tail=False,
        )

        merged = merged.sort_values("Date").set_index("Date")

        # 뉴스 기반 감성 피처 정의
        merged["sentiment_mean"] = merged["sentiment_mean_7d"]
        merged["sentiment_vol"] = (
            merged["sentiment_mean_1d"]
            .rolling(window=7, min_periods=1)
            .std()
            .fillna(0.0)
        )

        df = merged

    except Exception as e:
        print(f"⚠️ 뉴스 피처 병합 실패, placeholder 감성 사용: {e}")
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
    debate_agent.py에서 agent_id를 넘겨주면, 여기서 분기 처리
    - agent_id == 'MacroSentiAgent' / '매크로' / 'macro'
    - agent_id == 'SentimentalAgent' / '센티멘탈' / 'sentimental'
    - agent_id == 'TechnicalAgent' / '테크니컬' / 'technical' (추후)
    """
    os.makedirs(save_dir, exist_ok=True)
    # 공통 RAW는 테크니컬 전용 fetch로 통일
    raw = _techds.fetch_ticker_data(
        ticker,
        period=period or "5y",
        interval=interval or "1d",
    )
    raw.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True)

    # Agent별 데이터셋을 CSV로 저장
    for aid, _ in agents_info.items():
        # ---------- macro_agent ----------
        if aid in {"MacroSentiAgent","macrosentiagent", "macro", "매크로"}:
            if not _HAS_MACRO or macro_dataset is None:
                raise ImportError(
                    "macro_dataset 모듈을 찾을 수 없습니다. core/macro_classes 확인 필요 "
                    f"details={_MACRO_IMPORT_ERROR}"
                )
            macro_dataset(ticker_name=ticker)
            print(f"✅ {ticker} MacroSentiAgent dataset saved (macro_dataset 호출 via {_MACRO_SRC})")
            return

        # ---------- sentimental_agent ----------
        elif aid in {"SentimentalAgent", "sentimentalagent", "sentimental", "센티멘탈"}:

            # 1) 이 에이전트 설정 불러오기
            senti_cfg = agents_info.get("SentimentalAgent", {})

            # 2) period / interval
            senti_period = period or senti_cfg.get("period", "5y")
            senti_interval = interval or senti_cfg.get("interval", "1d")

            # 3) 가격 + 뉴스 감성 포함된 df 생성
            df = _fetch_ticker_data_for_sentimental(
                ticker,
                period=senti_period,
                interval=senti_interval,
            )

            df.to_csv(
                os.path.join(save_dir, f"{ticker}_raw_data.csv"),
                index=True,
                encoding="utf-8"
            )

            # ★ 여기! FEATURE_COLS 대신 SENTI_FEATURE_COLS 사용
            required_cols = list(SENTI_FEATURE_COLS)

            df_feat = df.copy()

            # return_1d / hl_range / Volume 보정은 그대로 두고
            if "return_1d" not in df_feat.columns:
                df_feat["return_1d"] = df_feat["Close"].pct_change().fillna(0.0)

            if "hl_range" not in df_feat.columns:
                rng = (df_feat["High"] - df_feat["Low"]) / df_feat["Close"].replace(0, np.nan)
                df_feat["hl_range"] = rng.fillna(0.0)

            if "Volume" not in df_feat.columns and "volume" in df_feat.columns:
                df_feat["Volume"] = df_feat["volume"].fillna(0.0)

            # ⚠ 뉴스 피처 0으로 채우던 for-loop는 제거 (이미 제거했다고 봄)

            missing = [c for c in required_cols if c not in df_feat.columns]
            if missing:
                raise ValueError(
                    f"[SentimentalAgent build_dataset] FEATURE_COLS 중 누락: {missing}\n"
                    f"현재 df_feat.columns = {df_feat.columns.tolist()}"
                )

            returns = df_feat["Close"].pct_change().shift(-1)
            valid_mask = ~returns.isna()
            y = returns[valid_mask].to_numpy().reshape(-1, 1)
            X = df_feat.loc[valid_mask, required_cols]

            window_size = int(senti_cfg.get("window_size", 40))
            X_seq, y_seq = create_sequences(X, y, window_size=window_size)
            samples, time_steps, n_feats = X_seq.shape
            print(f"[SentimentalAgent] X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")

            # 8) 플랫 CSV 저장
            flattened = []
            for sample_idx in range(samples):
                for time_idx in range(time_steps):
                    row = {
                        "sample_id": sample_idx,
                        "time_step": time_idx,
                        "target": float(y_seq[sample_idx, 0]) if time_idx == time_steps - 1 else np.nan,
                    }
                    for feat_idx, feat_name in enumerate(required_cols):
                        row[feat_name] = float(X_seq[sample_idx, time_idx, feat_idx])
                    flattened.append(row)

            csv_path = os.path.join(save_dir, f"{ticker}_{aid}_dataset.csv")
            _save_agent_csv(flattened, csv_path)

            print(
                f"✅ {ticker} {aid} dataset saved to CSV "
                f"({samples} samples, {len(required_cols)} features)"
            )
            return

        # ---------- technical_agent ----------
        elif aid in {"TechnicalAgent","technicalagent", "technical", "테크니컬"}:
            _techds.build_dataset(
                ticker=ticker,
                save_dir=save_dir,
                period=period or agents_info["TechnicalAgent"].get("period", "5y"),
                interval=interval or agents_info["TechnicalAgent"].get("interval", "1d"),
            )
            print(f"✅ {ticker} TechnicalAgent dataset saved via technical_data_set")
            return

        else:
            raise ValueError(f"지원하지 않는 agent_id: {agent_id}")


# -----------sentimental ---------------
def build_sentimental_dataset(
    ticker: str,
    days: int = 365 * 5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    SentimentalAgent 전용 pretrain 데이터셋 생성

    1) yfinance 로 OHLCV(price) 다운로드
    2) FinBERT 점수가 포함된 뉴스 DF(df_news)를 로드
    3) merge_price_with_news_features 로 가격 + 뉴스 피처 결합
        - news_count_1d, news_count_7d
        - sentiment_mean_1d, sentiment_mean_7d
        - sentiment_vol_7d
    4) return_1d, hl_range 계산
    5) FEATURE_COLS 순서대로 시계열 윈도우 (WINDOW_SIZE) 생성
    6) npz 저장: data/datasets/{ticker}_SentimentalAgent.npz
    """
    # 0) 날짜 범위 설정
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)

    # 1) 가격 데이터 다운로드 (OHLCV)
    df_price = yf.download(ticker, start=start, end=end)
    if df_price.empty:
        raise ValueError(f"[build_sentimental_dataset] No price data for {ticker}")

    # 멀티인덱스일 경우 (종종 yfinance가 Ticker 레벨을 붙임) Close 등만 추출
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price = df_price.xs(ticker, axis=1, level=0)

    # 2) 뉴스 데이터 로드

    news_dir = os.path.join("data", "raw", "news")
    news_path = os.path.join(news_dir, f"{ticker}_news_finbert.csv")

    if not os.path.exists(news_path):
        raise FileNotFoundError(
            f"[build_sentimental_dataset] news file not found: {news_path}\n"
            "FinBERT 감성 점수가 포함된 뉴스 CSV 경로를 확인해 주세요."
        )

    df_news = pd.read_csv(news_path)

    # 3) 가격 + 뉴스 피처 머지
    df_merged = merge_price_with_news_features(df_price, df_news, tz="Asia/Seoul")

    # 4) 가격 기반 추가 피처 계산
    #     - return_1d   : 종가 기준 일일 수익률
    #     - hl_range    : (고가-저가) / 종가
    df_merged = df_merged.copy()
    df_merged["return_1d"] = df_merged["Close"].pct_change().fillna(0.0)

    # 분모 0 방지
    close_safe = df_merged["Close"].replace(0, np.nan)
    df_merged["hl_range"] = ((df_merged["High"] - df_merged["Low"]) / close_safe).fillna(0.0)

    # 5) FEATURE_COLS만 선택
    missing = [c for c in FEATURE_COLS if c not in df_merged.columns]
    if missing:
        raise ValueError(
            f"[build_sentimental_dataset] missing feature columns in df_merged: {missing}"
        )

    df_feat = df_merged[FEATURE_COLS].dropna()

    # 6) 시계열 윈도우 생성 (WINDOW_SIZE, next-day return_1d 예측)
    values = df_feat.values  # shape: (T, F)
    F = values.shape[1]
    T_total = values.shape[0]
    win = int(WINDOW_SIZE)

    X_list = []
    y_list = []

    # 타깃은 다음 날의 return_1d (FEATURE_COLS[0] 이 return_1d 라고 가정)
    target_idx = FEATURE_COLS.index("return_1d")

    for t in range(win, T_total - 1):
        X_window = values[t - win : t, :]
        y_next = values[t + 1, target_idx]  # 다음 날 수익률
        X_list.append(X_window)
        y_list.append(y_next)

    if not X_list:
        raise ValueError(
            f"[build_sentimental_dataset] not enough data to build windows. "
            f"Got T_total={T_total}, WINDOW_SIZE={win}"
        )

    X = np.stack(X_list, axis=0)               # (N, win, F)
    y = np.array(y_list, dtype=float).reshape(-1, 1)  # (N, 1)

    # 7) 저장 (npz + 미리보기 CSV)
    # -----------------------------
    os.makedirs(os.path.join("data", "datasets"), exist_ok=True)
    npz_path = os.path.join("data", "datasets", f"{ticker}_SentimentalAgent.npz")
    np.savez(
        npz_path,
        X=X,
        y=y,
        feature_cols=np.array(FEATURE_COLS),
    )

    # optional: 미리보기용 CSV (마지막 윈도우만)
    preview_dir = os.path.join("data", "preview")
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, f"{ticker}_SentimentalAgent_preview.csv")
    preview_df = pd.DataFrame(
        X[-1],
        columns=FEATURE_COLS,
    )
    preview_df.to_csv(preview_path, index=False)

    print(
        f"[SentimentalAgent] X_seq: {X.shape}, y_seq: {y.shape}\n"
        f"✅ {ticker} SentimentalAgent dataset saved to NPZ: {npz_path}"
    )

    return X, y, FEATURE_COLS


def load_dataset(
    ticker: str,
    agent_id: str,
    save_dir: str = dir_info["data_dir"],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    위에서 저장한 CSV({ticker}_{agent_id}_dataset.csv)를 다시 시퀀스로 복원
    - 숫자형 컬럼만 사용하도록 안전 가드 추가(날짜/문자열 혼입 방지)
    - TechnicalAgent / SentimentalAgent는 전용 분기 처리
    """
    norm = str(agent_id).lower()

    # 1) 테크니컬은 전용 로더로 위임
    if norm in {"technicalagent", "technical", "테크니컬"}:
        X, y, feature_cols, _dates = _techds.load_dataset(
            ticker=ticker,
            agent_id="TechnicalAgent",
            save_dir=save_dir,
        )
        return X, y, feature_cols

    # 공통 CSV 경로
    csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 공통 후보 피처: 플랫 CSV 기준 기본 제외 컬럼
    candidate_cols = [c for c in df.columns if c not in ["sample_id", "time_step", "target"]]

    unique_samples = df["sample_id"].nunique()
    time_steps = df["time_step"].nunique()

    # ---------- ① SentimentalAgent 전용 ----------
    if norm in {"sentimentalagent", "sentimental", "센티멘탈"}:
        # 1) 우선 SENTI_FEATURE_COLS 가 모두 있는지 확인
        has_all_senti_cols = all(c in candidate_cols for c in SENTI_FEATURE_COLS)

        if has_all_senti_cols:
            # ✅ 새 구조(SENTI_FEATURE_COLS)로 로드
            feature_cols: List[str] = list(SENTI_FEATURE_COLS)
            n_features = len(feature_cols)

            X = np.zeros((unique_samples, time_steps, n_features), dtype=np.float32)
            y = np.zeros((unique_samples, 1), dtype=np.float32)

            for i, sample_id in enumerate(sorted(df["sample_id"].unique())):
                block = df[df["sample_id"] == sample_id].sort_values("time_step")

                block_numeric = block[feature_cols].apply(pd.to_numeric, errors="coerce")
                if block_numeric.isna().any().any():
                    block_numeric = block_numeric.fillna(method="ffill").fillna(0.0)

                X[i] = block_numeric.to_numpy(dtype=np.float32)

                y_val = block["target"].dropna()
                y[i, 0] = float(y_val.iloc[-1]) if not y_val.empty else np.nan

            return X, y, feature_cols

        else:
            # ⚠ 새 피처가 없으면 경고 찍고 "기존 방식"으로 폴백
            print(
                "[warn] SENTI_FEATURE_COLS not found in SentimentalAgent dataset. "
                "기존 numeric feature 기반 로더로 폴백합니다.\n"
                f"  - 누락 컬럼: {[c for c in SENTI_FEATURE_COLS if c not in candidate_cols]}\n"
                f"  - candidate_cols: {candidate_cols}"
            )
            # 아래에서 공통 numeric 로직으로 이어짐

    # ---------- ② 그 외 에이전트: 기존 공통 로직 ----------
    # 최초 블록(가장 작은 sample_id)에서 숫자형 컬럼만 확정
    first_id = sorted(df["sample_id"].unique())[0]
    first_block = df[df["sample_id"] == first_id].sort_values("time_step")[candidate_cols]

    numeric_feature_cols: List[str] = []
    for c in candidate_cols:
        s = first_block[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_feature_cols.append(c)

    dropped = [c for c in candidate_cols if c not in numeric_feature_cols]
    if dropped:
        print(f"[warn] Non-numeric features dropped in load_dataset(): {dropped}")

    feature_cols = numeric_feature_cols
    n_features = len(feature_cols)

    if n_features == 0:
        raise ValueError(
            "No numeric feature columns found after filtering. "
            "Check your dataset builder."
        )

    X = np.zeros((unique_samples, time_steps, n_features), dtype=np.float32)
    y = np.zeros((unique_samples, 1), dtype=np.float32)

    for i, sample_id in enumerate(sorted(df["sample_id"].unique())):
        block = df[df["sample_id"] == sample_id].sort_values("time_step")

        block_numeric = block[feature_cols].apply(pd.to_numeric, errors="coerce")
        if block_numeric.isna().any().any():
            block_numeric = block_numeric.fillna(method="ffill").fillna(0.0)

        X[i] = block_numeric.to_numpy(dtype=np.float32)

        y_val = block["target"].dropna()
        y[i, 0] = float(y_val.iloc[-1]) if not y_val.empty else np.nan

    return X, y, feature_cols


def get_latest_close_price(ticker: str, save_dir: str = dir_info["data_dir"]) -> float:
    raw_data_path = os.path.join(save_dir, f"{ticker}_raw_data.csv")
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, index_col=0)
        return float(df["Close"].iloc[-1])
    data = yf.download(ticker, period="1d", interval="1d", progress=False)
    return float(data["Close"].iloc[-1])


def build_targets(close: pd.Series) -> pd.Series:
    # P_{t+1} / P_t - 1
    ret = close.shift(-1) / close - 1.0
    # 마지막 행은 타깃 없음 → 제거
    return ret.iloc[:-1]

def build_features(df: pd.DataFrame, window: int):
    # df.index는 날짜 정렬 가정, 마지막 행 제외하고 X, y 정렬 맞추기
    y = build_targets(df["Close"])
    X = df.iloc[:-1]          # y와 길이 정합
    return X, y
