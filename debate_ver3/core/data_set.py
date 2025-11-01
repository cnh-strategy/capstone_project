# debate_ver3/core/data_set.py
# ===============================================================
# 실제 뉴스(EODHD) + 가격(yfinance) → 일일 프레임 → 시퀀스(X, y) 생성
# - 감성값이 API에서 비어 있으면 TextBlob 또는 간단 어휘기반 fallback으로 계산
# - 멀티인덱스/병합 에러 방지, tz-naive 일관화
# - Sentimental/Technical/Fundamental 에이전트별 시퀀스 저장
# - SentimentalAgent 저장 직후 SentimentalAgentV3 별칭 파일도 함께 저장
# ===============================================================

from __future__ import annotations
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from shutil import copyfile

# ---------------------------------------------------------------
# 환경
# ---------------------------------------------------------------
load_dotenv()
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "").strip()

# ---------------------------------------------------------------
# 유틸: 안전한 폴더 생성
# ---------------------------------------------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------
# 유틸: 심볼 정규화
# - EODHD 뉴스: "TSLA.US" 형태 권장
# - yfinance: "TSLA"
# ---------------------------------------------------------------
def _to_eodhd_symbol(ticker: str) -> str:
    t = ticker.upper()
    if "." not in t:
        return f"{t}.US"
    return t

# ---------------------------------------------------------------
# 유틸: 문자열 → 날짜(UTC 날짜만 보존)
# ---------------------------------------------------------------
def _to_date_only(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x.normalize()
    try:
        dt = pd.to_datetime(x, utc=True, errors="coerce")
        if dt is pd.NaT:
            return pd.NaT
        return dt.tz_convert("UTC").tz_localize(None).normalize()
    except Exception:
        try:
            return pd.to_datetime(x).normalize()
        except Exception:
            return pd.NaT

# ---------------------------------------------------------------
# 감성 Fallback: TextBlob 있으면 사용, 없으면 간단 어휘 점수
# ---------------------------------------------------------------
_POS_WORDS = {
    "beat", "beats", "surge", "soar", "soars", "gain", "gains", "bullish", "outperform",
    "strong", "better", "growth", "record", "profit", "profits", "improve", "improves",
    "positive", "up", "launch", "wins", "award", "expand", "expands", "partnership"
}
_NEG_WORDS = {
    "miss", "misses", "plunge", "plunges", "fall", "falls", "bearish", "underperform",
    "weak", "worse", "decline", "loss", "losses", "downgrade", "cuts", "recall",
    "delay", "delays", "problem", "lawsuit", "negative", "down", "halt", "halts"
}

def _sentiment_from_text(text: str) -> float:
    text = text or ""
    text_l = text.lower().strip()
    if not text_l:
        return 0.0
    try:
        from textblob import TextBlob  # type: ignore
        pol = float(TextBlob(text_l).sentiment.polarity)  # [-1, 1]
        return round(pol, 3)
    except Exception:
        pass
    pos = sum(word in text_l for word in _POS_WORDS)
    neg = sum(word in text_l for word in _NEG_WORDS)
    score = 0.0
    if pos + neg > 0:
        score = (pos - neg) / (pos + neg)
    return round(score, 3)

# ---------------------------------------------------------------
# 가격 수집 (yfinance) — 일봉 (초강력 방어형)
# ---------------------------------------------------------------
def fetch_price_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"fetch_price_daily: empty data for {ticker}")
    df = df.copy()

    # MultiIndex 컬럼 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns]

    # Date 컬럼 생성
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df["Date"] = pd.to_datetime(df.index, errors="coerce", utc=True).tz_convert(None)
        except Exception:
            df["Date"] = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    else:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        else:
            df = df.reset_index()
            first_col = df.columns[0]
            if first_col != "Date":
                df.rename(columns={first_col: "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)

    # 인덱스 모호성 제거
    df.reset_index(drop=True, inplace=True)

    # 컬럼 정규화 (Open_TSLA → Open 등)
    std_fields = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    up_tk = ticker.upper()

    def canonical(col: str) -> str:
        c = col.strip()
        c = c.replace("adj close", "Adj Close").replace("Adj close", "Adj Close")
        for sep in ["_", " ", "."]:
            for base in std_fields:
                if c.lower().startswith(base.lower() + sep) and c.split(sep, 1)[1].upper() == up_tk:
                    return base
        for sep in ["_", " ", "."]:
            for base in std_fields:
                if c.lower().endswith(sep + base.lower()) and c.split(sep, 1)[0].upper() == up_tk:
                    return base
        for base in std_fields:
            if c.lower() == base.lower():
                return base
        return c

    normalized = {}
    for col in list(df.columns):
        new = canonical(col)
        if new in std_fields and new not in normalized:
            normalized[new] = col
    rename_map = {normalized[f]: f for f in normalized}
    if rename_map:
        df = df.rename(columns=rename_map)

    # Close 보장
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # 정리
    if "Date" not in df.columns:
        raise RuntimeError(f"[fetch_price_daily] 'Date' column missing. columns={list(df.columns)}")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 필수 컬럼 경고
    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[fetch_price_daily][WARN] missing columns for {ticker}: {missing}")

    return df

# ---------------------------------------------------------------
# 뉴스 수집 (EODHD) — 최근 N일 (감성 fallback 포함)
# ---------------------------------------------------------------
def fetch_news_eodhd(ticker: str, days: int = 30, limit: int = 200) -> pd.DataFrame:
    if not EODHD_API_KEY:
        print("[fetch_news_eodhd] EODHD_API_KEY not set in .env — returning empty news")
        return pd.DataFrame(columns=["date", "title", "content", "polarity"])

    sym = _to_eodhd_symbol(ticker)
    url = f"https://eodhd.com/api/news?s={sym}&limit={limit}&api_token={EODHD_API_KEY}&fmt=json"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            print("[fetch_news_eodhd] status:", r.status_code, "text:", r.text[:200])
            return pd.DataFrame(columns=["date", "title", "content", "polarity"])
        data = r.json()
    except Exception as e:
        print("[fetch_news_eodhd] request failed:", repr(e))
        return pd.DataFrame(columns=["date", "title", "content", "polarity"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for item in (data or []):
        dts = item.get("date") or item.get("published_at") or item.get("publishedAt")
        dt = pd.to_datetime(dts, utc=True, errors="coerce")
        if dt is pd.NaT or dt < cutoff:
            continue
        title = item.get("title") or ""
        content = item.get("content") or ""
        sent = item.get("sentiment") or {}
        pol = None
        if isinstance(sent, dict):
            pol = sent.get("polarity", item.get("polarity", None))
        if pol is None:
            pol = _sentiment_from_text(f"{title}\n{content}")
        rows.append({
            "date": dt, "title": title, "content": content,
            "polarity": float(pol) if pol is not None else 0.0,
        })
    if not rows:
        return pd.DataFrame(columns=["date", "title", "content", "polarity"])

    df = pd.DataFrame(rows)
    df["Date"] = df["date"].apply(_to_date_only)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df[["Date", "title", "content", "polarity"]]

# ---------------------------------------------------------------
# 일단위 뉴스 집계: 평균/분산/건수 + 7일 롤링 건수
# ---------------------------------------------------------------
def aggregate_news_daily(ticker: str, days: int = 60) -> pd.DataFrame:
    df = fetch_news_eodhd(ticker, days=days, limit=400)
    if df is None or df.empty:
        rng = pd.date_range(datetime.now().date() - timedelta(days=days-1), periods=days, freq="B")
        return pd.DataFrame({
            "Date": rng, "sentiment_mean": np.nan, "sentiment_vol": np.nan,
            "news_count_1d": 0, "news_count_7d": 0.0,
        })
    g = df.groupby("Date", as_index=False)["polarity"].agg(["mean", "std", "count"]).reset_index()
    g = g.rename(columns={
        "Date": "Date", "mean": "sentiment_mean",
        "std": "sentiment_vol", "count": "news_count_1d",
    })
    g["Date"] = pd.to_datetime(g["Date"]).dt.normalize()
    g = g.sort_values("Date").reset_index(drop=True)
    full = pd.DataFrame({"Date": pd.bdate_range(g["Date"].min(), g["Date"].max(), freq="B")})
    g = full.merge(g, on="Date", how="left")
    g["news_count_1d"] = g["news_count_1d"].fillna(0).astype(int)
    g["news_count_7d"] = g["news_count_1d"].rolling(window=7, min_periods=1).sum().astype(float)
    return g[["Date", "sentiment_mean", "sentiment_vol", "news_count_1d", "news_count_7d"]]

# ---------------------------------------------------------------
# 가격 + 뉴스 병합 / 파생치 생성
# ---------------------------------------------------------------
def make_price_news_frame(ticker: str) -> pd.DataFrame:
    df_p = fetch_price_daily(ticker)

    # === 가격/거래 피처 생성 ===
    # 수익률
    df_p["returns"] = df_p["Close"].pct_change()
    df_p["lag_ret_1"]  = df_p["returns"].shift(1)
    df_p["lag_ret_5"]  = df_p["returns"].rolling(5).mean().shift(1)
    df_p["lag_ret_20"] = df_p["returns"].rolling(20).mean().shift(1)

    # 추세/변동성
    df_p["trend_7d"]       = df_p["Close"].pct_change(7)
    df_p["rolling_vol_20"] = df_p["returns"].rolling(20).std()

    # ATR(14)
    hl = (df_p["High"] - df_p["Low"]).abs()
    hc = (df_p["High"] - df_p["Close"].shift(1)).abs()
    lc = (df_p["Low"]  - df_p["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_p["atr_14"] = tr.rolling(14).mean()

    # 20일 최고가 돌파 여부
    df_p["breakout_20"] = (df_p["Close"] > df_p["Close"].rolling(20).max().shift(1)).fillna(False)

    # 20일 종가 z-score
    mean20 = df_p["Close"].rolling(20).mean()
    std20  = df_p["Close"].rolling(20).std()
    df_p["zscore_close_20"] = (df_p["Close"] - mean20) / std20

    # 20일 최대 낙폭 (드로다운 근사)
    roll_max20 = df_p["Close"].rolling(20).max()
    df_p["drawdown_20"] = (df_p["Close"] / roll_max20) - 1.0

    # 거래량 지표
    vol_mean20 = df_p["Volume"].rolling(20).mean()
    vol_std20  = df_p["Volume"].rolling(20).std()
    df_p["vol_zscore_20"] = (df_p["Volume"] - vol_mean20) / vol_std20
    df_p["volume_spike"]  = (df_p["vol_zscore_20"] >= 2.0).fillna(False)

    # turnover_rate (대체치: 거래량/20일 평균거래량)
    df_p["turnover_rate"] = (df_p["Volume"] / (vol_mean20.replace(0, np.nan))).fillna(0.0)

    # === 뉴스 집계/병합 ===
    df_n = aggregate_news_daily(ticker, days=365)
    for d in (df_p, df_n):
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce", utc=True).dt.tz_localize(None)

    if getattr(df_p.index, "nlevels", 1) > 1: df_p = df_p.reset_index()
    if getattr(df_n.index, "nlevels", 1) > 1: df_n = df_n.reset_index()

    df = pd.merge(df_p, df_n, on="Date", how="left").sort_values("Date").reset_index(drop=True)

    # target
    if "returns" not in df.columns:
        df["returns"] = df["Close"].pct_change()
    df["target_return"] = df["returns"].shift(-1)

    # 뉴스 결측 보정
    for c in ["sentiment_mean", "sentiment_vol"]:
        if c not in df.columns: df[c] = np.nan
        df[c] = df[c].ffill().fillna(0.0)
    for c in ["news_count_1d", "news_count_7d"]:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    return df

# ---------------------------------------------------------------
# 시퀀스 변환
# ---------------------------------------------------------------
def _to_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df2 = df.dropna(subset=[target_col]).reset_index(drop=True)
    X_list, y_list = [], []
    for i in range(len(df2) - window):
        seg = df2.iloc[i:i+window]
        X_list.append(seg[feature_cols].to_numpy(dtype=np.float32))
        y_list.append([float(df2.iloc[i+window][target_col])])
    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, window, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty((0, 1), dtype=np.float32)
    return X, y, feature_cols

# ---------------------------------------------------------------
# 각 에이전트용 데이터 저장 (+ V3 별칭 저장)
# ---------------------------------------------------------------
def save_datasets_for_agents(ticker: str, save_dir: str, df: pd.DataFrame):
    _ensure_dir(save_dir)

    # 1) SentimentalAgent
    senti_feats = ["returns", "sentiment_mean", "sentiment_vol", "Close", "Volume", "Open", "High", "Low"]
    f_exist = [c for c in senti_feats if c in df.columns]
    missing = set(senti_feats) - set(f_exist)
    if missing:
        for c in missing:
            df[c] = 0.0
        f_exist = senti_feats

    Xs, ys, cols_s = _to_sequences(df, f_exist, "target_return", window=40)
    np.savez_compressed(os.path.join(save_dir, f"{ticker}_SentimentalAgent_dataset.npz"),
                        X=Xs, y=ys, feature_cols=np.array(cols_s, dtype=object), window_size=40)
    pd.DataFrame({
        "feature_cols": [json.dumps(cols_s)],
        "window_size": [40],
        "X_shape": [list(Xs.shape)],
        "y_shape": [list(ys.shape)],
    }).to_csv(os.path.join(save_dir, f"{ticker}_SentimentalAgent_dataset.csv"), index=False)
    print(f"[SentimentalAgent] X_seq: {Xs.shape}, y_seq: {ys.shape}")
    print(f"✅ {ticker} SentimentalAgent dataset saved to CSV/NPZ ({Xs.shape[0]} samples, {Xs.shape[2]} features)")

    # 🔁 SentimentalAgentV3 별칭 파일도 함께 저장
    npz_main = os.path.join(save_dir, f"{ticker}_SentimentalAgent_dataset.npz")
    csv_main = os.path.join(save_dir, f"{ticker}_SentimentalAgent_dataset.csv")
    npz_v3 = os.path.join(save_dir, f"{ticker}_SentimentalAgentV3_dataset.npz")
    csv_v3 = os.path.join(save_dir, f"{ticker}_SentimentalAgentV3_dataset.csv")
    try:
        copyfile(npz_main, npz_v3)
        copyfile(csv_main, csv_v3)
        print(f"[SentimentalAgentV3] aliased dataset saved → {npz_v3}, {csv_v3}")
    except Exception as e:
        print(f"[SentimentalAgentV3][WARN] alias copy failed: {e}")

    # 2) TechnicalAgent (예시 6피처)
    tech_feats = ["returns", "Close", "Volume", "Open", "High", "Low"]
    for c in tech_feats:
        if c not in df.columns:
            df[c] = 0.0
    Xt, yt, cols_t = _to_sequences(df, tech_feats, "target_return", window=14)
    np.savez_compressed(os.path.join(save_dir, f"{ticker}_TechnicalAgent_dataset.npz"),
                        X=Xt, y=yt, feature_cols=np.array(cols_t, dtype=object), window_size=14)
    pd.DataFrame({
        "feature_cols": [json.dumps(cols_t)],
        "window_size": [14],
        "X_shape": [list(Xt.shape)],
        "y_shape": [list(yt.shape)],
    }).to_csv(os.path.join(save_dir, f"{ticker}_TechnicalAgent_dataset.csv"), index=False)
    print(f"[TechnicalAgent] X_seq: {Xt.shape}, y_seq: {yt.shape}")
    print(f"✅ {ticker} TechnicalAgent dataset saved to CSV/NPZ ({Xt.shape[0]} samples, {Xt.shape[2]} features)")

    # 3) FundamentalAgent (동일 6피처 예시)
    fund_feats = ["returns", "Close", "Volume", "Open", "High", "Low"]
    for c in fund_feats:
        if c not in df.columns:
            df[c] = 0.0
    Xf, yf, cols_f = _to_sequences(df, fund_feats, "target_return", window=14)
    np.savez_compressed(os.path.join(save_dir, f"{ticker}_FundamentalAgent_dataset.npz"),
                        X=Xf, y=yf, feature_cols=np.array(cols_f, dtype=object), window_size=14)
    pd.DataFrame({
        "feature_cols": [json.dumps(cols_f)],
        "window_size": [14],
        "X_shape": [list(Xf.shape)],
        "y_shape": [list(yf.shape)],
    }).to_csv(os.path.join(save_dir, f"{ticker}_FundamentalAgent_dataset.csv"), index=False)
    print(f"[FundamentalAgent] X_seq: {Xf.shape}, y_seq: {yf.shape}")
    print(f"✅ {ticker} FundamentalAgent dataset saved to CSV/NPZ ({Xf.shape[0]} samples, {Xf.shape[2]} features)")

# ---------------------------------------------------------------
# 공개 API: 데이터셋 생성
# ---------------------------------------------------------------
def build_dataset(ticker: str, save_dir: str = "data/processed"):
    df = make_price_news_frame(ticker)
    save_datasets_for_agents(ticker, save_dir, df)

# ---------------------------------------------------------------
# 공개 API: 로드 (V3 → 기존 이름 폴백 지원)
# ---------------------------------------------------------------
def load_dataset(
    ticker: str,
    agent_id: str,
    save_dir: str = "data/processed"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    def _bases(tk, aid):
        bases = [os.path.join(save_dir, f"{tk}_{aid}_dataset")]
        alias = {
            "SentimentalAgentV3": "SentimentalAgent",
            # 필요시 추가: "TechnicalAgentV3": "TechnicalAgent", ...
        }
        if aid in alias:
            bases.append(os.path.join(save_dir, f"{tk}_{alias[aid]}_dataset"))
        return bases

    bases = _bases(ticker, agent_id)

    # NPZ 우선
    for base in bases:
        npz_path = base + ".npz"
        if os.path.exists(npz_path):
            z = np.load(npz_path, allow_pickle=True)
            X = z["X"]; y = z["y"]
            cols = list(z["feature_cols"])
            if len(cols) and isinstance(cols[0], (bytes, bytearray)):
                cols = [c.decode("utf-8") for c in cols]
            return X, y, cols

    # CSV 메타 → 원천에서 재생성
    for base in bases:
        csv_path = base + ".csv"
        if os.path.exists(csv_path):
            meta = pd.read_csv(csv_path)
            cols_json = meta["feature_cols"].iloc[0]
            try:
                cols = json.loads(cols_json)
            except Exception:
                cols = [c.strip().strip('"').strip("'") for c in cols_json.strip("[]").split(",")]
            df = make_price_news_frame(ticker)
            default_win = 40 if (agent_id.startswith("SentimentalAgent")) else 14
            window = int(meta["window_size"].iloc[0]) if "window_size" in meta.columns else default_win
            X, y, cols_back = _to_sequences(df, cols, "target_return", window)
            return X, y, cols_back

    # 없으면 생성 후 재시도
    build_dataset(ticker, save_dir=save_dir)
    return load_dataset(ticker, agent_id=agent_id, save_dir=save_dir)
