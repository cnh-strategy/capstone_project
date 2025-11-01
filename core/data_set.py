# core/data_set.py
# ===============================================================
# 멀티에이전트용 데이터셋 빌더 및 로더
#  - Sentimental / Technical / Macro* 데이터 구성
#  - CSV(미리보기) + NPZ(학습용)
#  - 타깃: "다음날 수익률" (Close_{t+1}/Close_t - 1)
# ===============================================================

from __future__ import annotations
import os
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---- optional macro import (지연) ----
try:
    from core.macro_classes.macro_funcs import macro_dataset
    _HAS_MACRO = True
except Exception:
    macro_dataset = None
    _HAS_MACRO = False

# ---- config fallback ----
try:
    from config.agents import agents_info, dir_info
except Exception:
    agents_info = {
        "SentimentalAgent": {"window_sizes": [40], "period": "2y", "interval": "1d"}
    }
    dir_info = {
        "data_root": "data",
        "processed_dir": os.path.join("data", "processed"),
        "raw_dir": os.path.join("data", "raw"),
        "preview_dir": os.path.join("data", "preview"),
    }

# ===============================================================
# utils
# ===============================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip("_") for tup in df.columns]
    return df

def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return (s - m) / sd

def _standardize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df).copy()
    close_candidates = [c for c in df.columns if c.lower().startswith("close")]
    adj_candidates = [c for c in df.columns if c.lower().startswith("adj close") or c.lower().startswith("adjclose")]

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        elif close_candidates:
            df = df.rename(columns={close_candidates[0]: "Close"})
        elif adj_candidates:
            df = df.rename(columns={adj_candidates[0]: "Close"})

    keep_names = []
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        if base in df.columns:
            keep_names.append(base)
            continue
        cand = [c for c in df.columns if c.lower().startswith(base.lower())]
        if cand:
            df = df.rename(columns={cand[0]: base})
            keep_names.append(base)

    if "Close" not in keep_names:
        raise ValueError("Price dataframe must contain 'Close' (or an equivalent).")

    df = df[keep_names]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def _add_target_next_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        else:
            cand = [c for c in df.columns if c.lower().startswith("close")]
            if cand:
                df = df.rename(columns={cand[0]: "Close"})
            if "Close" not in df.columns:
                raise KeyError("Close 컬럼이 없습니다.")
    df = df.assign(target_return_next=(df["Close"].shift(-1) / df["Close"] - 1.0))
    if "target_return_next" not in df.columns:
        df["target_return_next"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df = df[df["target_return_next"].notna()]
    return df

def _build_basic_features(df_price: pd.DataFrame) -> pd.DataFrame:
    df = df_price.copy()
    df["returns"] = df["Close"].pct_change()
    df["rolling_vol_20"] = df["returns"].rolling(20).std(ddof=0)
    df["zscore_close_20"] = _zscore(df["Close"], 20)
    if "Volume" in df.columns:
        df["zscore_volume_20"] = _zscore(df["Volume"].fillna(0), 20)
        v_mean = df["Volume"].rolling(20).mean()
        df["turnover_rate"] = (df["Volume"] / (v_mean.replace(0, np.nan))).clip(upper=10).fillna(0)
        v_mean2 = df["Volume"].rolling(20).mean()
        df["volume_spike"] = (df["Volume"] >= (v_mean2 * 1.5)).astype(int)
    rolling_high = df["Close"].rolling(20).max()
    df["breakout_20"] = (df["Close"] >= rolling_high).astype(int)
    return df

def _merge_macro_if_available(df_feat: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not _HAS_MACRO or macro_dataset is None:
        return df_feat
    try:
        df_macro = macro_dataset(ticker)
        if not isinstance(df_macro.index, pd.DatetimeIndex):
            df_macro.index = pd.to_datetime(df_macro.index)
        df_macro = _flatten_columns(df_macro).sort_index()
        df_macro = df_macro.add_prefix("macro_")
        return df_feat.join(df_macro, how="left")
    except Exception:
        return df_feat

def _sanity_clean(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    existing = [c for c in required_cols if c in df.columns]
    if not existing:
        return df.dropna()
    return df.dropna(subset=existing)

def _to_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert target_col in df.columns, f"target_col '{target_col}' not in columns"
    feature_cols = [c for c in feature_cols if c in df.columns]
    values = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)

    X_list, y_list = [], []
    end = len(df) - window_size + 1 - 1  # shift(-1) 반영
    for i in range(max(0, end)):
        X_list.append(values[i:i + window_size])
        y_list.append(target[i + window_size - 1])

    if not X_list:
        raise ValueError(f"Not enough rows: len(df)={len(df)}, window={window_size}")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y

# ===============================================================
# build / load
# ===============================================================

def _resolve_dirs(save_dir: Optional[str]) -> Dict[str, str]:
    data_root_default = "data"
    processed_default = os.path.join("data", "processed")
    raw_default = os.path.join("data", "raw")
    preview_default = os.path.join("data", "preview")
    try:
        from config.agents import dir_info as _di
        data_root_default = _di.get("data_root", data_root_default)
        processed_default = _di.get("processed_dir", processed_default)
        raw_default = _di.get("raw_dir", raw_default)
        preview_default = _di.get("preview_dir", preview_default)
    except Exception:
        pass

    if not save_dir:
        processed, raw, preview = processed_default, raw_default, preview_default
        root = data_root_default
    else:
        base = os.path.basename(os.path.normpath(save_dir)).lower()
        if base in {"processed", "raw", "preview"}:
            parent = os.path.dirname(os.path.normpath(save_dir)) or "."
            if base == "processed":
                processed = save_dir; raw = os.path.join(parent, "raw"); preview = os.path.join(parent, "preview")
            elif base == "raw":
                raw = save_dir; processed = os.path.join(parent, "processed"); preview = os.path.join(parent, "preview")
            else:
                preview = save_dir; processed = os.path.join(parent, "processed"); raw = os.path.join(parent, "raw")
            root = parent
        else:
            processed = os.path.join(save_dir, "processed")
            raw = os.path.join(save_dir, "raw")
            preview = os.path.join(save_dir, "preview")
            root = save_dir

    for p in (processed, raw, preview):
        os.makedirs(p, exist_ok=True)
    return {"root": root, "processed": processed, "raw": raw, "preview": preview}

def build_dataset(
    ticker: str,
    save_dir: str,
    agent_id: str = "SentimentalAgent",
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> None:
    dirs = _resolve_dirs(save_dir)
    agent_cfg = agents_info.get(agent_id, {})
    window_sizes = agent_cfg.get("window_sizes", [40])
    period = period or agent_cfg.get("period", "2y")
    interval = interval or agent_cfg.get("interval", "1d")

    df_price = yf.download(ticker, period=period, interval=interval, progress=False)
    df_price = _standardize_price_df(df_price)

    try:
        df_price.to_csv(os.path.join(dirs["raw"], f"{ticker}_{agent_id}_price.csv"))
    except Exception:
        pass

    df_feat = _build_basic_features(df_price)
    df_feat = _merge_macro_if_available(df_feat, ticker)
    df_feat = _add_target_next_return(df_feat)

    feature_cols: List[str] = [c for c in df_feat.columns if c != "target_return_next"]
    df_feat = _sanity_clean(df_feat, required_cols=feature_cols + ["target_return_next"])

    w = int(window_sizes[0])
    X, y = _to_sequences(df_feat, feature_cols, target_col="target_return_next", window_size=w)

    npz_path = os.path.join(dirs["processed"], f"{ticker}_{agent_id}.npz")
    meta_path = os.path.join(dirs["processed"], f"{ticker}_{agent_id}.json")
    preview_path = os.path.join(dirs["preview"], f"{ticker}_{agent_id}_preview.csv")

    np.savez_compressed(npz_path, X=X, y=y)
    meta = {
        "ticker": ticker, "agent_id": agent_id,
        "feature_cols": feature_cols, "window_size": w,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[2]),
        "period": period, "interval": interval,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    try:
        df_feat.tail(200).to_csv(preview_path, index=True)
    except Exception:
        pass

    print(f"[{agent_id}] X_seq: {X.shape}, y_seq: {y.shape}")
    print(f"✅ {ticker} {agent_id} dataset saved to NPZ/CSV (window={w}, features={len(feature_cols)})")

def load_dataset(
    ticker: str,
    agent_id: str,
    save_dir: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dirs = _resolve_dirs(save_dir)
    npz_path = os.path.join(dirs["processed"], f"{ticker}_{agent_id}.npz")
    meta_path = os.path.join(dirs["processed"], f"{ticker}_{agent_id}.json")

    if not os.path.exists(npz_path):
        print(f"[load_dataset] {npz_path} 없음 → 새로 생성 중...")
        build_dataset(ticker=ticker, save_dir=save_dir, agent_id=agent_id)

    with np.load(npz_path) as data:
        X = data["X"]; y = data["y"]

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feature_cols = meta.get("feature_cols") or [f"f{i}" for i in range(X.shape[2])]
    else:
        feature_cols = [f"f{i}" for i in range(X.shape[2])]

    return X, y, feature_cols
