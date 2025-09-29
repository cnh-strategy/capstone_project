# -*- coding: utf-8 -*-
import warnings, os, random, math, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

import yfinance as yf

# ML
from sklearn.preprocessing import MinMaxScaler

# DL
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Tuning
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Progress bar
from tqdm.auto import tqdm

# =========================
# 재현성 & GPU 메모리 그로스
# =========================
SEED = 1234
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# =========================
# Keras용 tqdm 콜백
# =========================
class TqdmProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, epochs:int, description:str="Training"):
        super().__init__()
        self.epochs = epochs
        self.description = description
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.epochs, desc=self.description, unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.update(1)
            if logs:
                self.pbar.set_postfix({
                    k: f"{v:.4f}" for k, v in logs.items()
                    if isinstance(v, (int, float))
                })

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()

# =========================
# 0) 한글→티커 매핑
# =========================
nasdaq100_kor = {
    "엔비디아": "NVDA",
    "마이크로소프트": "MSFT",
    "애플": "AAPL"
}

# =========================
# 1) 유틸/지표/다운로드
# =========================
START_DATE = "2020-01-01"
END_DATE_INCLUSIVE = "2024-12-31"

def _end_to_exclusive(end_inclusive: str) -> str:
    return (pd.to_datetime(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

def _to_naive_utc_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

def _coerce_tz_naive(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj.copy(); out.index = _to_naive_utc_index(out.index); return out
    return obj

def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff(); u = d.clip(lower=0); v = -d.clip(upper=0)
    au = u.rolling(p).mean(); av = v.rolling(p).mean()
    rs = au/av.replace(0,np.nan); return (100 - (100/(1+rs))).fillna(50)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def tema(s: pd.Series, span: int = 10) -> pd.Series:
    e1 = ema(s, span); e2 = ema(e1, span); e3 = ema(e2, span)
    return 3*e1 - 3*e2 + e3

def bollinger(s: pd.Series, p=20, k=2.0):
    m = s.rolling(p).mean(); sd = s.rolling(p).std()
    up = m + k*sd; lo = m - k*sd
    width = (up - lo) / m.replace(0,np.nan)
    pb = (s - lo) / (up - lo)
    return up, lo, width, pb

def atr(h, l, c, n: int=14):
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def obv(c: pd.Series, v: pd.Series):
    direction = np.sign(c.diff().fillna(0.0))
    return (direction * v.fillna(0.0)).cumsum()

def robust_yf_history(ticker: str, start: str, end_exclusive: str) -> pd.DataFrame:
    df = pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(start=start, end=end_exclusive, interval="1d", auto_adjust=True)
    except Exception:
        pass
    if df is None or df.empty:
        try:
            df = yf.download(ticker, start=start, end=end_exclusive, interval="1d", auto_adjust=True, progress=False)
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty:
        df = pd.DataFrame()
    return df

def fetch_ohlcv_fixed_window(ticker: str, start_date: str, end_inclusive: str) -> pd.DataFrame:
    end_exclusive = _end_to_exclusive(end_inclusive)
    df = robust_yf_history(ticker, start_date, end_exclusive)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"yfinance empty for {ticker} in range {start_date}~{end_inclusive}")
    df = df[["Open","High","Low","Close","Volume"]].dropna(how="any")
    return _coerce_tz_naive(df)

def fetch_market_series_fixed_window(start_date: str, end_inclusive: str) -> Optional[pd.DataFrame]:
    end_exclusive = _end_to_exclusive(end_inclusive)
    market_tickers = ["SPY", "QQQ", "^VIX", "^TNX", "DX-Y.NYB"]
    try:
        data = yf.download(market_tickers, start=start_date, end=end_exclusive, interval="1d",
                           auto_adjust=True, progress=False)
    except Exception:
        data = pd.DataFrame()

    if data is None or data.empty or "Close" not in data.columns:
        return None

    close = data["Close"].copy()
    close.rename(columns={"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY"}, inplace=True)

    for col in ["SPY", "QQQ", "VIX", "TNX", "DXY"]:
        if col not in close.columns:
            print(f"Warning: Market ticker {col} not found. Skipping.")
            close[col] = np.nan

    close = _coerce_tz_naive(close.dropna(how="all"))
    if close.empty: return None

    out = pd.DataFrame(index=close.index)
    ret = close[["SPY", "QQQ"]].pct_change()
    out["SPY_ret_1d"] = ret["SPY"]
    out["QQQ_ret_1d"] = ret["QQQ"]
    out["SPY_rv_20"]  = out["SPY_ret_1d"].rolling(20).std()
    out["QQQ_rv_20"]  = out["QQQ_ret_1d"].rolling(20).std()

    macro_cols = ["VIX", "TNX", "DXY"]
    for col in macro_cols:
        out[col] = close[col]
        out[f"{col}_ret_1d"] = close[col].pct_change()
        out[f"{col}_rv_20"]  = out[f"{col}_ret_1d"].rolling(20).std()

    out = out.replace([np.inf,-np.inf], np.nan).ffill()
    return out

def fetch_next_trading_close_after(basis_date_inclusive: str, ticker: str) -> Tuple[str, float]:
    start = (pd.to_datetime(basis_date_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_exclusive = (pd.to_datetime(basis_date_inclusive) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    df = robust_yf_history(ticker, start, end_exclusive)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"실제 종가 조회 실패: {ticker}, {start}~{end_exclusive}")
    df = _coerce_tz_naive(df)[["Close"]].dropna()
    actual_dt = pd.to_datetime(df.index[0]).strftime("%Y-%m-%d")
    actual_close = float(df["Close"].iloc[0])
    return actual_dt, actual_close

def build_features(df: pd.DataFrame, mkt: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    o,h,l,c,v = df["Open"],df["High"],df["Low"],df["Close"],df["Volume"]
    out = pd.DataFrame(index=df.index)
    out["dayofweek"] = df.index.dayofweek
    out["month"] = df.index.month
    out["weekofyear"] = df.index.isocalendar().week.astype(float)
    out["ret_1d"]  = c.pct_change(1)
    out["rsi_14"]  = rsi(c,14)
    out["ma_5"]    = c.rolling(5).mean()
    out["ma_20"]   = c.rolling(20).mean()
    out["ma_50"]   = c.rolling(50).mean()
    out["ma_200"]  = c.rolling(200).mean()
    out["tema_10"] = tema(c,10)
    _,_,bb_w,bbp   = bollinger(c,20,2.0)
    out["bbp"]     = bbp; out["bb_w"] = bb_w
    out["atr_14"]  = atr(h,l,c,14)
    out["hl_range"]= (h-l) / c.replace(0,np.nan)
    out["obv"]     = obv(c,v)
    if isinstance(mkt, pd.DataFrame):
        need = [
            "SPY_ret_1d", "QQQ_ret_1d", "SPY_rv_20", "QQQ_rv_20",
            "VIX", "TNX", "DXY",
            "VIX_ret_1d", "TNX_ret_1d", "DXY_ret_1d",
            "VIX_rv_20", "TNX_rv_20", "DXY_rv_20"
        ]
        available_need = [col for col in need if col in mkt.columns]
        if available_need:
            m_ = mkt.copy().sort_index().replace([np.inf,-np.inf],np.nan).ffill()
            m_aligned = m_.reindex(out.index, method="ffill")
            cov = m_aligned[available_need].notna().mean()
            valid_cols = cov[cov >= 0.3].index.tolist()
            if valid_cols:
                out[valid_cols] = m_aligned[valid_cols]
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    na_ratio = out.isna().mean()
    drop_cols = na_ratio[na_ratio > 0.98].index.tolist()
    if drop_cols: out = out.drop(columns=drop_cols)
    out = out.ffill().dropna(how="all")
    return out

def make_target_logret_next(close: pd.Series, horizon: int=1) -> pd.Series:
    return np.log(close.shift(-horizon) / close)

# =========================
# 2) 프레임 & 시퀀스
# =========================
def build_frames_for_tickers(tickers: List[str], start_date: str, end_inclusive: str):
    mkt = fetch_market_series_fixed_window(start_date, end_inclusive)
    frames = {}
    for t in tqdm(tickers, desc="Build frames", unit="ticker"):
        try:
            raw = fetch_ohlcv_fixed_window(t, start_date, end_inclusive)
            feat = build_features(raw, mkt=mkt)
            y = make_target_logret_next(raw["Close"], 1)
            frame = pd.concat([feat, y.rename("y"), raw["Close"].rename("C")], axis=1)
            frame = frame.replace([np.inf,-np.inf], np.nan).ffill().dropna()
            if len(frame) >= 400:
                frames[t] = frame
            else:
                print(f"[skip] {t}: too few rows ({len(frame)})")
        except Exception as e:
            print(f"[skip] {t}: {e}")
    if not frames:
        raise ValueError("No usable tickers.")
    return frames

def make_sequences_from_frame(frame: pd.DataFrame, lookback: int):
    feat_cols = frame.columns[:-2]  # 마지막 두 개는 y, C
    X_seq, y_seq, C_seq = [], [], []
    for i in range(lookback, len(frame)):
        xwin = frame.iloc[i-lookback:i][feat_cols].values
        y_i  = frame.iloc[i]["y"]
        c_i  = frame.iloc[i]["C"]
        if np.isnan(xwin).any() or np.isnan(y_i) or np.isnan(c_i):
            continue
        X_seq.append(xwin); y_seq.append(y_i); C_seq.append(c_i)
    if len(X_seq)==0:
        return np.empty((0,lookback,len(feat_cols))), np.array([]), np.array([])
    return np.asarray(X_seq), np.asarray(y_seq), np.asarray(C_seq)

def split_train_val_test_by_time(frame: pd.DataFrame, test_ratio=0.2):
    cut = int(len(frame)*(1.0 - test_ratio))
    return frame.iloc[:cut], frame.iloc[cut:]  # trainval, test

def fit_transform_3d(X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray):
    Ntr,W,F = X_tr.shape
    scaler = MinMaxScaler()
    scaler.fit(X_tr.reshape(Ntr*W, F))
    def trns(X):
        if X.size==0: return X
        N = X.shape[0]
        return scaler.transform(X.reshape(N*W, F)).reshape(N, W, F)
    return scaler, trns(X_tr), trns(X_va), trns(X_te)

def attach_onehot(X: np.ndarray, ids: np.ndarray, K: int) -> np.ndarray:
    N,W,F = X.shape; eye = np.eye(K, dtype=np.float32)
    Xout = np.empty((N,W,F+K), dtype=np.float32)
    for i in range(N):
        oh = eye[int(ids[i])]; tile = np.repeat(oh[np.newaxis,:], W, axis=0)
        Xout[i] = np.concatenate([X[i], tile], axis=1)
    return Xout

# =========================
# 3) 모델/튜닝/학습/평가
# =========================
def build_lstm(input_shape, u1=64, u2=64, dr=0.2, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    m = Sequential([
        LSTM(u1, return_sequences=True, input_shape=input_shape),
        Dropout(dr),
        LSTM(u2),
        Dropout(dr),
        Dense(1)
    ])
    m.compile(optimizer=opt, loss="mse")
    return m

def returns_from_logret(y_log: np.ndarray) -> np.ndarray:
    return np.exp(y_log) - 1.0

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def metrics_return_rmse_from_logret(y_log_true, y_log_pred):
    r_true = returns_from_logret(np.asarray(y_log_true))
    r_pred = returns_from_logret(np.asarray(y_log_pred))
    out = {"RMSE_return": rmse(r_pred, r_true)}
    return out, r_true, r_pred

# ---------- 워크포워드: 보조 ----------
def _derive_walkforward_years(trainval_frames: Dict[str, pd.DataFrame], max_folds:int=3) -> List[int]:
    # trainval 구간 전체에서 존재하는 연도 집합
    years = set()
    for t, df in trainval_frames.items():
        ys = pd.DatetimeIndex(df.index).year
        if len(ys)>0:
            years.update(ys.tolist())
    years = sorted(list(years))
    # 첫 해는 최소 학습 기간으로 남겨두고, 뒤쪽에서 최대 max_folds개를 검증 연도로 사용
    if len(years) <= 2:
        return years[-1:]  # 데이터가 적으면 1개라도
    cand = years[1:]      # 첫 해 제외
    return cand[-max_folds:]

def _slice_by_dates(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df

# ---------- 메인 튜닝/학습 ----------
def tune_and_train_global(
    frames: Dict[str, pd.DataFrame],
    val_ratio: float = 0.2,         # (단일 split 백업용, 실제 튜닝엔 미사용)
    test_ratio: float = 0.2,
    n_trials: int = 30,
    wf_max_folds: int = 3           # 워크포워드 폴드 수(최대 연도 수)
):
    tickers = sorted(frames.keys())
    id_map = {t:i for i,t in enumerate(tickers)}
    # 우선 test 홀드아웃 분리
    split = {t: split_train_val_test_by_time(frames[t], test_ratio=test_ratio) for t in tickers}
    trainval_frames = {t: split[t][0] for t in tickers}
    test_frames     = {t: split[t][1] for t in tickers}

    # 워크포워드 검증 연도들(예: 2022, 2023, 2024)
    wf_years = _derive_walkforward_years(trainval_frames, max_folds=wf_max_folds)

    def build_global_seqs_from_tv_with_valyear(lookback: int, val_year: int):
        XtrL, ytrL, CtrL, IDtrL = [], [], [], []
        XvaL, yvaL, CvaL, IDvaL = [], [], [], []
        val_start = pd.Timestamp(f"{val_year}-01-01")
        val_end   = pd.Timestamp(f"{val_year}-12-31")
        for t in tickers:
            tv = trainval_frames[t]
            tr_slice = _slice_by_dates(tv, None, val_start - pd.Timedelta(days=1))
            va_slice = _slice_by_dates(tv, val_start, val_end)
            Xtr,ytr,Ctr = make_sequences_from_frame(tr_slice, lookback)
            Xva,yva,Cva = make_sequences_from_frame(va_slice, lookback)
            if len(Xtr)>0 and len(Xva)>0:
                ids_tr = np.full(len(Xtr), id_map[t]); ids_va = np.full(len(Xva), id_map[t])
                XtrL.append(Xtr); ytrL.append(ytr); CtrL.append(Ctr); IDtrL.append(ids_tr)
                XvaL.append(Xva); yvaL.append(yva); CvaL.append(Cva); IDvaL.append(ids_va)
        cat = lambda lst: np.concatenate(lst, axis=0) if lst else np.empty((0,))
        return cat(XtrL), cat(ytrL), cat(CtrL), cat(IDtrL), cat(XvaL), cat(yvaL), cat(CvaL), cat(IDvaL)

    # ---- Optuna: 다중 워크포워드 폴드 평균 RMSE를 최소화
    def objective(trial):
        tf.keras.backend.clear_session()
        lookback = trial.suggest_int("lookback", 40, 60, step=10)
        u1       = trial.suggest_int("units1", 32, 128, step=32)
        u2       = trial.suggest_int("units2", 32, 128, step=32)
        dr       = trial.suggest_float("dropout", 0.0, 0.4)
        lr       = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch    = trial.suggest_categorical("batch_size", [32, 64])
        epochs   = trial.suggest_int("epochs", 60, 100)

        fold_metrics = []
        for vy in wf_years:
            Xtr,ytr,Ctr,IDtr, Xva,yva,Cva,IDva = build_global_seqs_from_tv_with_valyear(lookback, vy)
            if Xtr.size==0 or Xva.size==0:
                # 이 폴드는 스킵 (데이터 부족). 전체 폴드가 모두 비면 큰 페널티.
                continue

            scaler, Xtr_s, Xva_s, _ = fit_transform_3d(Xtr, Xva, Xva)
            K = len(id_map)
            Xtr_s = attach_onehot(Xtr_s, IDtr, K)
            Xva_s = attach_onehot(Xva_s, IDva, K)

            model = build_lstm(Xtr_s.shape[1:], u1,u2,dr,lr)
            es  = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
            tcb = TqdmProgressBar(epochs, description=f"Trial {trial.number+1} (val={vy})")
            # 폴드 내부 체크포인트는 덮어쓰기(무거움 방지)
            model.fit(Xtr_s, ytr, validation_data=(Xva_s, yva),
                      epochs=epochs, batch_size=batch, shuffle=False,
                      callbacks=[es, tcb], verbose=0)

            y_pred_va = model.predict(Xva_s, verbose=0).ravel()
            met, _, _ = metrics_return_rmse_from_logret(yva, y_pred_va)
            fold_metrics.append(met["RMSE_return"])

        if not fold_metrics:
            return 1e9  # 전 폴드가 비면 큰 페널티
        return float(np.mean(fold_metrics))

    os.makedirs("model_artifacts", exist_ok=True)
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print("Best params:", best)

    # 튜닝 결과 저장
    with open("model_artifacts/best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    # ---- 최종 학습 (Train+Val 전체를 사용하되, 내부 val_split로 monitor=val_loss)
    lookback = best["lookback"]; u1=best["units1"]; u2=best["units2"]
    dr=best["dropout"]; lr=best["lr"]; batch=best["batch_size"]; epochs=best["epochs"]

    # 전체 trainval에서 시퀀스 생성
    def build_global_seqs_from_tv_all(lookback:int):
        Xs, ys, Cs, IDs = [], [], [], []
        for t in tickers:
            tv = trainval_frames[t]
            X, y, C = make_sequences_from_frame(tv, lookback)
            if len(X)==0: continue
            ids = np.full(len(X), id_map[t])
            Xs.append(X); ys.append(y); Cs.append(C); IDs.append(ids)
        cat = lambda lst: np.concatenate(lst, axis=0) if lst else np.empty((0,))
        return cat(Xs), cat(ys), cat(Cs), cat(IDs)

    Xtv, ytv, Ctv, IDtv = build_global_seqs_from_tv_all(lookback)
    XteL, yteL, CteL, IDteL = [], [], [], []
    for t in tickers:
        te = test_frames[t]
        Xte, yte, Cte = make_sequences_from_frame(te, lookback)
        if len(Xte)==0: continue
        XteL.append(Xte); yteL.append(yte); CteL.append(Cte); IDteL.append(np.full(len(Xte), id_map[t]))
    cat = lambda lst: np.concatenate(lst, axis=0) if lst else np.empty((0,))
    Xte, yte, Cte, IDte = cat(XteL), cat(yteL), cat(CteL), cat(IDteL)

    # 스케일링: trainval 기준
    scaler, Xtv_s, _, Xte_s = fit_transform_3d(Xtv, Xtv, Xte)
    K = len(id_map)
    Xtv_s = attach_onehot(Xtv_s, IDtv, K)
    Xte_s = attach_onehot(Xte_s, IDte, K)

    # 최종 모델 학습 (내부 시계열 val_split)
    model = build_lstm(Xtv_s.shape[1:], u1, u2, dr, lr)
    es_final  = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
    tcb_final = TqdmProgressBar(epochs, description="Final training")
    ckpt_final = ModelCheckpoint(
        filepath="model_artifacts/final_best.keras",
        monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=0
    )
    model.fit(
        Xtv_s, ytv,
        validation_split=0.05, shuffle=False,
        epochs=epochs, batch_size=batch,
        callbacks=[es_final, tcb_final, ckpt_final],
        verbose=0
    )

    # Test 성능: 상승하락률 RMSE + 종가 MAPE
    y_pred_te = model.predict(Xte_s, verbose=0).ravel()
    test_met, r_true, r_pred = metrics_return_rmse_from_logret(yte, y_pred_te)

    # 다음 거래일 실제/예측 종가 복원(C_t는 '기준일 종가')
    actual_next_close_te = Cte * np.exp(yte)
    pred_next_close_te   = Cte * np.exp(y_pred_te)
    mape_close = float(np.mean(np.abs((pred_next_close_te - actual_next_close_te) / np.clip(actual_next_close_te, 1e-8, None))) * 100.0)

    test_report_fmt = {
        "상승하락률 RMSE": round(test_met["RMSE_return"], 8),
        "MAPE(다음거래일 종가, %)": round(mape_close, 4)
    }

    artifact = {
        "model": model,
        "scaler": scaler,
        "id_map": id_map,
        "lookback": lookback,
        "tickers": tickers,
        "best_params": best
    }
    return artifact, test_report_fmt

# =========================
# 4) 예측 (한글/티커 입력 → 다음날 종가)
# =========================
def resolve_ticker(user_input: str, kor_map: Dict[str,str]) -> str:
    user_input = user_input.strip().upper()
    for k,v in kor_map.items():
        if user_input in [k.upper(), v.upper()]:
            return v.upper()
    return user_input

def predict_next_close(user_input: str, artifact: dict,
                       start_date: str = START_DATE,
                       end_inclusive: str = END_DATE_INCLUSIVE) -> Dict[str, float]:
    ticker = resolve_ticker(user_input, nasdaq100_kor)
    model, scaler, id_map, lookback = artifact["model"], artifact["scaler"], artifact["id_map"], artifact["lookback"]
    K = len(id_map)

    mkt = fetch_market_series_fixed_window(start_date, end_inclusive)
    raw = fetch_ohlcv_fixed_window(ticker, start_date, end_inclusive)
    feat = build_features(raw, mkt=mkt).replace([np.inf,-np.inf], np.nan).ffill()
    close = raw["Close"]
    if len(feat) < lookback+1:
        raise ValueError(f"Not enough rows for {ticker} to build a window of {lookback}.")
    X_win = feat.iloc[-lookback:].values
    C_t   = float(close.iloc[-1])

    W,F = X_win.shape
    X_win_s = scaler.transform(X_win.reshape(W,F)).reshape(1,W,F)
    if ticker not in id_map:
        print(f"[warn] '{ticker}' not in trained tickers -> using zero one-hot (may degrade accuracy).")
        oh = np.zeros((1,W,K), dtype=np.float32)
    else:
        idx = id_map[ticker]; oh1 = np.eye(K, dtype=np.float32)[idx]
        oh = np.repeat(oh1[np.newaxis, np.newaxis, :], repeats=W, axis=1)
    X_in = np.concatenate([X_win_s, oh], axis=2)

    rhat_log = float(model.predict(X_in, verbose=0).ravel()[0])
    next_close = float(C_t * np.exp(rhat_log))
    return {"ticker": ticker, "today_close": C_t, "pred_next_close": next_close}

# ===================================================================
# 5) 실행: 학습, 평가, 저장 + 예측/실제 비교
# ===================================================================
if __name__ == "__main__":
    TICKERS = sorted(set(nasdaq100_kor.values()))
    TEST_RATIO = 0.2

    print(f"데이터 프레임 구축 시작... 기간: {START_DATE} ~ {END_DATE_INCLUSIVE}")
    frames = build_frames_for_tickers(TICKERS, start_date=START_DATE, end_inclusive=END_DATE_INCLUSIVE)
    print("데이터 프레임 구축 완료. 유효 티커 수:", len(frames))

    os.makedirs("model_artifacts", exist_ok=True)

    print("\n모델 튜닝 및 학습 시작...")
    artifact, test_report = tune_and_train_global(
        frames,
        val_ratio=0.2,
        test_ratio=TEST_RATIO,
        n_trials=30,
        wf_max_folds=3
    )
    print("모델 학습 완료.")

    try:
        print("\n[테스트 리포트]")
        for k,v in test_report.items():
            print(f"{k}: {v}")
    finally:
        print("\n학습된 아티팩트 저장 시작...")
        model = artifact.pop("model")
        model.save("model_artifacts/lstm_model.keras")
        with open("model_artifacts/other_artifacts.pkl", "wb") as f:
            pickle.dump(artifact, f)
        with open("model_artifacts/best_params.json", "w") as f:
            json.dump(artifact.get("best_params", {}), f, indent=2)
        print("아티팩트 저장 완료. (폴더: model_artifacts)")

    # 예측 vs 실제 (휴장일 보정)
    try:
        basis_dt = pd.to_datetime(END_DATE_INCLUSIVE)
        basis_str = basis_dt.strftime("%Y-%m-%d")

        tickers_to_test = ["엔비디아", "애플", "마이크로소프트"]
        abs_pct_errs = []

        print("\n[예측 vs 실제]")
        for name in tickers_to_test:
            pred = predict_next_close(
                name,
                {"model": model, **artifact},
                start_date=START_DATE,
                end_inclusive=END_DATE_INCLUSIVE
            )
            actual_dt, actual_close = fetch_next_trading_close_after(END_DATE_INCLUSIVE, pred["ticker"])
            err_abs = pred["pred_next_close"] - actual_close
            err_pct = (err_abs / actual_close) * 100.0
            abs_pct_errs.append(abs(err_pct))

            print(
                f"기준일={basis_str}  예측대상일(다음 거래일)={actual_dt}  종목={name}({pred['ticker']})  "
                f"기준일종가={pred['today_close']:.2f}  예측종가={pred['pred_next_close']:.2f}  "
                f"실제종가={actual_close:.2f}  오차={err_abs:+.2f} ({err_pct:+.2f}%)"
            )

        if abs_pct_errs:
            mape = np.mean(abs_pct_errs)
            print(f"\n[요약] MAPE(다음 거래일 종가, %): {mape:.2f}%")

    except Exception as e:
        print("\n[예측/실제 비교 실패]", e)
