# -*- coding: utf-8 -*-
import warnings, os, random, json, pickle
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
# === 고정 하이퍼파라미터 (Optuna 생략 모드) ===
BEST_PARAMS = {
    "lookback": 50,
    "units1": 128,
    "units2": 64,
    "dropout": 0.35533278899551846,
    "lr": 0.002163105979620468,
    "batch_size": 64,
    "epochs": 68
}

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
# 1) 유틸/다운로드/타깃
# =========================
START_DATE = "2020-01-01"
END_DATE_INCLUSIVE = "2024-12-31"   # 학습 종료일 (고정)
JAN_START = "2025-01-01"
JAN_END   = "2025-01-31"

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

def make_target_logret_next(close: pd.Series, horizon: int=1) -> pd.Series:
    return np.log(close.shift(-horizon) / close)

# =========================
# 2) 프레임(종가만) & 시퀀스
# =========================
def build_frames_close_only(tickers: List[str], start_date: str, end_inclusive: str):
    frames = {}
    for t in tqdm(tickers, desc="Build frames (close-only)", unit="ticker"):
        try:
            raw = fetch_ohlcv_fixed_window(t, start_date, end_inclusive)
            y = make_target_logret_next(raw["Close"], 1)
            feats = pd.DataFrame({"feat_close": raw["Close"]}, index=raw.index)  # 단일 특성=종가
            frame = pd.concat([feats, y.rename("y"), raw["Close"].rename("C")], axis=1)
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

def align_feature_columns_across_frames(frames: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    all_feat = set()
    for df in frames.values():
        all_feat.update(df.columns[:-2])  # 마지막 2개(y, C) 제외
    feat_cols_ref = sorted(list(all_feat))
    aligned = {}
    for t, frame in frames.items():
        feats = frame.iloc[:, :-2].reindex(columns=feat_cols_ref)
        feats = feats.ffill().fillna(0.0)
        aligned[t] = pd.concat([feats, frame[["y","C"]]], axis=1)
    return aligned, feat_cols_ref

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
        Z = scaler.transform(X.reshape(N*W, F)).reshape(N, W, F)
        return Z.astype(np.float32)
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

def direction_accuracy(y_true_returns: np.ndarray, y_pred_returns: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true_returns) == np.sign(y_pred_returns)))

# ---------- 워크포워드: 보조 ----------
def _derive_walkforward_years(trainval_frames: Dict[str, pd.DataFrame], max_folds:int=3) -> List[int]:
    years = set()
    for t, df in trainval_frames.items():
        ys = pd.DatetimeIndex(df.index).year
        if len(ys)>0:
            years.update(ys.tolist())
    years = sorted(list(years))
    if len(years) <= 2:
        return years[-1:]
    cand = years[1:]
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
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    n_trials: int = 30,
    wf_max_folds: int = 3
):
    tickers = sorted(frames.keys())
    id_map = {t:i for i,t in enumerate(tickers)}
    split = {t: split_train_val_test_by_time(frames[t], test_ratio=test_ratio) for t in tickers}
    trainval_frames = {t: split[t][0] for t in tickers}
    test_frames     = {t: split[t][1] for t in tickers}

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
                continue

            scaler, Xtr_s, Xva_s, _ = fit_transform_3d(Xtr, Xva, Xva)
            K = len(id_map)
            Xtr_s = attach_onehot(Xtr_s, IDtr, K)
            Xva_s = attach_onehot(Xva_s, IDva, K)

            model = build_lstm(Xtr_s.shape[1:], u1,u2,dr,lr)
            es  = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
            tcb = TqdmProgressBar(epochs, description=f"Trial {trial.number+1} (val={vy})")
            model.fit(Xtr_s, ytr, validation_data=(Xva_s, yva),
                      epochs=epochs, batch_size=batch, shuffle=False,
                      callbacks=[es, tcb], verbose=0)

            y_pred_va = model.predict(Xva_s, verbose=0).ravel()
            met, _, _ = metrics_return_rmse_from_logret(yva, y_pred_va)
            fold_metrics.append(met["RMSE_return"])

        if not fold_metrics:
            return 1e9
        return float(np.mean(fold_metrics))

    os.makedirs("model_artifacts", exist_ok=True)
    
    ### optuna 튜닝 생략 모드 ###
    # study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    # study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    # best = study.best_params
    # print("Best params:", best)

    # 고정 하이퍼 파라미터 사용
    best = BEST_PARAMS
    print("Fixed best params:", best)

    with open("model_artifacts/best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    lookback = best["lookback"]; u1=best["units1"]; u2=best["units2"]
    dr=best["dropout"]; lr=best["lr"]; batch=best["batch_size"]; epochs=best["epochs"]

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

    scaler, Xtv_s, _, Xte_s = fit_transform_3d(Xtv, Xtv, Xte)
    K = len(id_map)
    Xtv_s = attach_onehot(Xtv_s, IDtv, K)
    Xte_s = attach_onehot(Xte_s, IDte, K)

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

    # 내부 holdout 간단 요약
    y_pred_te = model.predict(Xte_s, verbose=0).ravel()
    test_met, r_true, r_pred = metrics_return_rmse_from_logret(yte, y_pred_te)
    actual_next_close_te = Cte * np.exp(yte)
    pred_next_close_te   = Cte * np.exp(y_pred_te)
    mape_close = float(np.mean(np.abs((pred_next_close_te - actual_next_close_te) /
                                      np.clip(actual_next_close_te, 1e-8, None))) * 100.0)
    dir_acc = direction_accuracy(r_true, r_pred) * 100.0

    test_report_fmt = {
        "상승하락률 RMSE": round(test_met["RMSE_return"], 8),
        "MAPE(다음거래일 종가, %)": round(mape_close, 4),
        "방향 정확도(%)": round(dir_acc, 2)
    }

    any_t = next(iter(trainval_frames))
    feat_cols_ref = trainval_frames[any_t].columns[:-2].tolist()

    artifact = {
        "model": model,
        "scaler": scaler,
        "id_map": id_map,
        "lookback": lookback,
        "tickers": tickers,
        "best_params": best,
        "feat_cols": feat_cols_ref
    }
    return artifact, test_report_fmt

# =========================
# 4) 예측 (한글/티커 입력 → 다음날 종가)  *종가 단일 특성*
# =========================
def resolve_ticker(user_input: str, kor_map: Dict[str,str]) -> str:
    user_input = user_input.strip().upper()
    for k,v in kor_map.items():
        if user_input in [k.upper(), v.upper()]:
            return v.upper()
    return user_input

def _build_features_for_infer_close_only(ticker: str, start_date: str, end_inclusive: str,
                                         feat_cols_ref: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    raw = fetch_ohlcv_fixed_window(ticker, start_date, end_inclusive)
    feat = pd.DataFrame({"feat_close": raw["Close"]}, index=raw.index)
    if feat_cols_ref is not None:
        feat = feat.reindex(columns=feat_cols_ref).ffill().fillna(0.0)
    close = raw["Close"]
    return feat, close

def predict_next_close(user_input: str, artifact: dict,
                       start_date: str = START_DATE,
                       end_inclusive: str = END_DATE_INCLUSIVE) -> Dict[str, float]:
    ticker = resolve_ticker(user_input, nasdaq100_kor)
    model, scaler, id_map, lookback = artifact["model"], artifact["scaler"], artifact["id_map"], artifact["lookback"]
    feat_cols_ref = artifact.get("feat_cols")
    K = len(id_map)

    feat, close = _build_features_for_infer_close_only(ticker, start_date, end_inclusive, feat_cols_ref=feat_cols_ref)
    if len(feat) < lookback+1:
        raise ValueError(f"Not enough rows for {ticker} to build a window of {lookback}.")
    X_win = feat.iloc[-lookback:].values
    C_t   = float(close.iloc[-1])

    W,F = X_win.shape
    X_win_s = scaler.transform(X_win.reshape(W,F)).reshape(1,W,F).astype(np.float32)
    if ticker not in id_map:
        oh = np.zeros((1,W,K), dtype=np.float32)
    else:
        idx = id_map[ticker]; oh1 = np.eye(K, dtype=np.float32)[idx]
        oh = np.repeat(oh1[np.newaxis, np.newaxis, :], repeats=W, axis=1).astype(np.float32)
    X_in = np.concatenate([X_win_s, oh], axis=2).astype(np.float32)

    rhat_log = float(model.predict(X_in, verbose=0).ravel()[0])
    next_close = float(C_t * np.exp(rhat_log))
    return {"ticker": ticker, "today_close": C_t, "pred_next_close": next_close}

# =========================
# 5) 2025-01 일별 동화(실제값 입력만 갱신) 월간 예측
# =========================
def predict_month_with_daily_assimilation(user_input: str, artifact: dict,
                                          month_start: str = JAN_START, month_end: str = JAN_END,
                                          start_date: str = START_DATE, last_train_day: str = END_DATE_INCLUSIVE):
    ticker = resolve_ticker(user_input, nasdaq100_kor)
    basis = last_train_day  # 최초 기준일 = 2024-12-31
    rows = []

    while True:
        pred = predict_next_close(user_input, artifact, start_date=start_date, end_inclusive=basis)
        next_dt, actual_close = fetch_next_trading_close_after(basis, ticker)

        next_dt_ts = pd.to_datetime(next_dt)
        if next_dt_ts < pd.to_datetime(month_start):
            basis = next_dt
            continue
        if next_dt_ts > pd.to_datetime(month_end):
            break

        rows.append({
            "기준일": basis,
            "예측대상일": next_dt,
            "ticker": ticker,
            "기준일종가": pred["today_close"],
            "예측종가": pred["pred_next_close"],
            "실제종가": actual_close,
            "오차": pred["pred_next_close"] - actual_close,
            "오차율(%)": ((pred["pred_next_close"] - actual_close) / actual_close) * 100.0
        })

        basis = next_dt
        if pd.to_datetime(basis) >= pd.to_datetime(month_end):
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        prev = df["기준일종가"].values
        direction_pred = np.sign(df["예측종가"].values - prev)
        direction_true = np.sign(df["실제종가"].values - prev)
        df["방향일치"] = (direction_pred == direction_true).astype(int)
        mape = float(np.mean(np.abs((df["예측종가"] - df["실제종가"]) / df["실제종가"])) * 100.0)
        dir_acc = float(df["방향일치"].mean() * 100.0)
    else:
        mape, dir_acc = np.nan, np.nan
    return df, mape, dir_acc

# ===================================================================
# 6) 실행: 학습(Optuna) + 2025-01 월간 예측(동일 모델 사용, 재학습 없음)
# ===================================================================
if __name__ == "__main__":
    TICKERS = sorted(set(nasdaq100_kor.values()))
    TEST_RATIO = 0.2

    print(f"[프레임 구축] {START_DATE} ~ {END_DATE_INCLUSIVE}")
    raw_frames = build_frames_close_only(TICKERS, start_date=START_DATE, end_inclusive=END_DATE_INCLUSIVE)
    frames, feat_cols_ref = align_feature_columns_across_frames(raw_frames)
    os.makedirs("model_artifacts", exist_ok=True)

    print("\n[튜닝/학습] Optuna로 최적 하이퍼파라미터 탐색")
    artifact, test_report = tune_and_train_global(
        frames,
        val_ratio=0.2,
        test_ratio=TEST_RATIO,
        n_trials=30,
        wf_max_folds=3
    )

    if "feat_cols" not in artifact or not artifact["feat_cols"]:
        artifact["feat_cols"] = feat_cols_ref

    print("\n[최적 하이퍼파라미터]")
    print(json.dumps(artifact["best_params"], indent=2, ensure_ascii=False))

    model = artifact["model"]
    model.save("model_artifacts/lstm_model.keras")
    with open("model_artifacts/other_artifacts.pkl", "wb") as f:
        pickle.dump({k:v for k,v in artifact.items() if k!="model"}, f)
    with open("model_artifacts/best_params.json", "w") as f:
        json.dump(artifact["best_params"], f, indent=2)

    print("\n[내부 테스트 리포트]")
    for k,v in test_report.items():
        print(f"{k}: {v}")

    print("\n[2025-01 월간 예측: 재학습 없음, 일별 실제값 동화]")
    for name in ["엔비디아", "애플", "마이크로소프트"]:
        df_jan, mape_jan, dir_acc_jan = predict_month_with_daily_assimilation(
            name,
            artifact,
            month_start=JAN_START,
            month_end=JAN_END,
            start_date=START_DATE,
            last_train_day=END_DATE_INCLUSIVE
        )
        out_csv = f"model_artifacts/base_jan2025_{resolve_ticker(name, nasdaq100_kor)}.csv"
        df_jan.to_csv(out_csv, index=False)
        print(f"{name} → 2025-01 MAPE={mape_jan:.2f}% | 방향정확도={dir_acc_jan:.2f}% | CSV={out_csv}")
