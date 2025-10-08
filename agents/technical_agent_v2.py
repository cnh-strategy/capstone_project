# ============================================================
# Stage 1 Pre-training (Tech/Fund/Sent 공통 골격) — PyTorch(.pt)
# - Dropout 포함(0.2 적용)
# - 입력 스케일 표준화, 목표는 log-return(다음날)
# - Optuna: 주석 처리
# - .pt 파일에 params(드롭아웃 포함) 동봉 저장
# ============================================================

import warnings, os, random, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

import yfinance as yf

# ML
from sklearn.preprocessing import MinMaxScaler

# DL (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Progress bar
from tqdm.auto import tqdm

# =========================
# 재현성
# =========================
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
END_DATE_INCLUSIVE = "2024-12-31"   # 학습 종료일 (고정)
JAN_START = "2025-01-01"
JAN_END   = "2025-01-31"

OUTDIR = "model_artifacts_filtered"
os.makedirs(OUTDIR, exist_ok=True)

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

# =========================
# 피처 생성
# =========================
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

# ===== Feature 그룹(함수별) 정의 =====
FEATURE_GROUPS = {
    "RSI": ["rsi_14"],
    "MA/EMA/TEMA": ["ma_5","ma_20","ma_50","ma_200","tema_10"],
    "Bollinger": ["bbp","bb_w"],
    "Volatility": ["atr_14","hl_range","SPY_rv_20","QQQ_rv_20","VIX_rv_20","TNX_rv_20","DXY_rv_20"],
    "Returns-Mkt": ["SPY_ret_1d","QQQ_ret_1d","VIX_ret_1d","TNX_ret_1d","DXY_ret_1d"],
    "Level-Mkt": ["VIX","TNX","DXY"],
    "Volume": ["obv"],
    "Calendar": ["dayofweek","month","weekofyear"],
}

# 선택 그룹
SELECTED_GROUPS = ["RSI", "MA/EMA/TEMA", "Returns-Mkt", "Level-Mkt", "Calendar"]
ALLOWED_FEATURES = sorted({f for g in SELECTED_GROUPS for f in FEATURE_GROUPS[g]})

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
            keep_cols = [c for c in feat.columns if c in ALLOWED_FEATURES]
            feat = feat[keep_cols]
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

def align_feature_columns_across_frames(frames: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    all_feat = set()
    for df in frames.values():
        all_feat.update([c for c in df.columns[:-2] if c in ALLOWED_FEATURES])
    feat_cols_ref = sorted(list(all_feat))
    aligned = {}
    for t, frame in frames.items():
        feats = frame.iloc[:, :-2].reindex(columns=feat_cols_ref)
        feats = feats.ffill().fillna(0.0)
        aligned[t] = pd.concat([feats, frame[["y","C"]]], axis=1)
    return aligned, feat_cols_ref

def make_sequences_from_frame(frame: pd.DataFrame, lookback: int):
    feat_cols = frame.columns[:-2]
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
    return frame.iloc[:cut], frame.iloc[cut:]

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
# 3) 모델/학습/평가 (PyTorch)
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim:int, u1:int=64, u2:int=32, dropout:float=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=u1, batch_first=True)
        self.do1   = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=u1, hidden_size=u2, batch_first=True)
        self.do2   = nn.Dropout(dropout)
        self.fc    = nn.Linear(u2, 1)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.do1(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.do2(out2[:, -1, :])
        y = self.fc(out2).squeeze(-1)
        return y

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

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_with_early_stopping(model, train_loader, val_loader, epochs:int, lr:float,
                              patience:int=8, outdir:str=OUTDIR, save_params:Optional[dict]=None):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = np.inf
    best_state = None
    wait = 0

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss_sum += loss.item()*len(xb); tr_n += len(xb)
        train_loss = tr_loss_sum / max(tr_n,1)

        model.eval()
        va_loss_sum, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                va_loss_sum += loss.item()*len(xb); va_n += len(xb)
        val_loss = va_loss_sum / max(va_n,1)
        print(f"Epoch {epoch}/{epochs} - train {train_loss:.6f} - val {val_loss:.6f}")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(
                {"state_dict": best_state, "params": (save_params or {})},
                os.path.join(outdir, "final_best.pt")
            )
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {"state_dict": model.state_dict(), "params": (save_params or {})},
        os.path.join(outdir, "lstm_model.pt")
    )
    return model

# ---------- 워크포워드 유틸(미사용) ----------
def _derive_walkforward_years(trainval_frames: Dict[str, pd.DataFrame], max_folds:int=3) -> List[int]:
    years = set()
    for _, df in trainval_frames.items():
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

# =========================
# 4) 예측
# =========================
def resolve_ticker(user_input: str, kor_map: Dict[str,str]) -> str:
    user_input = user_input.strip().upper()
    for k,v in kor_map.items():
        if user_input in [k.upper(), v.upper()]:
            return v.upper()
    return user_input

def _build_features_for_infer(ticker: str, start_date: str, end_inclusive: str,
                              feat_cols_ref: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    mkt = fetch_market_series_fixed_window(start_date, end_inclusive)
    raw = fetch_ohlcv_fixed_window(ticker, start_date, end_inclusive)
    feat = build_features(raw, mkt=mkt).replace([np.inf,-np.inf], np.nan).ffill()
    feat = feat[[c for c in feat.columns if c in ALLOWED_FEATURES]]
    if feat_cols_ref is not None:
        feat = feat.reindex(columns=feat_cols_ref)
        feat = feat.ffill().fillna(0.0)
    close = raw["Close"]
    return feat, close

def predict_next_close(user_input: str, artifact: dict,
                       start_date: str = START_DATE,
                       end_inclusive: str = END_DATE_INCLUSIVE) -> Dict[str, float]:
    ticker = resolve_ticker(user_input, nasdaq100_kor)
    model, scaler, id_map, lookback = artifact["model"], artifact["scaler"], artifact["id_map"], artifact["lookback"]
    feat_cols_ref = artifact.get("feat_cols")
    K = len(id_map)

    feat, close = _build_features_for_infer(ticker, start_date, end_inclusive, feat_cols_ref=feat_cols_ref)
    if len(feat) < lookback+1:
        raise ValueError(f"Not enough rows for {ticker} to build a window of {lookback}.")
    X_win = feat.iloc[-lookback:].values.astype(np.float32)
    C_t   = float(close.iloc[-1])

    W,F = X_win.shape
    X_win_s = scaler.transform(X_win.reshape(W,F)).reshape(1,W,F).astype(np.float32)
    if ticker not in id_map:
        oh = np.zeros((1,W,K), dtype=np.float32)
    else:
        idx = id_map[ticker]; oh1 = np.eye(K, dtype=np.float32)[idx]
        oh = np.repeat(oh1[np.newaxis, np.newaxis, :], repeats=W, axis=1).astype(np.float32)
    X_in = np.concatenate([X_win_s, oh], axis=2).astype(np.float32)

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X_in).to(DEVICE)
        rhat_log = float(model(x_t).detach().cpu().numpy().ravel()[0])
    next_close = float(C_t * np.exp(rhat_log))
    return {"ticker": ticker, "today_close": C_t, "pred_next_close": next_close}

# =========================
# 5) 2025-01 일별 동화 월간 예측
# =========================
def predict_month_with_daily_assimilation(user_input: str, artifact: dict,
                                          month_start: str = JAN_START, month_end: str = JAN_END,
                                          start_date: str = START_DATE, last_train_day: str = END_DATE_INCLUSIVE):
    ticker = resolve_ticker(user_input, nasdaq100_kor)
    basis = last_train_day
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
            "오차율(%)": ((pred["pred_next_close"] - actual_close) * 100.0 / actual_close)
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
# 6) 실행
# ===================================================================
if __name__ == "__main__":
    FIXED_PARAMS = {
        "lookback": 40,
        "units1": 64,
        "units2": 32,
        "dropout": 0.20,  # ← 드롭아웃 0.2 고정
        "lr": 0.0028218899701628703,
        "batch_size": 128,
        "epochs": 91
    }

    TICKERS = sorted(set(nasdaq100_kor.values()))
    TEST_RATIO = 0.2

    print(f"[프레임 구축] {START_DATE} ~ {END_DATE_INCLUSIVE} (선택 피처만 사용: {SELECTED_GROUPS})")
    raw_frames = build_frames_for_tickers(TICKERS, start_date=START_DATE, end_inclusive=END_DATE_INCLUSIVE)
    frames, feat_cols_ref = align_feature_columns_across_frames(raw_frames)

    split = {t: split_train_val_test_by_time(frames[t], test_ratio=TEST_RATIO) for t in TICKERS}
    trainval_frames = {t: split[t][0] for t in TICKERS}
    test_frames     = {t: split[t][1] for t in TICKERS}

    id_map = {t:i for i,t in enumerate(TICKERS)}
    lookback = FIXED_PARAMS["lookback"]

    def build_global_seqs_from_tv_all(lookback:int):
        Xs, ys, Cs, IDs = [], [], [], []
        for t in TICKERS:
            tv = trainval_frames[t]
            X, y, C = make_sequences_from_frame(tv, lookback)
            if len(X)==0: continue
            ids = np.full(len(X), id_map[t])
            Xs.append(X); ys.append(y); Cs.append(C); IDs.append(ids)
        cat = lambda lst: np.concatenate(lst, axis=0) if lst else np.empty((0,))
        return cat(Xs), cat(ys), cat(Cs), cat(IDs)

    Xtv, ytv, Ctv, IDtv = build_global_seqs_from_tv_all(lookback)
    XteL, yteL, CteL, IDteL = [], [], [], []
    for t in TICKERS:
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

    n_tv = Xtv_s.shape[0]
    val_size = max(int(n_tv * 0.05), 1)
    train_size = n_tv - val_size
    Xtr_s, ytr = Xtv_s[:train_size], ytv[:train_size]
    Xva_s, yva = Xtv_s[train_size:], ytv[train_size:]

    bs = FIXED_PARAMS["batch_size"]
    train_loader = DataLoader(SeqDataset(Xtr_s, ytr), batch_size=bs, shuffle=False,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(SeqDataset(Xva_s, yva), batch_size=bs, shuffle=False,
                              pin_memory=torch.cuda.is_available())

    input_dim = Xtv_s.shape[2]
    model = LSTMRegressor(
        input_dim=input_dim,
        u1=FIXED_PARAMS["units1"],
        u2=FIXED_PARAMS["units2"],
        dropout=float(FIXED_PARAMS["dropout"])
    )

    print("\n[학습 시작 - 고정 하이퍼파라미터]")
    model = train_with_early_stopping(
        model, train_loader, val_loader,
        epochs=int(FIXED_PARAMS["epochs"]),
        lr=float(FIXED_PARAMS["lr"]),
        patience=8,
        outdir=OUTDIR,
        save_params=FIXED_PARAMS   # ← .pt에 params 포함 저장
    )

    model.eval()
    with torch.no_grad():
        y_pred_te = []
        for i in range(0, Xte_s.shape[0], bs):
            xb = torch.from_numpy(Xte_s[i:i+bs]).to(DEVICE)
            yp = model(xb).detach().cpu().numpy().ravel()
            y_pred_te.append(yp)
        y_pred_te = np.concatenate(y_pred_te) if y_pred_te else np.array([])

    def returns_from_logret(y_log: np.ndarray) -> np.ndarray:
        return np.exp(y_log) - 1.0

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def direction_accuracy(y_true_returns: np.ndarray, y_pred_returns: np.ndarray) -> float:
        return float(np.mean(np.sign(y_true_returns) == np.sign(y_pred_returns)))

    test_met = {"RMSE_return": rmse(returns_from_logret(y_pred_te), returns_from_logret(yte))}
    actual_next_close_te = Cte * np.exp(yte)
    pred_next_close_te   = Cte * np.exp(y_pred_te)
    mape_close = float(np.mean(np.abs((pred_next_close_te - actual_next_close_te) /
                                      np.clip(actual_next_close_te, 1e-8, None))) * 100.0)
    dir_acc = direction_accuracy(returns_from_logret(yte), returns_from_logret(y_pred_te)) * 100.0

    test_report_fmt = {
        "상승하락률 RMSE": round(test_met["RMSE_return"], 8),
        "MAPE(다음거래일 종가, %)": round(mape_close, 4),
        "방향 정확도(%)": round(dir_acc, 2)
    }

    best_params = FIXED_PARAMS.copy()
    with open(os.path.join(OUTDIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    with open(os.path.join(OUTDIR, "other_artifacts.pkl"), "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "id_map": id_map,
            "lookback": lookback,
            "tickers": TICKERS,
            "best_params": best_params,
            "feat_cols": frames[TICKERS[0]].columns[:-2].tolist() if TICKERS else []
        }, f)

    artifact = {
        "model": model,
        "scaler": scaler,
        "id_map": id_map,
        "lookback": lookback,
        "tickers": TICKERS,
        "best_params": best_params,
        "feat_cols": frames[TICKERS[0]].columns[:-2].tolist() if TICKERS else []
    }

    print("\n[내부 테스트 리포트 - 선택 피처]")
    for k,v in test_report_fmt.items():
        print(f"{k}: {v}")

    print("\n[2025-01 월간 예측: 재학습 없음, 일별 실제값 동화] (선택 피처)")
    for name in ["엔비디아", "애플", "마이크로소프트"]:
        df_jan, mape_jan, dir_acc_jan = predict_month_with_daily_assimilation(
            name,
            artifact,
            month_start=JAN_START,
            month_end=JAN_END,
            start_date=START_DATE,
            last_train_day=END_DATE_INCLUSIVE
        )
        out_csv = os.path.join(OUTDIR, f"jan2025_{resolve_ticker(name, nasdaq100_kor)}.csv")
        df_jan.to_csv(out_csv, index=False)
        print(f"{name} → 2025-01 MAPE={mape_jan:.2f}% | 방향정확도={dir_acc_jan:.2f}% | CSV={out_csv}")

