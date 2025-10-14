# ============================================================
# Stage 1 — TECH(12) + FUND(7) → [GRU ⊕ MLP] Gated (2-layer gate)
# Fixed HP (dropout=0.12) + Jan-2025 Assimilation
# ============================================================

import warnings, os, random, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# 환경설정
# =========================
nasdaq100_kor = {"엔비디아":"NVDA","마이크로소프트":"MSFT","애플":"AAPL"}
START_DATE = "2020-01-01"
END_DATE_INCLUSIVE = "2024-12-31"
JAN_START = "2025-01-01"
JAN_END   = "2025-01-31"
OUTDIR = "model_artifacts_filtered"
os.makedirs(OUTDIR, exist_ok=True)

# 고정 하이퍼파라미터
FIXED = {
    "lookback": 40,
    "units1": 32,
    "units2": 16,
    "mlp_hidden": 32,
    "lr": 3.556365139054937e-4,
    "batch_size": 64,
    "epochs": 90,
    "patience": 8,
    "dropout": 0.12,   # 최종 확정
    "gate_hidden": 8
}
# 게이트 정규화 설정
GATE_TARGET = 0.45
GATE_LAMBDA = 0.01
GATE_CLIP = (0.05, 0.95)

# =========================
# Features
# =========================
FEATURES_TECH = [
    "vol_20d","obv","ret_3d","r2",
    "weekofyear","vol_ma_20","vol_chg",
    "bbp","ma_200","macd","adx_14","mom_10"
]
FEATURES_FUND = [
    "log_marketcap","earnings_yield","book_to_market",
    "dividend_yield","net_margin","short_signal","avg_vol_60d"
]
FEATURES = FEATURES_TECH + FEATURES_FUND

# ============================================================
# I/O helpers
# ============================================================
def _end_to_exclusive(end_inclusive: str) -> str:
    return (pd.to_datetime(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

def _to_naive_utc_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None: idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

def _coerce_tz_naive(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj.copy(); out.index = _to_naive_utc_index(out.index); return out
    return obj

def robust_yf_history(ticker: str, start: str, end_exclusive: str) -> pd.DataFrame:
    df = pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(start=start, end=end_exclusive, interval="1d", auto_adjust=True)
    except Exception: pass
    if df is None or df.empty:
        try:
            df = yf.download(ticker, start=start, end=end_exclusive, interval="1d",
                             auto_adjust=True, progress=False)
        except Exception: df = pd.DataFrame()
    return df if df is not None else pd.DataFrame()

def fetch_ohlcv_fixed_window(ticker: str, start_date: str, end_inclusive: str) -> pd.DataFrame:
    end_exclusive = _end_to_exclusive(end_inclusive)
    df = robust_yf_history(ticker, start_date, end_exclusive)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"yfinance empty for {ticker} in range {start_date}~{end_inclusive}")
    df = df[["Open","High","Low","Close","Volume"]].dropna(how="any")
    return _coerce_tz_naive(df)

# ============================================================
# Indicators — TECH
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def bollinger_pb(s: pd.Series, p=20, k=2.0):
    m = s.rolling(p).mean(); sd = s.rolling(p).std()
    up = m + k*sd; lo = m - k*sd
    pb = (s - lo) / (up - lo)
    return pb

def momentum(s: pd.Series, n: int) -> pd.Series:
    return s - s.shift(n)

def true_range(h, l, c):
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr

def adx(h, l, c, n: int = 14):
    up = h.diff(); dn = -l.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)
    atr_n = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan)
    adx_ = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx_

def obv(c: pd.Series, v: pd.Series):
    direction = np.sign(c.diff().fillna(0.0))
    return (direction * v.fillna(0.0)).cumsum()

# ============================================================
# Fundamentals — helpers
# ============================================================
def _safe_info_val(tkr, keys: List[str]):
    for k in keys:
        try:
            fi = getattr(tkr, "fast_info", None)
            if fi is not None and hasattr(fi, k):
                val = getattr(fi, k)
                if val is not None: return val
        except Exception: pass
        try:
            info = tkr.info
            if isinstance(info, dict) and k in info and info[k] is not None:
                return info[k]
        except Exception: pass
    return None

def _pick_row(df: pd.DataFrame, patterns: List[str]):
    if df is None or df.empty: return None
    idx = [str(i).lower() for i in df.index]
    for p in patterns:
        p = p.lower()
        for i,name in enumerate(idx):
            if p in name: return df.index[i]
    return None

def fundamentals_panel_yf(ticker: str, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Daily fundamentals aligned to daily_df.index with D+1 shift."""
    tkr = yf.Ticker(ticker)
    idx = pd.DatetimeIndex(daily_df.index).sort_values()
    close = daily_df["Close"].astype(float)
    volume = daily_df["Volume"].astype(float)

    shares_series = None
    try:
        sh = tkr.get_shares_full(
            start=idx.min().strftime("%Y-%m-%d"),
            end=(idx.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        if sh is not None and not sh.empty:
            sh = sh.sort_index()
            col = [c for c in sh.columns if "Shares" in str(c)]
            if col:
                shares_series = sh[col[0]].reindex(idx, method="ffill").astype(float)
    except Exception:
        shares_series = None
    if shares_series is None or shares_series.isna().all():
        so = _safe_info_val(tkr, ["shares_outstanding","sharesOutstanding"])
        so = float(so) if so not in (None, np.nan) else 0.0
        shares_series = pd.Series(so, index=idx, dtype=float)

    mktcap = (shares_series * close).replace([np.inf, -np.inf], np.nan)

    earnings_yield = pd.Series(np.nan, index=idx, dtype=float)
    net_margin     = pd.Series(np.nan, index=idx, dtype=float)
    qfin = None
    try: qfin = tkr.quarterly_financials
    except Exception: qfin = None
    if qfin is not None and not qfin.empty:
        qf = qfin.copy()
        ni_row  = _pick_row(qf, ["net income"])
        rev_row = _pick_row(qf, ["total revenue", "revenue"])
        if ni_row is not None and rev_row is not None:
            qf = qf.loc[[ni_row, rev_row]].T.sort_index()
            qf.columns = ["net_income","revenue"]
            qf.index = pd.to_datetime(qf.index)
            qf = qf.dropna(how="all")
            qf = qf[~qf.index.duplicated(keep="last")]
            ni_ttm  = qf["net_income"].rolling(4, min_periods=1).sum()
            rev_ttm = qf["revenue"].rolling(4, min_periods=1).sum()
            sh_q = shares_series.reindex(qf.index, method="ffill").replace(0, np.nan)
            eps_ttm_q = (ni_ttm / sh_q).replace([np.inf, -np.inf], np.nan)
            eps_ttm_daily = eps_ttm_q.reindex(idx, method="ffill").shift(1)
            earnings_yield = (eps_ttm_daily / close).replace([np.inf, -np.inf], np.nan)
            net_margin_q = (ni_ttm / rev_ttm).replace([np.inf, -np.inf], np.nan)
            net_margin   = net_margin_q.reindex(idx, method="ffill").shift(1)
    if earnings_yield.isna().all():
        pe = _safe_info_val(tkr, ["trailingPE", "forwardPE"])
        pe = float(pe) if pe not in (None, 0, np.nan) else np.nan
        ey = (1.0 / pe) if pe and np.isfinite(pe) and pe != 0 else np.nan
        earnings_yield = pd.Series(ey, index=idx, dtype=float).shift(1)
    if net_margin.isna().all():
        net_margin = pd.Series(0.0, index=idx, dtype=float)

    book_to_market = pd.Series(np.nan, index=idx, dtype=float)
    qbs = None
    try: qbs = tkr.quarterly_balance_sheet
    except Exception: qbs = None
    if qbs is not None and not qbs.empty:
        qbs = qbs.copy()
        eq_row = _pick_row(qbs, ["total stockholder", "total shareholders", "total equity"])
        if eq_row is not None:
            eq = qbs.loc[eq_row].T
            eq.index = pd.to_datetime(eq.index)
            eq = eq.sort_index()
            eq_daily = eq.reindex(idx, method="ffill").shift(1)
            bm = (eq_daily.squeeze() / mktcap.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            book_to_market = bm
    if book_to_market.isna().all():
        pb = _safe_info_val(tkr, ["priceToBook", "price_to_book"])
        pb = float(pb) if pb not in (None, 0, np.nan) else np.nan
        bm = (1.0 / pb) if pb and np.isfinite(pb) and pb != 0 else 0.0
        book_to_market = pd.Series(bm, index=idx, dtype=float).shift(1).fillna(0.0)

    try: dv = tkr.dividends
    except Exception: dv = None
    if dv is None or dv.empty:
        dividend_yield = pd.Series(0.0, index=idx, dtype=float)
    else:
        dv = dv.copy()
        dv.index = _to_naive_utc_index(dv.index)
        daily_div = dv.reindex(idx, fill_value=0.0)
        trailing_div = daily_div.rolling(252, min_periods=1).sum()
        dividend_yield = (trailing_div / close.replace(0, np.nan)).shift(1).fillna(0.0)
        dividend_yield = dividend_yield.replace([np.inf, -np.inf], 0.0)

    shares_short     = _safe_info_val(tkr, ["sharesShort","shares_short"])
    float_shares     = _safe_info_val(tkr, ["floatShares","float_shares"])
    short_ratio_info = _safe_info_val(tkr, ["shortRatio","short_ratio"])
    short_signal_val = np.nan
    try:
        if shares_short not in (None, np.nan) and float_shares not in (None, 0, np.nan):
            short_signal_val = float(shares_short) / float(float_shares)
        elif short_ratio_info not in (None, np.nan):
            short_signal_val = float(short_ratio_info)
    except Exception:
        short_signal_val = np.nan
    short_signal = pd.Series(short_signal_val, index=idx, dtype=float).fillna(0.0)

    avg_vol_60d = volume.rolling(60, min_periods=1).mean().astype(float)

    fund = pd.DataFrame({
        "log_marketcap": np.log((shares_series * close).replace(0,np.nan)),
        "earnings_yield": earnings_yield,
        "book_to_market": book_to_market,
        "dividend_yield": dividend_yield,
        "net_margin": net_margin,
        "short_signal": short_signal,
        "avg_vol_60d": avg_vol_60d
    }, index=idx)

    for col in ["earnings_yield", "book_to_market", "net_margin", "dividend_yield"]:
        if col in fund.columns:
            fund[col] = fund[col].rolling(63, min_periods=1).mean()

    def _winsor(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 30: return s
        q1, q99 = s.quantile(0.01), s.quantile(0.99)
        return s.clip(lower=q1, upper=q99)

    fund = fund.replace([np.inf,-np.inf], np.nan).ffill()
    for col in ["earnings_yield","book_to_market","net_margin"]:
        if fund[col].isna().mean()==1.0: fund[col]=0.0
    fund["dividend_yield"]=fund["dividend_yield"].fillna(0.0)
    fund["short_signal"]=fund["short_signal"].fillna(0.0)
    fund["avg_vol_60d"]=fund["avg_vol_60d"].fillna(method="ffill").fillna(0.0)
    fund["log_marketcap"]=fund["log_marketcap"].fillna(method="ffill")
    fund = fund.apply(_winsor).fillna(0.0)
    return fund

# ============================================================
# Feature builder (TECH+FUND)
# ============================================================
def build_features(df_price: pd.DataFrame, ticker: str) -> pd.DataFrame:
    o,h,l,c,v = df_price["Open"],df_price["High"],df_price["Low"],df_price["Close"],df_price["Volume"]
    out = pd.DataFrame(index=df_price.index)
    out["weekofyear"] = df_price.index.isocalendar().week.astype(float)
    out["r2"] = np.log(c.shift(1) / c.shift(2))
    out["ret_3d"] = c.pct_change(3)
    out["mom_10"] = momentum(c, 10)
    out["ma_200"] = c.rolling(200).mean()
    ema12, ema26 = ema(c,12), ema(c,26)
    out["macd"] = ema12 - ema26
    out["bbp"] = bollinger_pb(c,20,2.0)
    out["adx_14"] = adx(h,l,c,14)
    out["obv"] = obv(c,v)
    out["vol_ma_20"] = v.rolling(20).mean()
    out["vol_chg"] = v.pct_change(1)
    ret_1d = c.pct_change(1)
    out["vol_20d"] = ret_1d.rolling(20).std()

    fund = fundamentals_panel_yf(ticker, df_price)
    out = out.join(fund, how="left")
    out = out[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    out = out.ffill().dropna()
    return out

def make_target_logret_next(close: pd.Series, horizon: int=1) -> pd.Series:
    return np.log(close.shift(-horizon) / close)

# ============================================================
# Frame/Sequence (dual outputs)
# ============================================================
def build_frames_for_tickers(tickers: List[str], start_date: str, end_inclusive: str):
    frames = {}
    for t in tqdm(tickers, desc="Build frames", unit="ticker"):
        try:
            raw = fetch_ohlcv_fixed_window(t, start_date, end_inclusive)
            feat = build_features(raw, t)
            y = make_target_logret_next(raw["Close"], 1)
            frame = pd.concat([feat, y.rename("y"), raw["Close"].rename("C")], axis=1)
            frame = frame.replace([np.inf,-np.inf], np.nan).ffill().dropna()
            if len(frame) >= 400: frames[t] = frame
            else: print(f"[skip] {t}: too few rows ({len(frame)})")
        except Exception as e:
            print(f"[skip] {t}: {e}")
    if not frames: raise ValueError("No usable tickers.")
    return frames

def align_feature_columns_across_frames(frames: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    feat_cols_ref = FEATURES[:]
    aligned = {}
    for t, frame in frames.items():
        feats = frame.iloc[:, :-2].reindex(columns=feat_cols_ref)
        feats = feats.ffill().fillna(0.0)
        aligned[t] = pd.concat([feats, frame[["y","C"]]], axis=1)
    return aligned, feat_cols_ref

def make_sequences_from_frame_dual(frame: pd.DataFrame, lookback: int):
    Xtech, Xfund, y_seq, C_seq = [], [], [], []
    for i in range(lookback, len(frame)):
        xwin_tech = frame.iloc[i-lookback:i][FEATURES_TECH].values
        xvec_fund = frame.iloc[i-1][FEATURES_FUND].values
        y_i  = frame.iloc[i]["y"]; c_i = frame.iloc[i]["C"]
        if np.isnan(xwin_tech).any() or np.isnan(xvec_fund).any() or np.isnan(y_i) or np.isnan(c_i):
            continue
        Xtech.append(xwin_tech); Xfund.append(xvec_fund); y_seq.append(y_i); C_seq.append(c_i)
    if len(Xtech)==0:
        return (np.empty((0, lookback, len(FEATURES_TECH))),
                np.empty((0, len(FEATURES_FUND))),
                np.array([]), np.array([]))
    return np.asarray(Xtech), np.asarray(Xfund), np.asarray(y_seq), np.asarray(C_seq)

def split_train_val_test_by_time(frame: pd.DataFrame, test_ratio=0.2):
    cut = int(len(frame)*(1.0 - test_ratio))
    return frame.iloc[:cut], frame.iloc[cut:]

# ============================================================
# Scaling / one-hot
# ============================================================
def fit_transform_dual(Xt_tr, Xf_tr, Xt_va, Xf_va, Xt_te, Xf_te):
    Ntr, W, Ft = Xt_tr.shape
    scaler_tech = MinMaxScaler()
    scaler_tech.fit(Xt_tr.reshape(Ntr*W, Ft))
    def trns_tech(X):
        if X.size==0: return X
        N = X.shape[0]; W = X.shape[1]; F = X.shape[2]
        Z = scaler_tech.transform(X.reshape(N*W, F)).reshape(N, W, F)
        return Z.astype(np.float32)

    scaler_fund = MinMaxScaler()
    scaler_fund.fit(Xf_tr)
    def trns_fund(X):
        if X.size==0: return X
        return scaler_fund.transform(X).astype(np.float32)

    return scaler_tech, scaler_fund, trns_tech(Xt_tr), trns_fund(Xf_tr), trns_tech(Xt_va), trns_fund(Xf_va), trns_tech(Xt_te), trns_fund(Xf_te)

def attach_onehot_seq(Xseq: np.ndarray, ids: np.ndarray, K: int) -> np.ndarray:
    N,W,F = Xseq.shape; eye = np.eye(K, dtype=np.float32)
    Xout = np.empty((N,W,F+K), dtype=np.float32)
    for i in range(N):
        oh = eye[int(ids[i])]; tile = np.repeat(oh[np.newaxis,:], W, axis=0)
        Xout[i] = np.concatenate([Xseq[i], tile], axis=1)
    return Xout

def attach_onehot_vec(Xvec: np.ndarray, ids: np.ndarray, K: int) -> np.ndarray:
    N,F = Xvec.shape; eye = np.eye(K, dtype=np.float32)
    Xout = np.empty((N,F+K), dtype=np.float32)
    for i in range(N):
        oh = eye[int(ids[i])]
        Xout[i] = np.concatenate([Xvec[i], oh], axis=0)
    return Xout

# ============================================================
# Model: Dual-Branch GRU ⊕ MLP with 2-layer gating
# ============================================================
class DualBranchGRUGatedRegressor(nn.Module):
    def __init__(self, tech_input_dim:int, fund_input_dim:int,
                 u1:int=64, u2:int=32, mlp_hidden:int=32, dropout:float=0.2,
                 gate_hidden:int=8):
        super().__init__()
        # TECH branch
        self.gru1 = nn.GRU(input_size=tech_input_dim, hidden_size=u1, batch_first=True)
        self.do1  = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=u1, hidden_size=u2, batch_first=True)
        self.do2  = nn.Dropout(dropout)
        self.fc_gru = nn.Linear(u2, 1)
        # FUND branch (MLP)
        self.fc_f1 = nn.Linear(fund_input_dim, mlp_hidden)
        self.act   = nn.ReLU()
        self.do_f  = nn.Dropout(dropout)
        self.val_head  = nn.Linear(mlp_hidden, 1)
        # 2-layer gate MLP
        self.gate_head = nn.Sequential(
            nn.Linear(mlp_hidden, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )

    def forward(self, x_seq, x_fund):
        # TECH
        h1,_ = self.gru1(x_seq); h1 = self.do1(h1)
        h2,_ = self.gru2(h1);    h2 = self.do2(h2[:, -1, :])
        y_gru = self.fc_gru(h2).squeeze(-1)
        # FUND
        h = self.act(self.fc_f1(x_fund)); h = self.do_f(h)
        y_mlp = self.val_head(h).squeeze(-1)
        gate  = torch.sigmoid(self.gate_head(h)).squeeze(-1)
        # Mix
        y = (1.0 - gate) * y_gru + gate * y_mlp
        return y, gate, y_gru, y_mlp

# Utilities
def returns_from_logret(y_log: np.ndarray) -> np.ndarray:
    return np.exp(y_log) - 1.0

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def direction_accuracy(y_true_returns: np.ndarray, y_pred_returns: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true_returns) == np.sign(y_pred_returns)))

class DualSeqDataset(Dataset):
    def __init__(self, Xseq: np.ndarray, Xfund: np.ndarray, y: np.ndarray):
        self.Xseq = torch.from_numpy(Xseq.astype(np.float32))
        self.Xfund = torch.from_numpy(Xfund.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.Xseq.shape[0]
    def __getitem__(self, idx): return self.Xseq[idx], self.Xfund[idx], self.y[idx]

def train_with_early_stopping(model, train_loader, val_loader, epochs:int, lr:float,
                              patience:int=8, outdir:str=OUTDIR, tag:str="model",
                              save_params:Optional[dict]=None,
                              gate_target:float=GATE_TARGET, gate_lambda:float=GATE_LAMBDA,
                              gate_clip:Tuple[float,float]=GATE_CLIP):
    model = model.to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = np.inf
    best_state = None
    wait = 0

    for epoch in range(1, epochs+1):
        # Train
        model.train()
        tr_loss_sum = 0.0; tr_n = 0; gate_means = []
        for xs, xf, yb in train_loader:
            xs, xf, yb = xs.to(DEVICE), xf.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred, gate, _, _ = model(xs, xf)
            gate_reg = torch.mean((torch.clamp(gate, gate_clip[0], gate_clip[1]) - gate_target) ** 2)
            loss = criterion(pred, yb) + gate_lambda * gate_reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss_sum += loss.item() * len(xs); tr_n += len(xs)
            gate_means.append(gate.mean().item())
        train_loss = tr_loss_sum / max(tr_n, 1)
        gate_mean_train = np.mean(gate_means) if gate_means else np.nan

        # Valid
        model.eval()
        va_loss_sum = 0.0; va_n = 0; gate_means_val = []
        with torch.no_grad():
            for xs, xf, yb in val_loader:
                xs, xf, yb = xs.to(DEVICE), xf.to(DEVICE), yb.to(DEVICE)
                pred, gate, _, _ = model(xs, xf)
                loss = criterion(pred, yb)
                va_loss_sum += loss.item() * len(xs); va_n += len(xs)
                gate_means_val.append(gate.mean().item())
        val_loss = va_loss_sum / max(va_n, 1)
        gate_mean_val = np.mean(gate_means_val) if gate_means_val else np.nan

        print(f"[{tag}] {epoch:03d}/{epochs} - train {train_loss:.6f} - val {val_loss:.6f} - gate(train,val)=({gate_mean_train:.3f},{gate_mean_val:.3f})")

        # Early stopping + .pt 저장(best)
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save({"state_dict": best_state, "params": (save_params or {})},
                       os.path.join(outdir, f"{tag}_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # 마지막 상태 .pt 저장(final)
    torch.save({"state_dict": model.state_dict(), "params": (save_params or {})},
               os.path.join(outdir, f"{tag}_final.pt"))
    return model

# ============================================================
# 평가
# ============================================================
def eval_seq_model(model, Xseq_s, Xfund_s, y_log, C, tag):
    model.eval(); bs = 1024; pred=[]; gates=[]
    with torch.no_grad():
        for i in range(0, Xseq_s.shape[0], bs):
            xs = torch.from_numpy(Xseq_s[i:i+bs]).to(DEVICE)
            xf = torch.from_numpy(Xfund_s[i:i+bs]).to(DEVICE)
            yp, g, _, _ = model(xs, xf)
            pred.append(yp.detach().cpu().numpy().ravel())
            gates.append(g.detach().cpu().numpy().ravel())
    y_pred = np.concatenate(pred) if pred else np.array([])
    r_true = returns_from_logret(y_log); r_pred = returns_from_logret(y_pred)
    rmse_ret = rmse(r_pred, r_true)
    actual_next_close = C * np.exp(y_log)
    pred_next_close   = C * np.exp(y_pred)
    mape_close = float(np.mean(np.abs((pred_next_close - actual_next_close) /
                                      np.clip(actual_next_close, 1e-8, None))) * 100.0)
    dir_acc = direction_accuracy(r_true, r_pred) * 100.0
    avg_gate = float(np.mean(np.concatenate(gates))) if gates else np.nan
    rep = {"RMSE_return": round(rmse_ret,8), "MAPE_close_%": round(mape_close,4),
           "DirAcc_%": round(dir_acc,2), "AvgGate": round(avg_gate,3)}
    print(f"\n[{tag}] {rep}")
    return rep, y_pred

# ============================================================
# 예측 유틸
# ============================================================
def resolve_ticker(user_input: str, kor_map: Dict[str,str]) -> str:
    user_input = user_input.strip().upper()
    for k,v in kor_map.items():
        if user_input in [k.upper(), v.upper()]: return v.upper()
    return user_input

def _build_features_for_infer(ticker: str, start_date: str, end_inclusive: str,
                              feat_cols_ref: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    raw = fetch_ohlcv_fixed_window(ticker, start_date, end_inclusive)
    feat = build_features(raw, ticker).replace([np.inf,-np.inf], np.nan).ffill()
    if feat_cols_ref is not None:
        feat = feat.reindex(columns=feat_cols_ref)
        feat = feat.ffill().fillna(0.0)
    close = raw["Close"]
    return feat, close

def predict_next_close(user_input: str, artifact: dict,
                       start_date: str = START_DATE,
                       end_inclusive: str = END_DATE_INCLUSIVE) -> Dict[str, float]:
    ticker = resolve_ticker(user_input, nasdaq100_kor)
    model = artifact["model"]
    scaler_tech = artifact["scaler_tech"]
    scaler_fund = artifact["scaler_fund"]
    id_map = artifact["id_map"]
    lookback = artifact["lookback"]
    feat_cols_ref = artifact.get("feat_cols"); K = len(id_map)

    feat, close = _build_features_for_infer(ticker, start_date, end_inclusive, feat_cols_ref=feat_cols_ref)
    if len(feat) < lookback+1: raise ValueError(f"Not enough rows for {ticker} to build a window of {lookback}.")
    # TECH window
    Xwin_tech = feat[FEATURES_TECH].iloc[-lookback:].values.astype(np.float32)
    W,Ft = Xwin_tech.shape
    Xwin_tech_s = scaler_tech.transform(Xwin_tech.reshape(W, Ft)).reshape(1, W, Ft).astype(np.float32)
    # FUND vector (last day)
    xvec_fund = feat[FEATURES_FUND].iloc[-1:].values.astype(np.float32)  # shape (1,Ff)
    xvec_fund_s = scaler_fund.transform(xvec_fund).astype(np.float32)
    # one-hot
    if ticker in id_map:
        idx = id_map[ticker]; oh_seq = np.repeat(np.eye(K, dtype=np.float32)[idx][None,None,:], repeats=W, axis=1)
        oh_vec = np.eye(K, dtype=np.float32)[idx][None,:]
    else:
        oh_seq = np.zeros((1,W,K), dtype=np.float32); oh_vec = np.zeros((1,K), dtype=np.float32)
    Xseq_in = np.concatenate([Xwin_tech_s, oh_seq], axis=2).astype(np.float32)
    Xfund_in = np.concatenate([xvec_fund_s, oh_vec], axis=1).astype(np.float32)

    model.eval()
    with torch.no_grad():
        xs = torch.from_numpy(Xseq_in).to(DEVICE)
        xf = torch.from_numpy(Xfund_in).to(DEVICE)
        rhat_log,_,_,_ = model(xs, xf)
        rhat_log = float(rhat_log.detach().cpu().numpy().ravel()[0])
    C_t = float(close.iloc[-1])
    next_close = float(C_t * np.exp(rhat_log))
    return {"ticker": ticker, "today_close": C_t, "pred_next_close": next_close}

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

def predict_month_with_daily_assimilation(user_input: str, artifact: dict,
                                          month_start: str = JAN_START, month_end: str = JAN_END,
                                          start_date: str = START_DATE, last_train_day: str = END_DATE_INCLUSIVE):
    ticker = resolve_ticker(user_input, nasdaq100_kor)
    basis = last_train_day; rows = []
    while True:
        pred = predict_next_close(user_input, artifact, start_date=start_date, end_inclusive=basis)
        next_dt, actual_close = fetch_next_trading_close_after(basis, ticker)
        next_dt_ts = pd.to_datetime(next_dt)
        if next_dt_ts < pd.to_datetime(month_start): basis = next_dt; continue
        if next_dt_ts > pd.to_datetime(month_end): break
        rows.append({
            "기준일": basis, "예측대상일": next_dt, "ticker": ticker,
            "기준일종가": pred["today_close"], "예측종가": pred["pred_next_close"],
            "실제종가": actual_close,
            "오차": pred["pred_next_close"] - actual_close,
            "오차율(%)": ((pred["pred_next_close"] - actual_close) * 100.0 / actual_close)
        })
        basis = next_dt
        if pd.to_datetime(basis) >= pd.to_datetime(month_end): break
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

# ============================================================
# Global builder (no dates; fixed HP training)
# ============================================================
def build_global_dual(frames_dict, tickers, id_map, lookback):
    Xt, Xf, ys, Cs, IDs = [], [], [], [], []
    for t in tickers:
        df = frames_dict[t]
        xt, xf, y, C = make_sequences_from_frame_dual(df, lookback)
        if len(xt)==0: continue
        ids = np.full(len(xt), id_map[t])
        Xt.append(xt); Xf.append(xf); ys.append(y); Cs.append(C); IDs.append(ids)
    cat = lambda lst, axis=0: np.concatenate(lst, axis=axis) if lst else np.empty((0,))
    return cat(Xt), cat(Xf), cat(ys), cat(Cs), cat(IDs)

# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    TICKERS = sorted(set(nasdaq100_kor.values()))
    TEST_RATIO = 0.2

    print(f"[프레임 구축] {START_DATE} ~ {END_DATE_INCLUSIVE}")
    raw_frames = build_frames_for_tickers(TICKERS, start_date=START_DATE, end_inclusive=END_DATE_INCLUSIVE)
    frames_all, feat_cols_ref = align_feature_columns_across_frames(raw_frames)

    split = {t: split_train_val_test_by_time(frames_all[t], test_ratio=TEST_RATIO) for t in TICKERS}
    trainval_frames = {t: split[t][0] for t in TICKERS}
    test_frames     = {t: split[t][1] for t in TICKERS}
    id_map = {t:i for i,t in enumerate(TICKERS)}

    lookback   = FIXED["lookback"]
    Xt_tv, Xf_tv, y_tv, C_tv, ID_tv = build_global_dual(trainval_frames, TICKERS, id_map, lookback)
    Xt_te, Xf_te, y_te, C_te, ID_te = build_global_dual(test_frames,     TICKERS, id_map, lookback)

    n_tv = Xt_tv.shape[0]
    val_size = max(int(n_tv * 0.05), 1)
    tr_size = n_tv - val_size

    Xt_tr, Xf_tr, y_tr, C_tr, ID_tr = Xt_tv[:tr_size], Xf_tv[:tr_size], y_tv[:tr_size], C_tv[:tr_size], ID_tv[:tr_size]
    Xt_va, Xf_va, y_va, C_va, ID_va = Xt_tv[tr_size:], Xf_tv[tr_size:], y_tv[tr_size:], C_tv[tr_size:], ID_tv[tr_size:]

    scaler_tech, scaler_fund, Xt_tr_s, Xf_tr_s, Xt_va_s, Xf_va_s, Xt_te_s, Xf_te_s = \
        fit_transform_dual(Xt_tr, Xf_tr, Xt_va, Xf_va, Xt_te, Xf_te)

    K = len(id_map)
    Xt_tr_s = attach_onehot_seq(Xt_tr_s, ID_tr, K)
    Xt_va_s = attach_onehot_seq(Xt_va_s, ID_va, K)
    Xt_te_s = attach_onehot_seq(Xt_te_s, ID_te, K)
    Xf_tr_s = attach_onehot_vec(Xf_tr_s, ID_tr, K)
    Xf_va_s = attach_onehot_vec(Xf_va_s, ID_va, K)
    Xf_te_s = attach_onehot_vec(Xf_te_s, ID_te, K)

    train_loader = DataLoader(DualSeqDataset(Xt_tr_s, Xf_tr_s, y_tr), batch_size=FIXED["batch_size"], shuffle=False,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(DualSeqDataset(Xt_va_s, Xf_va_s, y_va), batch_size=FIXED["batch_size"], shuffle=False,
                              pin_memory=torch.cuda.is_available())

    tech_in = Xt_tr_s.shape[2]; fund_in = Xf_tr_s.shape[1]
    model = DualBranchGRUGatedRegressor(tech_in, fund_in,
                                        u1=FIXED["units1"], u2=FIXED["units2"],
                                        mlp_hidden=FIXED["mlp_hidden"],
                                        dropout=FIXED["dropout"], gate_hidden=FIXED["gate_hidden"])
    model = train_with_early_stopping(
        model, train_loader, val_loader,
        epochs=FIXED["epochs"], lr=FIXED["lr"], patience=FIXED["patience"],
        outdir=OUTDIR, tag="dual_best",
        save_params=FIXED
    )

    rep_test, _ = eval_seq_model(model, Xt_te_s, Xf_te_s, y_te, C_te, tag="DUAL TEST")

    # 아티팩트 저장
    with open(os.path.join(OUTDIR, "best_params_fixed.json"), "w") as f:
        json.dump(FIXED, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTDIR, "other_artifacts.pkl"), "wb") as f:
        pickle.dump({
            "scaler_tech": scaler_tech,
            "scaler_fund": scaler_fund,
            "id_map": id_map,
            "lookback": lookback,
            "tickers": TICKERS,
            "feat_cols": FEATURES
        }, f)

    # 추론용 번들(.pt) — 스케일러와 메타 포함
    inference_bundle = {
        "model_class": "DualBranchGRUGatedRegressor",
        "state_dict": model.state_dict(),
        "fixed": FIXED,
        "scaler_tech": scaler_tech,
        "scaler_fund": scaler_fund,
        "id_map": id_map,
        "lookback": lookback,
        "tickers": TICKERS,
        "feat_cols": FEATURES
    }
    torch.save(inference_bundle, os.path.join(OUTDIR, "inference_bundle.pt"))

    artifact = {
        "model": model,
        "scaler_tech": scaler_tech,
        "scaler_fund": scaler_fund,
        "id_map": id_map,
        "lookback": lookback,
        "tickers": TICKERS,
        "feat_cols": FEATURES
    }

    print("\n[2025-01 월간 예측: 재학습 없음, 일별 실제값 동화] (Dual Gated, Fixed-HP: dropout=0.12)")
    for name in ["엔비디아", "애플", "마이크로소프트"]:
        df_jan, mape_jan, dir_acc_jan = predict_month_with_daily_assimilation(
            name, artifact, month_start=JAN_START, month_end=JAN_END,
            start_date=START_DATE, last_train_day=END_DATE_INCLUSIVE
        )
        out_csv = os.path.join(OUTDIR, f"jan2025_{resolve_ticker(name, nasdaq100_kor)}.csv")
        df_jan.to_csv(out_csv, index=False)
        print(f"{name} → 2025-01 MAPE={mape_jan:.2f}% | 방향정확도={dir_acc_jan:.2f}% | CSV={out_csv}")

    print("\n저장 파일:")
    print(" - dual_best_best.pt")
    print(" - dual_best_final.pt")
    print(" - inference_bundle.pt  (스케일러/메타 포함 하나로 로드 가능)")
    print(" - best_params_fixed.json")
    print(" - other_artifacts.pkl")
    print("\n모든 작업 완료.")
