# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except Exception:
    HAS_TF = False

# ===================== 지표 함수 =====================
def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    u = d.clip(lower=0)
    v = -d.clip(upper=0)
    au = u.rolling(p).mean()
    av = v.rolling(p).mean()
    rs = au / av.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def wma(s: pd.Series, p: int) -> pd.Series:
    w = np.arange(1, p+1)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()

def macd(s: pd.Series, fast=12, slow=26, sig=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(s, fast) - ema(s, slow)
    return m, ema(m, sig), m - ema(m, sig)

def bollinger(s: pd.Series, p=20, k=2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    m = s.rolling(p).mean()
    sd = s.rolling(p).std()
    up = m + k*sd
    lo = m - k*sd
    width = (up - lo) / (m.replace(0, np.nan))
    pb = (s - lo) / (up - lo)
    return up, lo, width, pb

def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, p=14) -> pd.Series:
    return true_range(df).rolling(p).mean()

# ===== 핵심 지표 추가 =====
def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3.0

def obv(df: pd.DataFrame) -> pd.Series:
    c = df["Close"]; v = df["Volume"].astype(float)
    sign = np.sign(c.diff().fillna(0))
    return (sign * v).fillna(0).cumsum()

def tema(s: pd.Series, span: int = 20) -> pd.Series:
    e1 = ema(s, span); e2 = ema(e1, span); e3 = ema(e2, span)
    return 3 * (e1 - e2) + e3

def kama(s: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (s - s.shift(er_window)).abs()
    volatility = s.diff().abs().rolling(er_window).sum()
    er = (change / volatility).replace([np.inf, -np.inf], np.nan).fillna(0)
    fast_sc = 2 / (fast + 1); slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.zeros(len(s)); out[:] = np.nan
    idx0 = np.where(~s.isna())[0]
    if len(idx0) == 0: return pd.Series(out, index=s.index)
    out[idx0[0]] = s.iloc[idx0[0]]
    for i in range(idx0[0]+1, len(s)):
        if np.isnan(s.iloc[i]) or np.isnan(out[i-1]) or np.isnan(sc.iloc[i]):
            out[i] = out[i-1]
        else:
            out[i] = out[i-1] + sc.iloc[i]*(s.iloc[i]-out[i-1])
    return pd.Series(out, index=s.index)

# ===================== 인덱스/타임존 유틸 =====================
def _to_naive_utc_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

def _coerce_tz_naive(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj.copy()
        out.index = _to_naive_utc_index(out.index)
        return out
    return obj

# ===================== Feature & Target =====================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c, o, h, l = out["Close"], out["Open"], out["High"], out["Low"]
    v = out["Volume"].astype(float)

    # 수익률/모멘텀
    out["ret_1d"] = c.pct_change(1)
    out["ret_5d"] = c.pct_change(5)
    out["mom_20"] = c.pct_change(20)

    # 이동평균/추세
    out["rsi_14"] = rsi(c, 14)
    out["rsi_14_diff3"] = out["rsi_14"].diff(3)
    out["ema_20"] = ema(c, 20)
    out["wma_20"] = wma(c, 20)
    out["sma_20"] = sma(c, 20)
    out["sma_50"] = sma(c, 50)
    out["sma_200"] = sma(c, 200)
    out["sma20_slope"] = out["sma_20"].diff(5)
    out["ema_gt_wma"] = (out["ema_20"] > out["wma_20"]).astype(int)

    # 변동성/밴드
    _, _, bw, pb = bollinger(c, 20, 2.0)
    out["bb_width"] = bw
    out["bb_p"] = pb
    out["atr_14"] = atr(out, 14)

    # 실현변동성
    out["rv_20"] = out["ret_1d"].rolling(20).std()
    out["rv_60"] = out["ret_1d"].rolling(60).std()
    out["downvol_20"] = out["ret_1d"].clip(upper=0).rolling(20).std()

    # MACD
    m, s_, hst = macd(c)
    out["macd"], out["macd_sig"], out["macd_hist"] = m, s_, hst

    # 가격 상대위치/거래량
    out["px_vs_sma20"]  = (c - out["sma_20"]) / out["sma_20"]
    out["px_vs_sma50"]  = (c - out["sma_50"]) / out["sma_50"]
    out["px_vs_sma200"] = (c - out["sma_200"]) / out["sma_200"]
    out["vol_z"] = (v - v.rolling(60).mean()) / (v.rolling(60).std().replace(0, np.nan))

    # 캔들 구조
    prev_c = c.shift(1)
    out["gap_1d"] = (o - prev_c) / prev_c
    rng = (h - l).replace(0, np.nan)
    body = (c - o)
    out["body_rel"] = (body / rng).clip(-5, 5)
    out["upper_wick_rel"] = ((h - c.clip(lower=o)) / rng).clip(-5, 5)
    out["lower_wick_rel"] = (((o.clip(upper=c)) - l) / rng).clip(-5, 5)

    # 핵심 지표
    out["hlc3"] = hlc3(out)
    out["obv"] = obv(out)
    out["tema_20"] = tema(c, 20)
    out["kama_10"] = kama(c, 10, 2, 30)

    # 벤치마크 상대강도(충분히 있으면)
    for bn in ["SPY", "QQQ", "SOXX"]:
        c_bn = f"Close_{bn}"
        if c_bn in out.columns and out[c_bn].notna().mean() > 0.98:
            rs = c / out[c_bn]
            out[f"rs_{bn.lower()}_ma20_gap"] = rs - rs.rolling(20).mean()
            out[f"rs_{bn.lower()}_slope5"] = rs.diff(5)
            out[f"rs_{bn.lower()}_ret5"] = rs.pct_change(5)
    return out

def make_target_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    return (df["Close"].shift(-horizon) / df["Close"] - 1.0)

FEATURE_BASE = [
    "ret_1d","ret_5d","mom_20","rsi_14","rsi_14_diff3","ema_20","wma_20","sma_20","sma_50","sma_200",
    "sma20_slope","ema_gt_wma","bb_width","bb_p","atr_14","rv_20","rv_60","downvol_20",
    "macd","macd_sig","macd_hist","px_vs_sma20","px_vs_sma50","px_vs_sma200",
    "vol_z","gap_1d","body_rel","upper_wick_rel","lower_wick_rel",
    "hlc3","obv","tema_20","kama_10"
]
FEATURE_CORE6 = ["Close","hlc3","tema_20","atr_14","bb_p","obv","px_vs_sma20"]

def select_features(feat: pd.DataFrame) -> List[str]:
    cols = [f for f in FEATURE_BASE if f in feat.columns]
    cols += [c for c in feat.columns if c.startswith(("rs_spy_","rs_qqq_","rs_soxx_"))]
    return cols

# ===================== 시계열 split =====================
def purged_splits(n: int, n_splits: int = 5, embargo: int = 5):
    n_splits = max(2, min(n_splits, n // max(embargo, 1)))
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    idx = np.arange(n); current = 0
    for fs in fold_sizes:
        start, stop = current, current + fs
        va = idx[start:stop]
        tr_end = max(0, start - embargo)
        tr = idx[:tr_end]
        if len(tr)>0 and len(va)>0:
            yield tr, va
        current = stop

# ===================== LSTM 유틸 =====================
def build_sequences(df_feat: pd.DataFrame, y_ret: pd.Series, lookback: int, feature_cols: List[str]):
    Xdf = df_feat[feature_cols].copy()
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(Xdf), index=Xdf.index, columns=Xdf.columns)

    y = y_ret.astype(float).dropna()
    common = X_scaled.index.intersection(y.index)
    X_scaled = X_scaled.loc[common]
    y = y.loc[common]

    X_list, y_list, idx_list = [], [], []
    for i in range(lookback, len(X_scaled)):
        X_list.append(X_scaled.iloc[i-lookback:i].values)
        y_list.append(y.iloc[i])
        idx_list.append(X_scaled.index[i])

    if not X_list:
        return None, None, None, None
    return np.array(X_list, np.float32), np.array(y_list, np.float32), pd.DatetimeIndex(idx_list), scaler

def make_lstm(input_shape: Tuple[int,int], units=64, layers_n=2, dropout=0.2):
    model = keras.Sequential()
    if layers_n==1:
        model.add(layers.LSTM(units, input_shape=input_shape))
        model.add(layers.Dropout(dropout))
    else:
        model.add(layers.LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(dropout))
        model.add(layers.LSTM(units))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    return model

# ===================== 평가 보조 =====================
def fix_quantile_crossing(q10: pd.Series, q50: pd.Series, q90: pd.Series) -> Tuple[pd.Series,pd.Series,pd.Series]:
    """분위수 교차 방지: p10 <= p50 <= p90 강제."""
    p10 = pd.concat([q10, q50, q90], axis=1).min(axis=1)
    p90 = pd.concat([q10, q50, q90], axis=1).max(axis=1)
    p50 = q50.clip(lower=p10, upper=p90)
    return p10, p50, p90

# ===================== 홀드아웃 평가(우선) =====================
def evaluate_price_model_holdout(
    ticker: str,
    horizon: int = 1,
    years: int = 8,
    tz: str = "Asia/Seoul",
    test_ratio: float = 0.2,
    use_lstm: bool = True,
    lstm_feature_mode: str = "core6",
    blend_alpha: float = 0.6,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    # --- 데이터 가져오기
    as_of = pd.Timestamp.now(tz=tz).normalize()
    start = (as_of - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    end   = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    main = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    if main.empty:
        raise ValueError("No data fetched")
    main = _coerce_tz_naive(main[["Open","High","Low","Close","Volume"]].dropna())

    # 벤치마크(선택)
    for bn in ["SPY","QQQ","SOXX"]:
        try:
            bdf = yf.Ticker(bn).history(start=start, end=end, interval="1d", auto_adjust=True)
            if not bdf.empty:
                main = main.join(_coerce_tz_naive(bdf["Close"]).rename(f"Close_{bn}"), how="left")
        except Exception:
            pass
    for bn in ["SPY","QQQ","SOXX"]:
        col = f"Close_{bn}"
        if col in main.columns:
            main[col] = main[col].ffill().bfill()

    feat = build_features(main)
    y_ret = _coerce_tz_naive(make_target_return(main, horizon))

    # --- 조립 및 시간순 분할
    use_cols = select_features(feat)
    frame = pd.concat([feat[use_cols], y_ret.rename("y_ret")], axis=1).dropna(subset=use_cols + ["y_ret"])
    X = frame[use_cols]; y = frame["y_ret"]; idx = frame.index
    n = len(frame); cut = int(np.floor(n*(1.0-test_ratio))); cut = np.clip(cut, 50, n-1)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]; ytr, yte = y.iloc[:cut], y.iloc[cut:]; idx_te = idx[cut:]
    close_t = main["Close"].reindex(idx)  # t 시점 종가

    # --- 탭형 회귀 + 분위수
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

    hgb = HistGradientBoostingRegressor(max_depth=4, max_iter=800, learning_rate=0.05, min_samples_leaf=50, random_state=42)
    hgb.fit(Xtr_s, ytr)
    ret_tab = pd.Series(hgb.predict(Xte_s), index=idx_te)

    reg_q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, random_state=42).fit(Xtr_s, ytr)
    reg_q50 = GradientBoostingRegressor(loss="quantile", alpha=0.50, random_state=42).fit(Xtr_s, ytr)
    reg_q90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, random_state=42).fit(Xtr_s, ytr)
    q10 = pd.Series(reg_q10.predict(Xte_s), index=idx_te)
    q50 = pd.Series(reg_q50.predict(Xte_s), index=idx_te)
    q90 = pd.Series(reg_q90.predict(Xte_s), index=idx_te)
    q10, q50, q90 = fix_quantile_crossing(q10, q50, q90)

    # --- LSTM (옵션)
    ret_lstm = None
    if use_lstm and HAS_TF:
        cols = (["Close"] if lstm_feature_mode=="close" else [c for c in FEATURE_CORE6 if c in feat.columns]) or ["Close"]
        X_seq, y_seq, idx_seq, sc_mm = build_sequences(feat, y_ret, lookback=60, feature_cols=cols)
        if X_seq is not None:
            # 테스트 기간 기준으로 시퀀스를 분할
            te_start_time = idx_te[0]
            mask_tr = idx_seq < te_start_time
            mask_te = idx_seq >= te_start_time
            Xseq_tr, yseq_tr = X_seq[mask_tr], y_seq[mask_tr]
            Xseq_te, idx_seq_te = X_seq[mask_te], idx_seq[mask_te]
            if len(Xseq_tr) > 50 and len(Xseq_te) > 0:
                model = make_lstm(input_shape=(X_seq.shape[1], X_seq.shape[2]), units=64, layers_n=2, dropout=0.2)
                es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                cut_l = int(len(Xseq_tr)*0.8)
                model.fit(Xseq_tr[:cut_l], yseq_tr[:cut_l],
                          validation_data=(Xseq_tr[cut_l:], yseq_tr[cut_l:]),
                          epochs=100, batch_size=64, callbacks=[es], verbose=0)
                ret_lstm = pd.Series(model.predict(Xseq_te, verbose=0).ravel(), index=idx_seq_te)

    # --- 앙상블 & 가격 복원
    if ret_lstm is not None:
        common_te = ret_tab.index.intersection(ret_lstm.index)  # 공통 날짜에 맞춤
        ret_pred = blend_alpha*ret_tab.reindex(common_te) + (1-blend_alpha)*ret_lstm.reindex(common_te)
        q10, q50, q90 = q10.reindex(common_te), q50.reindex(common_te), q90.reindex(common_te)
        yte = yte.reindex(common_te); idx_te = common_te
    else:
        ret_pred = ret_tab

    close_base = close_t.reindex(idx_te)                         # t 시점 종가
    actual_next_close = (1.0 + yte) * close_base                 # 실제 t+1 종가
    pred_close_med   = (1.0 + ret_pred) * close_base
    pred_close_p10   = (1.0 + q10) * close_base
    pred_close_p50   = (1.0 + q50) * close_base
    pred_close_p90   = (1.0 + q90) * close_base

    # --- 메트릭(우선)
    rmse = float(np.sqrt(mean_squared_error(yte, ret_pred)))
    mae  = float(mean_absolute_error(yte, ret_pred))
    mape = float(np.mean(np.abs(ret_pred - yte) / (np.abs(yte)+1e-8)))
    mae_price = float(np.mean(np.abs(pred_close_med - actual_next_close)))
    r2 = float(r2_score(yte, ret_pred)) if len(yte) > 1 else np.nan
    dir_acc = float(np.mean(np.sign(ret_pred) == np.sign(yte)))
    in_band = (actual_next_close >= pred_close_p10) & (actual_next_close <= pred_close_p90)
    coverage = float(np.mean(in_band))
    band_width = float(np.mean((pred_close_p90 - pred_close_p10).abs()))

    metrics = {
        "RMSE_ret": round(rmse, 6),
        "MAE_ret": round(mae, 6),
        "MAPE_ret": round(mape, 6),
        "MAE_price": round(mae_price, 4),
        "R2_ret": round(r2, 4),
        "Directional_Accuracy": round(dir_acc, 3),
        "Coverage_p10_p90": round(coverage, 3),
        "Avg_Band_Width": round(band_width, 4),
        "test_points": int(len(idx_te)),
        "used_lstm": bool(ret_lstm is not None),
        "blend_alpha": blend_alpha
    }

    preds = pd.DataFrame({
        "ret_true": yte,
        "ret_pred": ret_pred,
        "pred_close_med": pred_close_med,
        "pred_close_p10": pred_close_p10,
        "pred_close_p50": pred_close_p50,
        "pred_close_p90": pred_close_p90,
        "actual_next_close": actual_next_close,
        "hit_p10_p90": in_band,
    })
    return metrics, preds

# ===================== 실시간 추정(원하면) =====================
class PriceForecastAgent:
    def __init__(self, scaler, hgb, reg_q10, reg_q50, reg_q90,
                 lstm_model=None, lstm_scaler=None, lstm_features: Optional[List[str]]=None,
                 blend_alpha: float = 0.6):
        self.scaler = scaler; self.hgb = hgb
        self.reg_q10 = reg_q10; self.reg_q50 = reg_q50; self.reg_q90 = reg_q90
        self.lstm_model = lstm_model; self.lstm_scaler = lstm_scaler
        self.lstm_features = lstm_features or []
        self.blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))

    def predict_next_close(self, raw_df: pd.DataFrame, feat: pd.DataFrame) -> Dict:
        last_close = float(raw_df["Close"].iloc[-1])
        X = feat.copy()
        X = X.dropna().tail(1)
        Xs = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

        ret_tab = float(self.hgb.predict(Xs)[0])
        q10 = float(self.reg_q10.predict(Xs)[0])
        q50 = float(self.reg_q50.predict(Xs)[0])
        q90 = float(self.reg_q90.predict(Xs)[0])
        # LSTM
        ret_lstm = None
        if (self.lstm_model is not None) and len(self.lstm_features)>0:
            need = 60
            Xdf = raw_df[self.lstm_features].tail(need).copy()
            X_scaled = pd.DataFrame(self.lstm_scaler.transform(Xdf), index=Xdf.index, columns=Xdf.columns)
            X_seq = np.expand_dims(X_scaled.values, axis=0)
            ret_lstm = float(self.lstm_model.predict(X_seq, verbose=0).ravel()[0])

        ret_med = ret_tab if ret_lstm is None else (self.blend_alpha*ret_tab + (1-self.blend_alpha)*ret_lstm)
        pred_close   = round(last_close*(1+ret_med), 2)
        pred_close_p10 = round(last_close*(1+min(q10,q50,q90)), 2)
        pred_close_p50 = round(last_close*(1+q50), 2)
        pred_close_p90 = round(last_close*(1+max(q10,q50,q90)), 2)
        return {
            "현재종가": round(last_close,2),
            "예측종가": pred_close,
            "예측구간": {"p10": pred_close_p10, "p50": pred_close_p50, "p90": pred_close_p90},
            "수익률(중위)": round(ret_med,6)
        }

# ===================== 실행 예시 =====================
if __name__ == "__main__":
    # 1) 홀드아웃 평가(지표 우선 출력)
    metrics, preds = evaluate_price_model_holdout(
        ticker="AAPL",
        horizon=1,
        years=8,
        tz="Asia/Seoul",
        test_ratio=0.2,
        use_lstm=True,            # TensorFlow 없으면 자동으로 False처럼 동작
        lstm_feature_mode="core6",
        blend_alpha=0.6
    )
    import json
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    # 예측 상세 상위 5행
    print(preds.head().round(4))

    # 2) (선택) 실시간 1스텝 예측 사용법
    #  - 평가용으로 학습된 개별 객체들을 재사용하려면,
    #    evaluate 함수 안 학습 파트를 모듈화해야 하지만, 간단 예시만 남김.
