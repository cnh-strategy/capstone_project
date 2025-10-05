import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from yfinace.nasdaq_100 import nasdaq100_eng
"""
[지금 파이프라인에서 라벨 정의]
y_close_t1 = 다음날 종가
ret_t1 = 다음날 수3익률
즉, 학습 타깃은 y_close_t1 또는 ret_t1 중 하나입니다.
close는 “오늘 종가”라서 타깃은 아니고 설명 변수로 쓸 수 있음입니다.

[상대 재무제표 가격은 반영하지 않음]
"""


# =========================
# 설정
# =========================
DATA_PATH = "2025/fallback_data.csv"
TICKERS = list(nasdaq100_eng.values())
WINDOW_CAP_DAYS = 730             # 최근 24개월 데이터만 학습
MIN_TRAIN_DAYS = 365              # 최소 학습 데이터 기간(1년)

# =========================
# 유틸
# =========================
def quarter_start(d):
    return pd.Timestamp(d.year, 3*((d.month-1)//3)+1, 1)

def quarter_end(d):
    return (quarter_start(d) + pd.offsets.QuarterEnd(0))

def mape(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    eps = 1e-8
    return np.mean(np.abs((yt - yp) / (np.clip(np.abs(yt), eps, None)))) * 100.0

# =========================
# 1) 데이터 로드/필터
# =========================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df[df["symbol"].isin(TICKERS)].copy()
df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)

# -------------------------
# 타깃: ret_t1 (다음날 수익률)
# -------------------------
df["target"] = df["ret_t1"]

# 원핫 인코딩 (심볼 구분)
sym_dum = pd.get_dummies(df["symbol"], prefix="sym")
df = pd.concat([df, sym_dum], axis=1)

# 피처 선택 (불필요한 컬럼 제거)
drop_cols = [
    "Date", "symbol", "period", "period_with_lag",
    "y_close_t1", "ret_t1", "target",  # 타깃과 라벨 관련 컬럼 제외
    "announce_date", "release_date", "Close"
]
drop_cols = [c for c in drop_cols if c in df.columns]  # 존재하는 컬럼만 drop
feature_cols = [c for c in df.columns if c not in drop_cols]

# 결측 제거
df = df.dropna(subset=["target"]).reset_index(drop=True)

# =========================
# 2) Walk-forward 분기별 학습/예측
# =========================
dates = df["Date"].sort_values().unique()
global_start = dates.min()
global_end = dates.max()

q_starts = (
    pd.Series(pd.to_datetime(dates))
    .map(quarter_start)
    .drop_duplicates()
    .sort_values()
    .tolist()
)

records = []
all_preds = []

scaler = StandardScaler()

for qs in q_starts:
    qe = quarter_end(qs)
    test_mask = (df["Date"] >= qs) & (df["Date"] <= qe)
    train_end = qs - pd.Timedelta(days=1)
    if train_end < global_start:
        continue
    train_start_cap = qs - pd.Timedelta(days=WINDOW_CAP_DAYS)
    train_start = max(global_start, train_start_cap)
    train_mask = (df["Date"] >= train_start) & (df["Date"] <= train_end)

    if df.loc[train_mask].shape[0] < MIN_TRAIN_DAYS:
        continue
    if df.loc[test_mask].empty:
        continue

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, "target"]
    X_test = df.loc[test_mask, feature_cols].fillna(0)
    y_test = df.loc[test_mask, "target"]

    # 스케일링
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 학습
    model = LGBMRegressor(
        n_estimators=1182,
        learning_rate=0.025956710071102375,
        num_leaves=237,
        max_depth=7,
        subsample=0.6017969264005049,
        colsample_bytree=0.8889777947280558,
        random_state=42,
        reg_alpha=2.690686521830618,
        reg_lambda=0.7632631167691681
    )
    model.fit(X_train_scaled, y_train)

    # 예측 (수익률 예측)
    y_pred_ret = model.predict(X_test_scaled)

    # -------------------------
    # 수익률 → 절대가격 복원
    # -------------------------
    last_close = df.loc[test_mask, "close"].values
    y_pred_close = last_close * (1 + y_pred_ret)
    y_true_close = df.loc[test_mask, "y_close_t1"].values

    # 평가
    rmse = np.sqrt(mean_squared_error(y_true_close, y_pred_close))
    mae = mean_absolute_error(y_true_close, y_pred_close)
    mp = mape(y_true_close, y_pred_close)

    records.append({
        "quarter_start": qs.date(),
        "quarter_end": qe.date(),
        "train_start": pd.Timestamp(train_start).date(),
        "train_end": train_end.date(),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mp,
    })

    part = df.loc[test_mask, ["Date", "symbol", "close", "y_close_t1"]].copy()
    part["y_true"] = y_true_close
    part["y_pred"] = y_pred_close
    part["quarter_start"] = qs.date()
    all_preds.append(part)

# =========================
# 3) 결과 출력
# =========================
res = pd.DataFrame(records)
preds = pd.concat(all_preds).sort_values(["Date", "symbol"])

print("=== Walk-forward Quarterly Metrics (tail) ===")
print(res.tail(6))
print("\n=== 전체 평균 성능 ===")
print(res[["RMSE", "MAE", "MAPE(%)"]].mean())
print("\n=== 예측 결과 샘플 ===")
print(preds.head(10))

# 피처 중요도 출력
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n=== Feature Importance (Top 20) ===")
print(importances.head(20))

# =========================
# 4) 모델 저장 (선택)
# =========================
MODEL_DIR = "2025/models22"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "final_lgbm.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)
