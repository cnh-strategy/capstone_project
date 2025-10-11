import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

from agents.fundamental_sub import MacroSentimentAgent

# -------------------------------------------------------------
# 1. 모델과 스케일러 불러오기
# -------------------------------------------------------------
model = load_model("models/multi_output_lstm_model.h5", compile=False)
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# -------------------------------------------------------------
# 2. 기준일 및 윈도우 설정
# -------------------------------------------------------------
base_date = datetime(2025, 10, 11)
window = 40  # 최근 40일 시퀀스 입력

# -------------------------------------------------------------
# 3. MacroSentimentAgent로 최신 데이터 가져오기
# -------------------------------------------------------------
macro_agent = MacroSentimentAgent(base_date=base_date, window=40)
macro_agent.fetch_data()       # yfinance API에서 직접 1d 데이터 다운로드
macro_agent.add_features()     # 수익률, 금리차, 위험심리 등 계산
macro_df = macro_agent.data
macro_df = macro_df.reset_index()

# 날짜 정리
macro_df = macro_df.tail(window + 5)


# -------------------------------------------------------------
# 4. 피처 정리 (학습 시 동일 구조)
# -------------------------------------------------------------
# 매크로 데이터 컬럼 평탄화
# ✅ 이미 MacroSentimentAgent에서 feature 생성 완료
macro_full = macro_df.copy()
# 피처 선택
feature_cols = [c for c in macro_full.columns if c != "Date"]
X_input = macro_full[feature_cols]

# -------------------------------------------------------------
# 4-1. 스케일러 feature 구조 맞추기 (안정 버전)
# -------------------------------------------------------------
expected_features = list(scaler_X.feature_names_in_)

# 1. 누락된 피처는 0으로 채움
for col in expected_features:
    if col not in X_input.columns:
        X_input[col] = 0

# 2. 불필요한 피처 제거
X_input = X_input[expected_features]

# ✅ 확인용 로그
print(f"[Check] 예측 데이터 feature 수: {X_input.shape[1]}, 스케일러 feature 수: {len(expected_features)}")
missing = [c for c in expected_features if c not in X_input.columns]
extra = [c for c in X_input.columns if c not in expected_features]
if missing:
    print(f"[WARN] 누락된 컬럼 {len(missing)}개: {missing[:5]} ...")
if extra:
    print(f"[WARN] 불필요 컬럼 {len(extra)}개: {extra[:5]} ...")

# -------------------------------------------------------------
# 5. 스케일링 및 시퀀스 생성
# -------------------------------------------------------------
X_scaled = scaler_X.transform(X_input)
X_scaled = pd.DataFrame(X_scaled, columns=expected_features)

# 최근 40일 데이터 시퀀스만
if len(X_scaled) < window:
    raise ValueError(f"데이터가 {window}일보다 적습니다.")
X_seq = np.expand_dims(X_scaled.tail(window).values, axis=0)


# -------------------------------------------------------------
# 6. 예측 수행
# -------------------------------------------------------------
pred_scaled = model.predict(X_seq)
pred_inv = scaler_y.inverse_transform(pred_scaled)

# -------------------------------------------------------------
# 7. 예측 결과: 수익률 → 종가 변환
# -------------------------------------------------------------
tickers = ["AAPL", "MSFT", "NVDA"]

# 최근 실제 종가 추출 (MacroSentimentAgent에서 가져온 마지막 행 기준)
last_prices = {}
for t in tickers:
    # MacroSentimentAgent 내부에서 AAPL_ma5, AAPL_ret1 등 있으므로 원 종가를 직접 다운로드했을 가능성 있음.
    # 종가 컬럼명을 추정 (AAPL_ma5 기준으로 앞부분만 가져옴)
    close_candidates = [c for c in macro_df.columns if c.startswith(t) and c.endswith("_ma5") is False and "ret" not in c]
    if len(close_candidates) == 0:
        raise ValueError(f"{t}의 종가 컬럼을 찾을 수 없습니다.")
    # 첫 번째 해당 컬럼 사용
    last_prices[t] = macro_df[close_candidates[0]].iloc[-1]

# 예측 수익률을 실제 종가로 변환
pred_prices = {}
for i, t in enumerate(tickers):
    pred_ret = pred_inv[0][i]
    last_price = last_prices[t]
    next_price = last_price * (1 + pred_ret)
    pred_prices[t] = next_price
    print(f"{t}: 마지막 종가={last_price:.2f} → 예측 종가={next_price:.2f} (예상 수익률 {pred_ret*100:.2f}%)")

# -------------------------------------------------------------
# 8. (선택) 예측 결과 요약
# -------------------------------------------------------------
pred_df = pd.DataFrame({
    "Ticker": tickers,
    "Last_Close": [last_prices[t] for t in tickers],
    "Predicted_Close": [pred_prices[t] for t in tickers],
    "Predicted_Return": pred_inv[0],
    "Predicted_%": pred_inv[0] * 100
})

print("\n================= 예측 결과 (종가 기준) =================")
print(pred_df.round(4))
