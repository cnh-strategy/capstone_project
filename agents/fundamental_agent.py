import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

from agents.fundamental_sub import MacroSentimentAgent


class MarketPredictor:
    """
    다중 자산 LSTM 예측 파이프라인 클래스
    - MacroSentimentAgent로 데이터 수집 및 피처 생성
    - 스케일링, 시퀀스 생성, 예측, 종가 변환 수행
    """

    def __init__(self,
                 model_path="models/multi_output_lstm_model.h5",
                 scaler_X_path="models/scaler_X.pkl",
                 scaler_y_path="models/scaler_y.pkl",
                 base_date=datetime(2025, 10, 11),
                 window=40,
                 tickers=None):
        self.model_path = model_path
        self.scaler_X_path = scaler_X_path
        self.scaler_y_path = scaler_y_path
        self.base_date = base_date
        self.window = window
        self.tickers = tickers or ["AAPL", "MSFT", "NVDA"]

        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.macro_df = None
        self.pred_df = None
        self.X_scaled = None

    # -------------------------------------------------------------
    # 1. 모델 및 스케일러 로드
    # -------------------------------------------------------------
    def load_assets(self):
        print("[INFO] 모델 및 스케일러 로드 중...")
        self.model = load_model(self.model_path, compile=False)
        self.scaler_X = joblib.load(self.scaler_X_path)
        self.scaler_y = joblib.load(self.scaler_y_path)
        print("[OK] 모델 및 스케일러 로드 완료")

    # -------------------------------------------------------------
    # 2. MacroSentimentAgent로 최신 매크로 데이터 수집
    # -------------------------------------------------------------
    def fetch_macro_data(self):
        print("[INFO] MacroSentimentAgent 데이터 수집 중...")
        macro_agent = MacroSentimentAgent(base_date=self.base_date, window=self.window)
        macro_agent.fetch_data()
        macro_agent.add_features()
        df = macro_agent.data.reset_index()
        self.macro_df = df.tail(self.window + 5)
        print(f"[OK] 매크로 데이터 수집 완료: {self.macro_df.shape}")

    # -------------------------------------------------------------
    # 3. 피처 정리 및 스케일링
    # -------------------------------------------------------------
    def prepare_features(self):
        print("[INFO] 피처 정리 및 스케일링 중...")

        macro_full = self.macro_df.copy()
        feature_cols = [c for c in macro_full.columns if c != "Date"]
        X_input = macro_full[feature_cols]

        expected_features = list(self.scaler_X.feature_names_in_)

        # 누락된 피처는 0으로 채움
        for col in expected_features:
            if col not in X_input.columns:
                X_input[col] = 0

        # 불필요한 피처 제거
        X_input = X_input[expected_features]

        print(f"[Check] 입력 피처 수: {X_input.shape[1]} / 스케일러 기준 피처 수: {len(expected_features)}")

        X_scaled = self.scaler_X.transform(X_input)
        X_scaled = pd.DataFrame(X_scaled, columns=expected_features)

        if len(X_scaled) < self.window:
            raise ValueError(f"데이터가 {self.window}일보다 적습니다.")

        X_seq = np.expand_dims(X_scaled.tail(self.window).values, axis=0)
        print("[OK] 스케일링 및 시퀀스 변환 완료")
        self.X_scaled = X_scaled
        return X_seq

    # -------------------------------------------------------------
    # 4. 예측 수행 및 결과 변환
    # -------------------------------------------------------------
    def predict(self, X_seq):
        print("[INFO] 예측 수행 중...")
        pred_scaled = self.model.predict(X_seq)
        pred_inv = self.scaler_y.inverse_transform(pred_scaled)

        last_prices = {}
        for t in self.tickers:
            close_candidates = [c for c in self.macro_df.columns
                                if c.startswith(t) and not c.endswith("_ma5") and "ret" not in c]
            if not close_candidates:
                raise ValueError(f"{t}의 종가 컬럼을 찾을 수 없습니다.")
            last_prices[t] = self.macro_df[close_candidates[0]].iloc[-1]

        pred_prices = {}
        for i, t in enumerate(self.tickers):
            pred_ret = pred_inv[0][i]
            last_price = last_prices[t]
            next_price = last_price * (1 + pred_ret)
            pred_prices[t] = next_price
            print(f"{t}: 마지막 종가={last_price:.2f} → 예측 종가={next_price:.2f} (예상 수익률 {pred_ret*100:.2f}%)")

        pred_df = pd.DataFrame({
            "Ticker": self.tickers,
            "Last_Close": [last_prices[t] for t in self.tickers],
            "Predicted_Close": [pred_prices[t] for t in self.tickers],
            "Predicted_Return": pred_inv[0],
            "Predicted_%": pred_inv[0] * 100
        })

        self.pred_df = pred_df.round(4)
        print("\n================= 예측 결과 (종가기준 표) =================")
        print(self.pred_df)

        self.pred_prices = pred_prices
        print("\n================= 예측 결과 (종가 기준) =================")
        print(pred_prices)

        return pred_prices

    # -------------------------------------------------------------
    # 5. 전체 실행 파이프라인
    # -------------------------------------------------------------
    def run_prediction(self):
        self.load_assets()
        self.fetch_macro_data()
        X_seq = self.prepare_features()
        pred_prices = self.predict(X_seq)

        # ✅ np.float64 → float 변환
        pred_prices = {k: float(v) for k, v in pred_prices.items()}
        return pred_prices, self.X_scaled


# -------------------------------------------------------------
# 실행 예시
# -------------------------------------------------------------
if __name__ == "__main__":
    predictor = MarketPredictor(
        base_date=datetime(2025, 10, 11),
        window=40,
        tickers=["AAPL", "MSFT", "NVDA"]
    )
    result, _ = predictor.run_prediction()

    print("\n[최종 결과 반환]")
    print(result)
