import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

from agents.macro_sub import get_std_pred, MakeDatasetMacro
from debate_ver3_tmp.agents.base_agent import BaseAgent, Target
from debate_ver3_tmp.config.agents import dir_info

model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]

class MacroPredictor(BaseAgent):
    """
    다중 자산 LSTM 예측 파이프라인 클래스
    - MacroSentimentAgent로 데이터 수집 및 피처 생성
    - 스케일링, 시퀀스 생성, 예측, 종가 변환 수행
    """

    def __init__(self,
                 base_date=datetime.today(),
                 window=40,
                 ticker=None
                 ,agent_id='MacroAgent',
                 **kwargs):
        self.agent_id = agent_id
        BaseAgent.__init__(self, self.agent_id, **kwargs)
        self.model_path = f"{model_dir}/{ticker}_{agent_id}.h5"
        self.scaler_X_path = f"{model_dir}/scaler_X.pkl"
        self.scaler_y_path = f"{model_dir}/scaler_y.pkl"
        self.base_date = base_date
        self.window = window
        self.tickers = [ticker] or ["AAPL", "MSFT", "NVDA"]
        # self.target_tickers = target_tickers or ["AAPL", "MSFT", "NVDA"]

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
        macro_agent = MakeDatasetMacro(base_date=self.base_date, window=self.window, target_tickers=self.tickers)
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
    def predictor(self, X_seq):
        print("[INFO] 예측 수행 중...")

        # 1. 모델 예측
        pred_scaled = self.model.predict(X_seq)
        pred_inv = self.scaler_y.inverse_transform(pred_scaled)

        # 2. 종가 추출
        last_prices = {}
        for t in self.tickers:
            close_candidates = [c for c in self.macro_df.columns
                                if c.startswith(t) and not c.endswith("_ma5") and "ret" not in c]
            if not close_candidates:
                raise ValueError(f"{t}의 종가 컬럼을 찾을 수 없습니다.")
            last_prices[t] = self.macro_df[close_candidates[0]].iloc[-1]

        # 3. 예측 종가 및 수익률 계산
        records = []
        pred_prices = {}
        for i, t in enumerate(self.tickers):
            pred_ret = float(pred_inv[0][i])
            last_price = float(last_prices[t])
            next_price = last_price * (1 + pred_ret)
            pred_prices[t] = next_price

            records.append({
                "Ticker": t,
                "Last_Close": last_price,
                "Predicted_Close": next_price,
                "Predicted_Return": pred_ret,
                "Predicted_%": pred_ret * 100
            })

            print(f"{t}: 마지막 종가={last_price:.2f} → 예측 종가={next_price:.2f} (예상 수익률 {pred_ret*100:.2f}%)")

        # 4. Monte Carlo Dropout 불확실성
        mean_pred, std_pred = get_std_pred(self.model, X_seq, n_samples=30, scaler_y=self.scaler_y)
        confidence = 1 / (std_pred + 1e-8)

        # 5. 결과 병합
        for i, r in enumerate(records):
            r["uncertainty"] = float(std_pred[i]) if len(std_pred) > 1 else float(std_pred[-1])
            r["confidence"] = float(confidence[i]) if len(confidence) > 1 else float(confidence[-1])

        pred_df = pd.DataFrame(records).round(4)
        self.pred_df = pred_df
        self.pred_prices = pred_prices

        print("\n================= 예측 결과 (표) =================")
        print(pred_df)

        print("\n================= 예측 결과 (값) =================")
        print(pred_prices)

        # 단일 티커일 경우 target 요약 제공
        target = Target(
            next_close=float(pred_df["Predicted_Close"].iloc[-1]),
            uncertainty=float(std_pred[-1]),
            confidence=float(pred_df["confidence"].iloc[-1])
        )


        return pred_prices, target


    # -------------------------------------------------------------
    # 5. 전체 실행 파이프라인
    # -------------------------------------------------------------
    def run_prediction(self):
        self.load_assets()               # 모델, 스케일러 등 불러오기
        self.fetch_macro_data()          # macro_df 불러오기
        X_seq = self.prepare_features()  # 입력 시퀀스 준비

        pred_prices, target = self.predictor(X_seq)

        # ✅ np.float64 → float 변환
        pred_prices = {k: float(v) for k, v in pred_prices.items()}

        return pred_prices, target, self.X_scaled



# -------------------------------------------------------------
# 실행 예시
# -------------------------------------------------------------
if __name__ == "__main__":
    predictor = MacroPredictor(
        base_date=datetime(2025, 10, 11),
        window=40,
        tickers=["AAPL"]  # ✅ 리스트 형태로 단일 티커 지정
    )
    pred_prices, target_json, _ = predictor.run_prediction()
