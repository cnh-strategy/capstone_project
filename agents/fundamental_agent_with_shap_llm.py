import os
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import pickle

import shap
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

from agents.dump import CAPSTONE_OPENAI_API
from fundamental_sub import MacroSentimentAgent
from openai import OpenAI

# ==============================================================
# 1️⃣ LLM 기반 설명 모듈
# ==============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 환경 변수 이름 수정

class LLMExplainer:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=CAPSTONE_OPENAI_API)
        self.model = model_name

    def generate_explanation(self, feature_summary, predictions, importance_summary):
        prompt = f"""
        당신은 금융 시장을 분석하는 인공지능 애널리스트입니다.

        아래는 모델 입력과 출력에 대한 요약 정보입니다.
        (1) 각 주식의 최근 입력 특성 요약 (마지막 5일 평균)
        (2) 모델이 예측한 다음날 종가
        (3) 주요 변수 중요도 (상위 10개)

        ### 최근 입력 특성 요약 (5일 평균)
        {feature_summary}

        ### 모델 예측값 (다음날 종가 예측)
        {predictions}

        ### 변수 중요도 (상위 10개 특징)
        {importance_summary}

        위 정보를 바탕으로, 모델이 이러한 주가 변동을 예측한 이유를
        금융적 관점에서 분석적으로 설명해 주세요.

        특히 다음 사항을 포함해 주십시오.
        - 금리, 원자재, 변동성 지수 등 거시 요인이 주가에 미치는 영향
        - 기술적 지표(추세, 거래량, 변동성 등)와의 관련성
        - 각 종목별(AAPL, MSFT, NVDA)로 나누어 간결하면서도 분석적인 설명
        - 어떤 변수가 AAPL, MSFT, NVDA의 예측에 각각 더 크게 작용했는가
        - 중요 피처 상위 3개가 종목마다 어떻게 다른지
        - 동일한 변수라도 종목별 영향 방향이 달랐다면 그 이유

        각 종목별로, 모델이 해당 방향의 예측을 한 주요 원인과
        그 배경 논리를 논리적으로 기술해 주세요.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()


# ==============================================================
# 2️⃣ Permutation Importance 기반 feature attribution
# ==============================================================
class AttributionAnalyzer:
    def __init__(self, model):
        self.model = model

    #종목별 중요도 따로 계산
    def compute_feature_importance(self, X_scaled, feature_names):
        print("⚙️ Calculating feature importance using SHAP (final stabilized version)...")

        try:
            # ✅ 1️⃣ 모델 복제 (원본 상태 보호)
            model_copy = tf.keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # ✅ 2️⃣ 입력 형태 변환
            time_steps = X_scaled.shape[1]
            num_features = X_scaled.shape[2]
            X_flat = X_scaled.reshape(X_scaled.shape[0], -1).astype(np.float32)
            background = X_flat[:min(10, len(X_flat))]

            # ✅ 3️⃣ SHAP용 wrapper
            def model_wrapper(X):
                X = np.array(X).astype(np.float32)
                X_reshaped = X.reshape(X.shape[0], time_steps, num_features)
                preds = model_copy.predict(X_reshaped, verbose=0)
                return preds  # 다중 출력 (AAPL/MSFT/NVDA)

            # ✅ 4️⃣ KernelExplainer
            explainer = shap.KernelExplainer(model_wrapper, background)
            shap_values = explainer.shap_values(X_flat[:5])

            # ✅ 5️⃣ 중요도 계산
            tickers = ["AAPL", "MSFT", "NVDA"]
            importance_dict = {}
            for i, ticker in enumerate(tickers):
                mean_importance = np.abs(shap_values[i]).mean(axis=0)
                df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": mean_importance
                }).sort_values("importance", ascending=False)
                importance_dict[ticker] = df.head(10).to_dict(orient="records")

        except Exception as e:
            print("⚠️ SHAP computation failed, falling back to permutation importance.")
            print(f"   Error details: {e}")

            # fallback: permutation importance
            X_flat = X_scaled.reshape(-1, X_scaled.shape[-1])
            y_pred = self.model.predict(X_scaled).flatten()
            importance_values = np.std(X_flat, axis=0) * np.mean(np.abs(y_pred))
            df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance_values
            }).sort_values("importance", ascending=False)
            top_features = df.head(10).to_dict(orient="records")
            return {"AAPL": top_features, "MSFT": top_features, "NVDA": top_features}



# ==============================================================
# 3️⃣ 메인 예측 에이전트
# ==============================================================
class FundamentalForecastAgent:
    def __init__(self):
        self.model_path = "models/multi_output_lstm_model.h5"
        self.scaler_x_path = "models/scaler_X.pkl"
        self.scaler_y_path = "models/scaler_y.pkl"

        self.model = load_model(self.model_path, compile=False)

        self.scaler_X = joblib.load(self.scaler_x_path)
        self.scaler_y = joblib.load(self.scaler_y_path)

        if isinstance(self.scaler_X, np.ndarray):
            print("⚠️ Detected ndarray instead of scaler — new MinMaxScaler will be fitted at runtime.")
            self.scaler_X = MinMaxScaler()
        if isinstance(self.scaler_y, np.ndarray):
            print("⚠️ Detected ndarray instead of scaler — new MinMaxScaler will be fitted at runtime.")
            self.scaler_y = MinMaxScaler()

    def run(self):
        print("1️⃣ Collecting macro & stock features...")
        # 예측 일자 설정
        # base_date = datetime.today()
        base_date = datetime(2025, 10, 11)

        macro_agent = MacroSentimentAgent(base_date)
        macro_agent.fetch_data()
        feature_df = macro_agent.add_features()
        feature_df = feature_df.tail(45).reset_index(drop=True)  # 기본 agent와 동일한 데이터 윈도우 (40 + margin 5)


        # -------------------------------------------------
        # (1) feature 순서 재정렬 (self.scaler_X 기준)
        # -------------------------------------------------
        expected_features = list(self.scaler_X.feature_names_in_)
        for col in expected_features:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

        feature_names = feature_df.columns.tolist()


        # -------------------------------------------------
        # (2) 스케일링 및 윈도우 구성
        # -------------------------------------------------
        X_scaled = self.scaler_X.transform(feature_df.values)
        window = 40
        X_scaled = np.expand_dims(X_scaled[-window:], axis=0)

        # feature_df와 macro_agent.data의 인덱스를 맞추기 위해 최신 40일만 사용
        macro_agent.data = macro_agent.data.iloc[-len(feature_df):].reset_index(drop=True)


        # -------------------------------
        # 2️⃣ 예측 수행 및 역변환
        # -------------------------------
        print("2️⃣ Predicting next-day prices...")
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred_inv = self.scaler_y.inverse_transform(y_pred_scaled)
        y_pred = y_pred_inv[0]


        # 최근 종가 (이동평균 기반)
        last_prices = {
            "AAPL": feature_df["AAPL_ma5"].iloc[-1],
            "MSFT": feature_df["MSFT_ma5"].iloc[-1],
            "NVDA": feature_df["NVDA_ma5"].iloc[-1],
        }

        # 종가 복원
        pred_prices = {}
        for i, ticker in enumerate(["AAPL", "MSFT", "NVDA"]):
            next_price = np.float64(last_prices[ticker]) * (1 + np.float64(y_pred[i]))
            pred_prices[ticker] = round(float(next_price), 2)


        print("\nPredicted next-day closing prices (actual scale):")
        for t in ["AAPL", "MSFT", "NVDA"]:
            pct = (pred_prices[t] - last_prices[t]) / last_prices[t] * 100
            print(f"  {t}: {pred_prices[t]} (Predicted return {pct:.2f}%)")

        # -------------------------------
        # 3️⃣ SHAP 계산
        # -------------------------------
        print("\n3️⃣ Calculating feature importance...")
        analyzer = AttributionAnalyzer(self.model)
        importance_dict = analyzer.compute_feature_importance(X_scaled, feature_names)
        shap_summary = {
            k: [{d["feature"]: round(d["importance"], 4)} for d in v]
            for k, v in importance_dict.items()
        }


        # -------------------------------
        # 4️⃣ llm 생
        # -------------------------------
        print("\n4️⃣ Generating explanation using LLM...")
        explanation=None
        llm = LLMExplainer()
        feature_summary = feature_df.tail(5).describe().round(3).to_dict()
        explanation = llm.generate_explanation(feature_summary, pred_prices, shap_summary)

        print(f"\n================= pred_prices:{pred_prices} =================")

        print("\n================= LLM Explanation =================")
        print(explanation)
        print("===================================================")

        return pred_prices, explanation, shap_summary



if __name__ == "__main__":
    agent = FundamentalForecastAgent()
    predictions, llm_explanation, shap_info = agent.run()
