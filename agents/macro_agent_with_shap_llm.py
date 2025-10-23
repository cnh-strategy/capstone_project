import os
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model
from openai import OpenAI
from datetime import datetime
import joblib

from agents.dump import CAPSTONE_OPENAI_API
from agents.macro_agent import MarketPredictor
from macro_sub import MacroSentimentAgent


# ==============================================================
# 1️⃣ LLM 기반 설명 모듈 (확장형)
# ==============================================================
class LLMExplainer:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=CAPSTONE_OPENAI_API)
        self.model = model_name

    def generate_explanation(self, feature_summary, predictions, importance_summary,
                             temporal_summary=None, causal_summary=None, interaction_summary=None):
        prompt = f"""
    당신은 금융 시장을 분석하는 인공지능 애널리스트입니다.
    아래는 LSTM 기반 예측 모델의 해석 결과입니다.
    주어진 데이터를 정량적으로 해석하고, 변수 간 관계를 논리적으로 분석하세요.

    ### 1. 모델 예측 결과
    {predictions}

    ### 2. 종목별 주요 변수 중요도 (Base SHAP)
    {importance_summary}

    ### 3. 시점별 변수 영향 변화 (Temporal SHAP)
    {temporal_summary}

    ### 4. 변수별 인과 효과 (Causal SHAP)
    {causal_summary}

    ### 5. 변수 간 상호작용 행렬 (Interaction SHAP)
    {interaction_summary}

    위 데이터를 바탕으로 아래 내용을 체계적으로 작성하세요:

    (1) **Temporal 분석:** 
        - 어떤 변수들이 최근 시점으로 갈수록 영향력이 커졌는가?
        - 영향이 약해진 변수는 무엇인가?
        - 시간 흐름에 따라 피처 영향이 달라진 이유를 금융적 관점에서 설명.

    (2) **Causal 분석:** 
        - causal_effect가 양(+)이면 주가 상승 요인, 음(-)이면 하락 요인으로 해석.
        - 각 종목별로 어떤 피처가 인과적으로 강한 영향을 미쳤는가?
        - 예: “금리 상승(+) → 달러 강세 → 기술주 약세” 형태로 인과 구조를 제시.

    (3) **Interaction 분석:** 
        - 상관도가 높은 피처 쌍을 찾아 그 상호작용을 해석.
        - 예: “유가와 금리 동반 상승 → 비용 압박 증가 → AAPL/MSFT 부정적 영향”.

    (4) **종합 결론:** 
        - 세 가지 관점을 통합하여 각 종목(AAPL, MSFT, NVDA)의 예측 방향과 원인을 설명.
        - 특정 종목이 타 종목 대비 어떤 변수에 더 민감했는지 논리적으로 요약.

    분석 시 단순 설명이 아니라, 위 수치들을 근거로 금융적 추론을 포함해 주세요.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()


# ==============================================================
# Base SHAP (종목별 기본 중요도)
# ==============================================================
class BaseSHAPAnalyzer:
    def __init__(self, model):
        self.model = model

    def compute_feature_importance(self, X_scaled, feature_names):
        X_scaled = np.array(X_scaled, dtype=np.float32)
        time_steps, num_features = X_scaled.shape[1], X_scaled.shape[2]

        # ✅ baseline 다양화 (mean + noise)
        X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
        background_mean = X_flat.mean(axis=0)
        background_noise = background_mean + np.random.normal(0, 0.03, size=X_flat.shape[1])
        background = np.vstack([background_mean, background_noise, X_flat[:5]])

        tickers = ["AAPL", "MSFT", "NVDA"]
        importance_dict = {}

        for i, ticker in enumerate(tickers):
            print(f"[SHAP] Computing Base SHAP for {ticker}...")

            def model_wrapper_single(X):
                X = np.array(X).astype(np.float32)
                X = X.reshape(X.shape[0], time_steps, num_features)
                return self.model.predict(X, verbose=0)[:, i]

            explainer = shap.KernelExplainer(model_wrapper_single, background)

            # ✅ 더 많은 샘플 반영
            sample_idx = np.random.choice(len(X_flat), size=min(25, len(X_flat)), replace=False)
            shap_values = explainer.shap_values(X_flat[sample_idx])

            shap_values = np.array(shap_values)
            shap_values = shap_values.reshape(-1, time_steps, num_features)
            mean_importance = np.abs(shap_values).mean(axis=(0, 1))

            df = pd.DataFrame({
                "feature": feature_names,
                "importance": mean_importance
            }).sort_values("importance", ascending=False)
            importance_dict[ticker] = df.head(10).to_dict(orient="records")

        return importance_dict


# ==============================================================
# Temporal SHAP (시점별 영향도)
# ==============================================================
class TemporalSHAPAnalyzer:
    def __init__(self, model):
        self.model = model

    def compute_temporal_shap(self, X_scaled, feature_names, target_idx=0):
        X_scaled = np.array(X_scaled, dtype=np.float32)
        time_steps, num_features = X_scaled.shape[1], X_scaled.shape[2]

        # ✅ baseline 다양화
        X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
        background_mean = X_flat.mean(axis=0)
        background_noise = background_mean + np.random.normal(0, 0.03, size=X_flat.shape[1])
        background = np.vstack([background_mean, background_noise, X_flat[:5]])

        def model_wrapper(X):
            X = np.array(X).astype(np.float32)
            X = X.reshape(X.shape[0], time_steps, num_features)
            return self.model.predict(X, verbose=0)[:, target_idx]

        explainer = shap.KernelExplainer(model_wrapper, background)

        # ✅ 더 많은 시점 샘플 사용
        sample_idx = np.random.choice(len(X_flat), size=min(30, len(X_flat)), replace=False)
        shap_values = explainer.shap_values(X_flat[sample_idx])
        shap_values = np.array(shap_values).reshape(-1, time_steps, num_features)

        temporal_importance = np.abs(shap_values).mean(axis=0)
        temporal_df = pd.DataFrame(temporal_importance, columns=feature_names)
        temporal_df["time_step"] = np.arange(1, time_steps + 1)
        return temporal_df


# ==============================================================
# Causal SHAP (인과 효과 근사)
# ==============================================================
class CausalSHAPAnalyzer:
    def __init__(self, model):
        self.model = model

    def compute_causal_shap(self, X_scaled, feature_names, target_idx=0):
        print("[Causal SHAP] Computing causal perturbation effects...")

        X_scaled = np.array(X_scaled, dtype=np.float32)
        baseline_pred = self.model.predict(X_scaled, verbose=0)[:, target_idx].mean()

        effects = []
        for j, feat in enumerate(feature_names):
            perturbed = X_scaled.copy()
            perturb_factor = np.random.uniform(1.1, 1.3)  # ✅ 더 강한 perturbation
            perturbed[:, :, j] *= perturb_factor
            new_pred = self.model.predict(perturbed, verbose=0)[:, target_idx].mean()
            effects.append(new_pred - baseline_pred)

        df = pd.DataFrame({"feature": feature_names, "causal_effect": effects})
        df["abs_effect"] = df["causal_effect"].abs()
        df = df.sort_values("abs_effect", ascending=False)
        return df.head(10)


# ==============================================================
# Interaction SHAP (상호작용 근사)
# ==============================================================
class InteractionSHAPAnalyzer:
    def __init__(self, model):
        self.model = model

    def compute_interaction_importance(self, X_scaled, feature_names, target_idx=0):
        print("[Interaction SHAP] Computing interaction correlations...")

        X_scaled = np.array(X_scaled, dtype=np.float32)
        time_steps, num_features = X_scaled.shape[1], X_scaled.shape[2]

        # baseline 다양화
        X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
        background_mean = X_flat.mean(axis=0)
        background_noise = background_mean + np.random.normal(0, 0.03, size=X_flat.shape[1])
        background = np.vstack([background_mean, background_noise, X_flat[:5]])

        def model_wrapper(X):
            X = np.array(X).astype(np.float32)
            X = X.reshape(X.shape[0], time_steps, num_features)
            return self.model.predict(X, verbose=0)[:, target_idx]

        explainer = shap.KernelExplainer(model_wrapper, background)

        # ✅ 더 많은 샘플 확보
        sample_idx = np.random.choice(len(X_flat), size=min(25, len(X_flat)), replace=False)
        shap_values = explainer.shap_values(X_flat[sample_idx])
        shap_values = np.array(shap_values).reshape(-1, time_steps, num_features)

        shap_matrix = shap_values.reshape(-1, num_features)
        # ✅ NaN 방지
        valid_mask = shap_matrix.std(axis=0) > 1e-6
        shap_matrix = shap_matrix[:, valid_mask]

        if shap_matrix.shape[1] < 2:
            print("⚠️ Not enough variance for interaction computation.")
            return pd.DataFrame()

        # ✅ 상호작용 상관 행렬
        inter_corr = np.corrcoef(shap_matrix, rowvar=False)
        inter_df = pd.DataFrame(inter_corr, index=np.array(feature_names)[valid_mask],
                                columns=np.array(feature_names)[valid_mask])
        inter_df = inter_df.round(3)
        return inter_df



class AttributionAnalyzer:
    """통합 SHAP 해석 레이어"""
    def __init__(self, model):
        self.model = model

    def run_all_shap(self, X_scaled, feature_names):
        base = BaseSHAPAnalyzer(self.model)
        base_result = base.compute_feature_importance(X_scaled, feature_names)

        temporal = TemporalSHAPAnalyzer(self.model)
        temporal_df = temporal.compute_temporal_shap(X_scaled, feature_names, target_idx=0)

        causal = CausalSHAPAnalyzer(self.model)
        causal_df = causal.compute_causal_shap(X_scaled, feature_names, target_idx=0)

        interaction = InteractionSHAPAnalyzer(self.model)
        interaction_df = interaction.compute_interaction_importance(X_scaled, feature_names, target_idx=0)

        return base_result, temporal_df, causal_df, interaction_df


# ==============================================================
# 3️⃣ 메인 예측 에이전트 (통합형)
# ==============================================================
class FundamentalForecastAgent:
    def __init__(self):
        self.model_path = "models/multi_output_lstm_model.h5"
        self.scaler_x_path = "models/scaler_X.pkl"
        self.scaler_y_path = "models/scaler_y.pkl"

        self.model = load_model(self.model_path, compile=False)
        self.scaler_X = joblib.load(self.scaler_x_path)
        self.scaler_y = joblib.load(self.scaler_y_path)

    def run(self):
        print("1️⃣ Collecting macro & stock features...")
        # 예측 일자 설정
        # base_date = datetime.today()
        base_date = datetime(2025, 10, 11)

        macro_agent = MacroSentimentAgent(base_date)
        macro_agent.fetch_data()
        feature_df = macro_agent.add_features()
        feature_df = feature_df.tail(45).reset_index(drop=True)


        # -------------------------------------------------
        # (1) feature 순서 재정렬 (self.scaler_X 기준)
        # -------------------------------------------------
        expected_features = list(self.scaler_X.feature_names_in_)
        for col in expected_features:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

        feature_names = feature_df.columns.tolist()


        # -------------------------------
        # 2️⃣ 예측 수행 및 역변환
        # -------------------------------
        predictor = MarketPredictor(
            base_date=base_date,
            window=40,
            tickers=["AAPL", "MSFT", "NVDA"]
        )
        pred_prices, X_scaled = predictor.run_prediction()

        # ✅ numpy 변환 추가
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.to_numpy()

        # ✅ LSTM 입력은 3D여야 함 (1, 40, num_features)
        if X_scaled.ndim == 2:
            X_scaled = np.expand_dims(X_scaled, axis=0)


        # -------------------------------
        # 3️⃣ SHAP 계산
        # -------------------------------
        print("\n3️⃣ Calculating feature importance...")
        analyzer = AttributionAnalyzer(self.model)
        importance_dict, temporal_df, causal_df, interaction_df = analyzer.run_all_shap(X_scaled, feature_names)

        temporal_summary = temporal_df.head().to_dict(orient="records")
        causal_summary = causal_df.to_dict(orient="records")
        interaction_summary = interaction_df.iloc[:5, :5].round(3).to_dict()

        # -------------------------------
        # 4️⃣ llm 생
        # -------------------------------
        print("\n4️⃣ Generating explanation using LLM...")
        explanation=None
        llm = LLMExplainer()
        feature_summary = feature_df.tail(5).describe().round(3).to_dict()
        explanation = llm.generate_explanation(feature_summary, pred_prices, importance_dict,
                                               temporal_summary, causal_summary, interaction_summary)

        print(f"\n================= pred_prices:{pred_prices} =================")

        print("\n================= LLM Explanation =================")
        print(explanation)
        print("===================================================")

        return pred_prices, explanation, importance_dict, temporal_df, causal_df, interaction_df


if __name__ == "__main__":
    agent = FundamentalForecastAgent()
    predictions, llm_explanation, shap_info, temporal_df, causal_df, interaction_df = agent.run()
