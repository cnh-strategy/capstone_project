import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import shap
from openai import OpenAI

from agents.dump_keys import CAPSTONE_OPENAI_API
from typing import Dict, List, Optional, Literal, Tuple
from collections import defaultdict

# -----------------------------
# 데이터 구조 정의
# -----------------------------
@dataclass
class Target:
    """예측 목표값 + 불확실성 정보 포함
    - next_close: 다음 거래일 종가 예측치
    - uncertainty: Monte Carlo Dropout 기반 예측 표준편차(σ)
    - confidence: 모델 신뢰도 β (정규화된 신뢰도; 선택적)
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    feature_cols: Optional[List[str]] = None
    importances: Optional[List[float]] = None

@dataclass
class Opinion:
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str

@dataclass
class RoundLog:
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    agent_id: str = ""
    ticker: str = ""
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    feature_cols: Optional[List[str]] = None
    last_price: Optional[float] = None
    technical: Optional[Dict] = None

    def __post_init__(self):
        if self.last_price is None:
            self.last_price = 100.0


# ==============================================================
# 1️⃣ LLM 기반 설명 모듈 (확장형)
# ==============================================================
class LLMExplainer:
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model_name="gpt-4o-mini",
                 model: Optional[str] = None,
                 preferred_models: Optional[List[str]] = None,
                 temperature: float = 0.2,
                 verbose: bool = False,
                 need_training: bool = True):
        self.client = OpenAI(api_key=CAPSTONE_OPENAI_API)
        self.model = model_name

        self.agent_id = 'MacroSentiAgent'
        self.temperature = temperature # Temperature 설정
        self.verbose = verbose            # 디버깅 모드
        self.need_training = need_training # 모델 학습 필요 여부
        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [
                m for m in self.preferred_models if m != model
            ]

        # API 키 로드
        self.api_key = CAPSTONE_OPENAI_API
        if not self.api_key:
            raise RuntimeError("환경변수 CAPSTONE_OPENAI_API가 설정되지 않았습니다.")

        # 공통 헤더
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 상태값
        self.stockdata: Optional[StockData] = None
        self.opinions: List[Opinion] = []
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)

        # JSON Schema
        self.schema_obj_opinion = {
            "type": "object",
            "properties": {
                "next_close": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["next_close", "reason"],
            "additionalProperties": False,
        }
        self.schema_obj_rebuttal = {
            "type": "object",
            "properties": {
                "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                "message": {"type": "string"},
            },
            "required": ["stance", "message"],
            "additionalProperties": False,
        }


    def generate_explanation(
            self,
            feature_summary,
            predictions,
            importance_summary,
            temporal_summary=None,
            causal_summary=None,
            interaction_summary=None,
            stock_data=None,
            target=None,
    ):
        """
        SHAP / LSTM 결과 기반으로 LLM Reasoning 생성 (system + user 구조)
        """

        # 1️⃣ system 메시지
        sys_text = (
            "너는 금융 시장을 분석하는 인공지능 애널리스트이며, "
            "LSTM 기반의 시계열 모델 예측 결과를 해석해야 한다. "
            "모델의 예측값, 변수 중요도, 인과 관계, 상호작용 정보를 종합하여 논리적 금융 분석을 수행한다."
        )

        # 2️⃣ user 메시지 (LSTM 예측 결과 중심)
        user_text = f"""
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
    
        ---
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
        
        ---
        추가 맥락:
        최근 종가: {getattr(stock_data, 'last_price', 'N/A')}
        예측 종가: {getattr(target, 'next_close', 'N/A')}
        """

        # 3️⃣ 메시지 빌드 (system + user)
        msg_sys = self._msg("system", sys_text)
        msg_user = self._msg("user", user_text)

        # 4️⃣ LLM 호출
        parsed = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj_opinion)
        reason = parsed.get("reason") or "(사유 생성 실패: 미입력)"

        return reason



    #[base_agent.py]
    def _msg(self, role: str, content: str) -> dict:
        """OpenAI ChatCompletion용 메시지 구조 생성"""
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}


    #[base_agent.py] OpenAI API 호출
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """Chat Completions API 호출 (fallback 지원)"""
        last_err = None
        for model in self.preferred_models:
            payload = {
                "model": model,
                "messages": [msg_sys, msg_user],
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Response",
                        "schema": schema_obj
                    }
                }
            }
            try:
                import requests
                r = requests.post(self.OPENAI_URL, headers=self.headers, json=payload, timeout=120)
                if r.ok:
                    data = r.json()
                    # 최신 Chat API의 응답 처리
                    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not msg:
                        continue
                    try:
                        return json.loads(msg)
                    except Exception:
                        return {"reason": msg.strip()}
                else:
                    last_err = (r.status_code, r.text)
                    continue
            except Exception as e:
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    def _p(self, msg: str):
        if self.verbose:
            print(f"[{self.agent_id}] {msg}")




# ==============================================================
# Base SHAP (종목별 기본 중요도)
# ==============================================================
class BaseSHAPAnalyzer:
    def __init__(self, model):
        self.model = model

    def compute_feature_importance(self, X_scaled, feature_names):
        X_scaled = np.array(X_scaled, dtype=np.float32)
        time_steps, num_features = X_scaled.shape[1], X_scaled.shape[2]

        # baseline 다양화
        X_flat = X_scaled.reshape(X_scaled.shape[0], -1).astype(np.float32)
        background_mean = X_flat.mean(axis=0)
        background_noise = background_mean + np.random.normal(0, 0.03, size=X_flat.shape[1])
        background = np.vstack([background_mean, background_noise]).astype(np.float32)

        print(f"[SHAP] Computing Base SHAP for AAPL (모델 입력 {num_features}개 피처)")

        def model_wrapper_single(X):
            X = np.array(X).astype(np.float32)
            X = X.reshape(X.shape[0], time_steps, num_features)
            return self.model.predict(X, verbose=0)[:, 0]

        # ✅ SamplingExplainer로 변경 (메모리 절약)
        explainer = shap.KernelExplainer(model_wrapper_single, background)

        # ✅ 샘플 수 제한
        sample_idx = np.random.choice(len(X_flat), size=min(3, len(X_flat)), replace=False)
        shap_values = explainer.shap_values(X_flat[sample_idx], nsamples=30)
        shap_values = np.array(shap_values).reshape(-1, time_steps, num_features)

        mean_importance = np.abs(shap_values).mean(axis=(0, 1))
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_importance
        })

        # ✅ AAPL 관련 + 매크로 피처만 남기기
        selected_features = [f for f in feature_names if "AAPL_" in f or not any(t in f for t in ["MSFT_", "NVDA_"])]
        df = df[df["feature"].isin(selected_features)]
        df = df.sort_values("importance", ascending=False)

        importance_dict = {"AAPL": df.head(10).to_dict(orient="records")}
        print(f"[OK] SHAP 계산 완료: {len(df)}개 feature 중 상위 10개 추출")
        return importance_dict



# ==============================================================
# Temporal SHAP (시점별 영향도)
# ==============================================================
class TemporalSHAPAnalyzer:
    def __init__(self, model):
        self.model = model

    def compute_temporal_shap(self, X_scaled, feature_names, target_idx=0):
        print("[Causal SHAP] Computing temporal perturbation effects...")
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
        sample_idx = np.random.choice(len(X_flat), size=min(5, len(X_flat)), replace=False)
        shap_values = explainer.shap_values(X_flat[sample_idx], nsamples=30)
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

        X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
        background_mean = X_flat.mean(axis=0)
        background_noise = background_mean + np.random.normal(0, 0.03, size=X_flat.shape[1])
        background = np.vstack([background_mean, background_noise, X_flat[:3]])

        def model_wrapper(X):
            X = np.array(X).astype(np.float32)
            X = X.reshape(X.shape[0], time_steps, num_features)
            return self.model.predict(X, verbose=0)[:, target_idx]

        explainer = shap.KernelExplainer(model_wrapper, background)

        sample_idx = np.random.choice(len(X_flat), size=min(3, len(X_flat)), replace=False)
        shap_values = explainer.shap_values(X_flat[sample_idx], nsamples=10)
        shap_values = np.array(shap_values).reshape(-1, time_steps, num_features)

        # ✅ 여기까진 full feature (189)로 계산

        shap_matrix = shap_values.reshape(-1, num_features).astype(np.float32)

        valid_mask = shap_matrix.std(axis=0) > 1e-6
        shap_matrix = shap_matrix[:, valid_mask]
        valid_features = np.array(feature_names)[valid_mask]

        if shap_matrix.shape[1] > 100:
            print(f"[Interaction SHAP] Reducing features for correlation: {shap_matrix.shape[1]} → 100")
            shap_matrix = shap_matrix[:, :100]
            valid_features = valid_features[:100]

        if shap_matrix.shape[1] < 2:
            print("⚠️ Not enough variance for interaction computation.")
            return pd.DataFrame()

        print(f"[Interaction SHAP] Computing correlation among {shap_matrix.shape[1]} features...")
        inter_corr = np.corrcoef(shap_matrix, rowvar=False)
        inter_df = pd.DataFrame(inter_corr, index=valid_features, columns=valid_features).round(3)
        return inter_df



class AttributionAnalyzer:
    """통합 SHAP 해석 레이어"""
    def __init__(self, model):
        self.model = model

    def run_all_shap(self, X_scaled, feature_names):
        base_result, temporal_df, causal_df, interaction_df = {},{},{},{}

        # Base SHAP
        try:
            base = BaseSHAPAnalyzer(self.model)
            base_result = base.compute_feature_importance(X_scaled, feature_names)
        except Exception as e:
            print(f"[⚠️ Base SHAP Error]: {e}")

        # Temporal SHAP
        try:
            temporal = TemporalSHAPAnalyzer(self.model)
            temporal_df = temporal.compute_temporal_shap(X_scaled, feature_names, target_idx=0)
        except Exception as e:
            print(f"[⚠️ Temporal SHAP Error]: {e}")

        # Causal SHAP
        try:
            causal = CausalSHAPAnalyzer(self.model)
            causal_df = causal.compute_causal_shap(X_scaled, feature_names, target_idx=0)
        except Exception as e:
            print(f"[⚠️ Causal SHAP Error]: {e}")

        # Interaction SHAP
        try:
            interaction = InteractionSHAPAnalyzer(self.model)
            interaction_df = interaction.compute_interaction_importance(X_scaled, feature_names, target_idx=0)
        except Exception as e:
            print(f"[⚠️ Interaction SHAP Error]: {e}")

        return base_result, temporal_df, causal_df, interaction_df
