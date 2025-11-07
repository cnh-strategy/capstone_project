import json
from dataclasses import asdict
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

from config.agents import dir_info
from core.macro_classes.macro_sub import get_std_pred, MakeDatasetMacro
from core.macro_classes.macro_llm import AttributionAnalyzer, LLMExplainer, Opinion, Rebuttal
from agents.base_agent import BaseAgent, Target, StockData
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS


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
                 ,agent_id='MacroSentiAgent',
                 **kwargs):
        # super().__init__(agent_id)  # ✅ 부모 초기화 필수

        self.last_price = None
        self.stockdata = None
        self.window_size = None
        self.agent_id = agent_id
        BaseAgent.__init__(self, self.agent_id, **kwargs)
        self.model_path = f"{model_dir}/{ticker}_{agent_id}.keras"
        self.scaler_X_path = f"{model_dir}/scalers/{ticker}_{agent_id}_xscaler.pkl"
        self.scaler_y_path = f"{model_dir}/scalers/{ticker}_{agent_id}_yscaler.pkl"
        self.base_date = base_date
        self.window = window
        self.tickers = [ticker] #or ["AAPL", "MSFT", "NVDA", "TSLA"]
        # self.target_tickers = target_tickers or ["AAPL", "MSFT", "NVDA"]

        self.ticker = ticker
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
        print(f"model_path: {self.model_path}")
        self.model = load_model(self.model_path, compile=False)
        self.scaler_X = joblib.load(self.scaler_X_path)
        self.scaler_y = joblib.load(self.scaler_y_path)
        print("[OK] 모델 및 스케일러 로드 완료")

    # -------------------------------------------------------------
    # 2. MacroSentimentAgent로 최신 매크로 데이터 수집
    # -------------------------------------------------------------
    def fetch_macro_data(self):
        print("[INFO] MacroSentimentAgent 데이터 수집 중...")
        macro_agent = MakeDatasetMacro(base_date=self.base_date,
                                       window=self.window, target_tickers=self.tickers)
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
        return X_seq, X_scaled

    # -------------------------------------------------------------
    # 4. 예측 수행 및 결과 변환
    # -------------------------------------------------------------
    def m_predictor(self, X_seq):
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
            self.last_price = float(last_prices[t])
            next_price = self.last_price * (1 + pred_ret)
            pred_prices[t] = next_price

            records.append({
                "Ticker": t,
                "Last_Close": self.last_price,
                "Predicted_Close": next_price,
                "Predicted_Return": pred_ret,
                "Predicted_%": pred_ret * 100
            })

            print(f"{t}: 마지막 종가={self.last_price:.2f} → 예측 종가={next_price:.2f} (예상 수익률 {pred_ret*100:.2f}%)")

        # 4. Monte Carlo Dropout 불확실성
        mean_pred, std_pred, confidence, predicted_price = get_std_pred(
                            self.model, X_seq, n_samples=30, scaler_y=self.scaler_y)
        confidence = 1 / (std_pred + 1e-8)

        # 5. 결과 병합
        for i, r in enumerate(records):
            r["uncertainty"] = float(std_pred[i]) if len(std_pred) > 1 else float(std_pred[-1])
            r["confidence"] = float(confidence[i]) if len(confidence) > 1 else float(confidence[-1])

        pred_df = pd.DataFrame(records)
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


    #macro_reviewer_draft 역할
    def macro_reviewer_draft(self, X_scaled, pred_prices, target):
        # 예측 일자 설정
        base_date = datetime.today()

        macro_agent = MakeDatasetMacro(base_date=self.base_date,
                                       window=self.window, target_tickers=self.tickers)
        macro_agent.fetch_data()
        feature_df = macro_agent.add_features()
        feature_df = feature_df.tail(45).reset_index(drop=True)


        # -------------------------------------------------
        # (1) feature 순서 재정렬 (self.scaler_X 기준)
        # -------------------------------------------------
        scaler_X = joblib.load(self.scaler_X_path)

        expected_features = list(scaler_X.feature_names_in_)
        for col in expected_features:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

        feature_names = feature_df.columns.tolist()

        # ✅ numpy 변환 추가
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.to_numpy()

        # ✅ LSTM 입력은 3D여야 함 (1, 40, num_features)
        if X_scaled.ndim == 2:
            X_scaled = np.expand_dims(X_scaled, axis=0)


        # -------------------------------
        # 3️⃣ SHAP 계산
        # -------------------------------
        # --- (run() 안의 안전 처리) ---
        X_scaled = X_scaled.astype(np.float32)
        X_scaled = X_scaled[:, :, :300]
        feature_names = feature_names[:300]

        print("\n3️⃣ Calculating feature importance...")
        analyzer = AttributionAnalyzer(self.model)
        importance_dict, temporal_df, causal_df, interaction_df = analyzer.run_all_shap(X_scaled, feature_names)

        temporal_summary = temporal_df.head().to_dict(orient="records") if temporal_df is not None else []
        causal_summary = causal_df.to_dict(orient="records") if causal_df is not None else []
        if isinstance(interaction_df, pd.DataFrame):
            interaction_summary = interaction_df.iloc[:5, :5].to_dict()
        else:
            interaction_summary = {}


        # -------------------------------
        # 4️⃣ llm 생성
        # -------------------------------
        print("\n4️⃣ Generating explanation using LLM...")

        llm  = LLMExplainer()
        feature_summary = feature_df.tail(5).describe().to_dict()
        explanation = llm.generate_explanation(feature_summary, pred_prices, importance_dict,
                                               temporal_summary, causal_summary, interaction_summary)

        print(f"\n================= pred_prices:{pred_prices} =================")

        print("\n================= LLM Explanation =================")
        print(explanation)
        print("===================================================")

        total_json = {
            'agent_id' : self.agent_id,
            'target' : target,
            'reason' : explanation
        }

        self.stockdata = StockData(
            MacroSentiAgent={                              # ✅ 필드명은 self.agent_id와 동일해야 함
                'feature_importance': {
                    'feature_summary': feature_summary,
                    'importance_dict': importance_dict,
                    'temporal_summary': temporal_summary,
                    'causal_summary': causal_summary,
                    'interaction_summary': interaction_summary
                },
                'our_prediction': pred_prices,
                'uncertainty': round(target.uncertainty or 0.0, 8),
                'confidence': round(target.confidence or 0.0, 8)
            },
            last_price=self.last_price,
            currency="USD"
        )


        # 1) 메시지 생성 (→ 여기서 ctx가 만들어짐)
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)
        msg_sys = self._msg("system", sys_text)
        msg_user = self._msg("user",   user_text)

        parsed = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj_opinion)
        prompt_set = OPINION_PROMPTS.get(self.agent_id, OPINION_PROMPTS[self.agent_id])


        context = json.dumps({
            "agent_id": self.agent_id,
            "next_close": round(target.next_close, 3),
            "uncertainty_sigma": round(target.uncertainty or 0.0, 8),
            "confidence_beta": round(target.confidence or 0.0, 8),
            "latest_data": str(self.stockdata),
            "feature_importance": {
                'feature_summary': feature_summary,
                'importance_dict': importance_dict,
                'temporal_summary': temporal_summary,
                'causal_summary': causal_summary,
                'interaction_summary': interaction_summary
            },
        }, ensure_ascii=False, indent=2)

        sys_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=context)
        print(f"\n sys_text:{sys_text} \n")
        print(f" user_text:{user_text} \n")

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")

        # reason = explanation

        # 4) Opinion 기록/반환 (항상 최신 값 append)
        self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason))


        return total_json, self.opinions[-1]


    # LLM Reasoning 메시지
    def _build_messages_opinion(self, stock_data, target):
        """ LLM 프롬프트 메시지 구성 """
        print(f"build messages opinion - {self.agent_id}")

        # ✅ 해당 agent_id의 데이터 가져오기
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 데이터가 필요함")

        # ✅ dataclass를 dict로 변환
        stock_data_dict = asdict(stock_data)

        # ✅ feature_importance는 agent_data 내부에서 가져오기
        feature_imp = agent_data.get("feature_importance", {})

        # ✅ context 구성
        ctx = {
            "agent_id": self.agent_id,
            "ticker": stock_data_dict.get("ticker", "Unknown"),
            "currency": stock_data_dict.get("currency", "USD"),
            "last_price": stock_data_dict.get("last_price", None),
            "our_prediction": float(target.next_close),     #our_prediction = next_close
            "uncertainty": float(target.uncertainty),
            "confidence": float(target.confidence),

            "feature_importance": {
                "feature_summary": feature_imp.get("feature_summary", []),
                "importance_dict": feature_imp.get("importance_dict", []),
                "temporal_summary": feature_imp.get("temporal_summary", []),
                "causal_summary": feature_imp.get("causal_summary", []),
                "interaction_summary": feature_imp.get("interaction_summary", {}),
            },
        }

        print(f"\n ctx:{ctx} \n")

        # 각 컬럼별 최근 시계열 그대로 포함
        # (최근 7~14일 정도면 LLM이 이해 가능한 범위)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[self.window_size:]  # 최근 14일치 전체 시계열
            else:
                ctx[col] = [values]

        # 프롬프트 구성
        system_text = OPINION_PROMPTS[self.agent_id]["system"]
        user_text = OPINION_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        # user_text = OPINION_PROMPTS[self.agent_id]["user"].format(**ctx)

        return system_text, user_text






    def _build_messages_rebuttal(self,
                                 my_opinion: Opinion,
                                 target_opinion: Opinion,
                                 stock_data: StockData) -> tuple[str, str]:

        t = stock_data.ticker or "UNKNOWN"
        ccy = (stock_data.currency or "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        ctx = {
            "ticker": t,
            "currency": ccy,
            "data_summary": getattr(stock_data, self.agent_id, {}).get("feature_cols", []),
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:2000],
                "uncertainty": float(my_opinion.target.uncertainty),
                "confidence": float(my_opinion.target.confidence),
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
                "uncertainty": float(target_opinion.target.uncertainty),
                "confidence": float(target_opinion.target.confidence),
            }
        }
        # 각 컬럼별 최근 시계열 그대로 포함
        # (최근 7~14일 정도면 LLM이 이해 가능한 범위)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[self.window_size:]  # 최근 14일치 전체 시계열
            else:
                ctx[col] = [values]

        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        user_text   = REBUTTAL_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text

    def _build_messages_revision(
            self,
            my_opinion: Opinion,
            others: List[Opinion],
            rebuttals: Optional[List[Rebuttal]] = None,
            stock_data: StockData = None,
    ) -> tuple[str, str]:
        """
        Revision용 LLM 메시지 생성기
        - 내 의견(my_opinion), 타 에이전트 의견(others), 주가데이터(stock_data) 기반
        - rebuttals 중 나(self.agent_id)를 대상으로 한 내용만 포함
        """
        # 기본 메타데이터
        t = getattr(stock_data, "ticker", "UNKNOWN")
        ccy = getattr(stock_data, "currency", "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        # 타 에이전트 의견 및 rebuttal 통합 요약
        others_summary = []
        for o in others:
            entry = {
                "agent_id": o.agent_id,
                "predicted_price": float(o.target.next_close),
                "confidence": float(o.target.confidence),
                "uncertainty": float(o.target.uncertainty),
                "reason": str(o.reason)[:500],
            }

            # 나에게 온 rebuttal만 stance/message 추출
            if rebuttals:
                related_rebuts = [
                    {"stance": r.stance, "message": r.message}
                    for r in rebuttals
                    if r.from_agent_id == o.agent_id and r.to_agent_id == self.agent_id
                ]
                if related_rebuts:
                    entry["rebuttals_to_me"] = related_rebuts

            others_summary.append(entry)

        # Context 구성
        ctx = {
            "ticker": t,
            "currency": ccy,
            "agent_type": self.agent_id,
            "my_opinion": {
                "predicted_price": float(my_opinion.target.next_close),
                "confidence": float(my_opinion.target.confidence),
                "uncertainty": float(my_opinion.target.uncertainty),
                "reason": str(my_opinion.reason)[:1000],
            },
            "others_summary": others_summary,
            "data_summary": getattr(stock_data, self.agent_id, {}).get("feature_cols", []),
        }

        # 최근 시계열 데이터 포함 (기술/심리적 패턴)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-14:]  # 최근 14일치
            else:
                ctx[col] = [values]

        # Prompt 구성
        prompt_set = REVISION_PROMPTS.get(self.agent_id)
        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

        return system_text, user_text