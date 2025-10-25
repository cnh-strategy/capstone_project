# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
from collections import defaultdict
import os, json, time, requests, yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS

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
    sentimental: Dict
    fundamental: Dict
    technical: Dict
    last_price: Optional[float] = None
    currency: Optional[str] = None

# ===============================================================
# BaseAgent 클래스
# ===============================================================
class BaseAgent:
    """LLM 기반 Multi-Agent Debate 공통 클래스"""

    OPENAI_URL = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        agent_id: str,
        model: Optional[str] = None,
        preferred_models: Optional[List[str]] = None,
        temperature: float = 0.2,
        verbose: bool = False,
    ):
        load_dotenv()
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        self.verbose = verbose

        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [
                m for m in self.preferred_models if m != model
            ]

        # API 키 로드
        self.api_key = os.getenv("CAPSTONE_OPENAI_API")
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

    # -----------------------------
    # 공통 유틸
    # -----------------------------
    def _p(self, msg: str):
        if self.verbose:
            print(f"[{self.agent_id}] {msg}")

    @staticmethod
    def _msg(role: str, text: str) -> dict:
        return {"role": role, "content": [{"type": "input_text", "text": text}]}

    # -----------------------------
    # 메인 워크플로
    # -----------------------------
    def reviewer_draft(self, stock_data, target: Target) -> Opinion:
        """LLM을 통해 '이 예측이 타당한 이유(reason)' 생성"""
        prompt_set = OPINION_PROMPTS.get(self.agent_id, OPINION_PROMPTS["technical_agent"])

        context = json.dumps({
            "agent_id": self.agent_id,
            "predicted_next_close": round(target.next_close, 3),
            "uncertainty_sigma": round(target.uncertainty or 0.0, 4),
            "confidence_beta": round(target.confidence or 0.0, 4),
            "latest_data": str(stock_data)
        }, ensure_ascii=False, indent=2)

        sys_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=context)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")
        return Opinion(agent_id=self.agent_id, target=target, reason=reason)

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""
        prompt_set = REBUTTAL_PROMPTS.get(self.agent_id, REBUTTAL_PROMPTS["technical_agent"])

        context = json.dumps({
            "self_agent": my_opinion.agent_id,
            "self_next_close": my_opinion.target.next_close,
            "self_confidence": my_opinion.target.confidence,
            "self_uncertainty": my_opinion.target.uncertainty,
            "self_reason": my_opinion.reason,
            "other_agent": other_opinion.agent_id,
            "other_next_close": other_opinion.target.next_close,
            "other_confidence": other_opinion.target.confidence,
            "other_uncertainty": other_opinion.target.uncertainty,
            "other_reason": other_opinion.reason
        }, ensure_ascii=False, indent=2)

        sys_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=context)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"}
                },
                "required": ["stance", "message"]
            }
        )

        return Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )

    def reviewer_revise(self, my_opinion: Opinion, rebuttals: list, others: list) -> Opinion:
        """LLM을 통해 수정된 reasoning 생성 (수치는 내부 알고리즘이 담당)"""
        prompt_set = REVISION_PROMPTS.get(self.agent_id, REVISION_PROMPTS["technical_agent"])

        context = json.dumps({
            "agent_id": self.agent_id,
            "my_reason": my_opinion.reason,
            "my_confidence": my_opinion.target.confidence,
            "my_uncertainty": my_opinion.target.uncertainty,
            "others": [
                {"agent": o.agent_id, "reason": o.reason, "confidence": o.target.confidence}
                for o in others
            ],
            "rebuttals": [
                {"from": r.from_agent_id, "stance": r.stance, "message": r.message}
                for r in rebuttals
            ],
        }, ensure_ascii=False, indent=2)

        sys_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=context)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
        )

        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        return Opinion(agent_id=self.agent_id, target=my_opinion.target, reason=revised_reason)

    # -----------------------------
    # 구현 필요 함수 (추상)
    # -----------------------------
    def searcher(self, ticker: str) -> StockData:
        """티커 기반 원천 데이터 수집 → StockData 반환(구현 필요)"""
        self._p(f"searcher(ticker={ticker})")
        raise NotImplementedError(f"{self.__class__.__name__} must implement searcher method")
    
    def predicter(self, stock_data: StockData) -> Target:
        """입력 데이터를 바탕으로 Target(next_close) 생성(구현 필요)"""
        self._p("predicter(stock_data)")
        raise NotImplementedError(f"{self.__class__.__name__} must implement predicter method")
    
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """LLM(system/user) 메시지 생성(구현 필요)"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_opinion method")
    
    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        """LLM(system/user) 메시지 생성(구현 필요)"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_rebuttal method")

    # -----------------------------
    # OpenAI API 호출
    # -----------------------------
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
        payload_base = {
            "input": [msg_sys, msg_user],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Response",
                    "strict": True,
                    "schema": schema_obj,
                }
            },
            "temperature": self.temperature,
        }
        last_err = None
        for model in self.preferred_models:
            payload = dict(payload_base, model=model)
            try:
                r = requests.post(self.OPENAI_URL, headers=self.headers, json=payload, timeout=120)
                if r.ok:
                    data = r.json()
                    # 1) output_text 우선 사용
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}  # JSON 실패 시 원문 텍스트 보존
                    # 2) output 배열에서 텍스트 모으기
                    out = data.get("output")
                    if isinstance(out, list) and out:
                        texts = []
                        for blk in out:
                            for c in blk.get("content", []):
                                if "text" in c:
                                    texts.append(c["text"])
                        joined = "\n".join(t for t in texts if t)
                        if joined.strip():
                            try:
                                return json.loads(joined)
                            except Exception:
                                return {"reason": joined}
                    # 비정상 응답
                    return {}
                # 400/404는 다음 모델로 폴백
                if r.status_code in (400, 404):
                    last_err = (r.status_code, r.text)
                    continue
                # 기타 에러는 즉시 예외
                r.raise_for_status()
            except Exception as e:
                self._p(f"⚠️ 모델 {model} 실패: {e}")
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")
    # -----------------------------------------
    # 🔹 추가: Monte Carlo Dropout 기반 불확실성 추정
    # -----------------------------------------
    def predict_with_uncertainty(self, X, n_samples: int = 30):
        """Monte Carlo Dropout 기반 예측 + 불확실성 계산"""
        if not hasattr(self, "model"):
            raise AttributeError("BaseAgent에 self.model이 정의되어야 합니다.")
        
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()

        device = next(self.model.parameters()).device
        X_tensor = X_tensor.to(device)
        self.model.train()  # Dropout 활성화

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = self.model(X_tensor).cpu().numpy().flatten()
                preds.append(y_pred)
        
        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        
        # 로컬 confidence (σ의 역수)
        confidence = 1 / (std_pred + 1e-8)
        
        return mean_pred, std_pred, confidence

    def searcher(self, ticker: str, rebuild: bool = False):
        """
        preprocessing.py 기반 데이터 검색기
        - CSV가 없을 경우 build_dataset()으로 자동 생성
        - 마지막 window 시퀀스를 torch.tensor로 반환
        """
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_type}_dataset.csv")

        # 데이터셋이 존재하지 않으면 생성
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {self.agent_type} dataset not found. Building new dataset...")
            build_dataset(ticker, save_dir=self.data_dir, window_size=self.window_size)

        # CSV에서 데이터셋 로드
        X, y, scaler_X, _, feature_cols = load_csv_dataset(
            ticker, agent_type=self.agent_type, save_dir=self.data_dir
        )

        # 가장 최근 window 데이터만 사용
        X_latest = X[-1:]  # shape: (1, window_size, n_features)
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        print(f"✅ {ticker} {self.agent_type} searcher loaded shape: {X_tensor.shape}")
        return X_tensor

    def predicter(self, X):
        """입력 텐서를 받아 Monte Carlo Dropout 기반 예측 수행"""
        mean_pred, std_pred, confidence = self.predict_with_uncertainty(X)
        return Target(
            next_close=float(mean_pred),
            uncertainty=float(std_pred),
            confidence=float(confidence)
        )
#         a
#     import torch
# import torch.nn as nn
# import numpy as np

# class BaseAgent:
#     def __init__(self, model: nn.Module, feature_cols: list[str], name: str):
#         self.model = model
#         self.feature_cols = feature_cols
#         self.name = name
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
    
#     # -----------------------------------------
#     # 🔹 기존: 단일 예측 (Deterministic)
#     # -----------------------------------------
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """평균 예측(드롭아웃 비활성화 상태)"""
#         self.model.eval()
#         with torch.no_grad():
#             X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
#             preds = self.model(X_tensor).cpu().numpy().flatten()
#         return preds

#     # -----------------------------------------
#     # 🔹 추가: Monte Carlo Dropout 기반 불확실성 추정
#     # -----------------------------------------
#     def predict_with_uncertainty(self, X, n_samples: int = 30):
#         """Monte Carlo Dropout 기반 예측 + 불확실성 계산"""
#         if not hasattr(self, "model"):
#             raise AttributeError("BaseAgent에 self.model이 정의되어야 합니다.")
        
#         if isinstance(X, np.ndarray):
#             X_tensor = torch.tensor(X, dtype=torch.float32)
#         else:
#             X_tensor = X.float()

#         device = next(self.model.parameters()).device
#         X_tensor = X_tensor.to(device)
#         self.model.train()  # Dropout 활성화

#         preds = []
#         with torch.no_grad():
#             for _ in range(n_samples):
#                 y_pred = self.model(X_tensor).cpu().numpy().flatten()
#                 preds.append(y_pred)
        
#         preds = np.stack(preds)
#         mean_pred = preds.mean(axis=0)
#         std_pred = preds.std(axis=0)
        
#         # 로컬 confidence (σ의 역수)
#         confidence = 1 / (std_pred + 1e-8)
        
#         return mean_pred, std_pred, confidence