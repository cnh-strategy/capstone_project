# ======================= Debate Framework =======================
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from datetime import datetime, timedelta

# -----------------------------
# 공통 데이터 포맷
# -----------------------------
@dataclass
class Target:
    """예측 목표값 묶음 (필요 시 필드 확장)
    - next_close: 다음 거래일 종가 예측치
    """
    next_close : float

@dataclass
class Opinion:
    """에이전트의 의견(초안/수정본 공통 포맷)
    - agent_id: 의견을 낸 에이전트 식별자
    - target  : 예측 타깃(예: next_close)
    - reason  : 근거 텍스트(LLM/룰 기반)
    """
    agent_id: str
    target: Target
    reason: str  # TODO: LLM/룰 기반 사유 텍스트 생성

@dataclass
class Rebuttal:
    """에이전트 간 반박/지지 메시지
    - from_agent_id: 보낸 쪽
    - to_agent_id  : 받는 쪽
    - stance       : REBUT(반박) | SUPPORT(지지)
    - message      : 근거 텍스트(간결 요약)
    """
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str  # TODO: LLM/룰 기반 한 줄 근거 생성

@dataclass
class RoundLog:
    """라운드별 기록 스냅샷(옵셔널로 사용)
    - round_no : 라운드 번호
    - opinions : 라운드 내 각 에이전트 최종 Opinion
    - rebuttals: 라운드 내 교환된 반박/지지
    - summary  : {"agent_id": Target(...)} 형태의 집계 요약
    """
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    """에이전트 입력 원천 데이터(필요 시 자유 확장)
    - sentimental: 심리/커뮤니티/뉴스 스냅샷
    - fundamental: 재무/밸류에이션 요약
    - technical  : 가격/지표 스냅샷
    - last_price : 최신 종가
    - currency   : 통화코드
    """
    sentimental: Dict
    fundamental: Dict
    technical: Dict
    last_price: Optional[float] = None
    currency: Optional[str] = None


# -----------------------------
# BaseAgent (인터페이스/공통 기능)
# -----------------------------
import os
import time
import json
import requests
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from datetime import datetime, timezone
from collections import defaultdict

# .env 로드 (환경변수 세팅)
load_dotenv()

# OpenAI Responses API 엔드포인트/UA
OPENAI_URL = "https://api.openai.com/v1/responses"
UA = "Mozilla/5.0"

class BaseAgent:
    def __init__(self, 
                 agent_id: str, 
                 model: str | None = None,           # 단일 고정 모델(선택)
                 preferred_models: list[str] | None = None,  # 폴백 순서
                 temperature: float = 0.2,
                 verbose: bool = False,
                 use_ml_modules: bool = False,
                 model_path: Optional[str] = None):
        # --- 런타임 설정 ---
        self.agent_id = agent_id
        self.temperature = temperature
        self.model = model
        self.verbose = verbose
        
        # --- ML 설정 ---
        self.use_ml_modules = use_ml_modules
        self.model_path = model_path
        self.ml_model = None
        self.scaler = None
        self.beta_value = 0.5  # 신뢰도 값

        # --- 상태 보관 ---
        self.stockdata = None                                  # 최신 stockdata
        self.opinions: list = []                               # Opinion 히스토리
        self.rebuttals: dict[int, List[Rebuttal]] = defaultdict(list) # {round: [Rebuttal, ...]}


        # --- 모델 우선순위 ---
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            # 요청 모델을 최우선으로, 중복 제거
            self.preferred_models = [model] + [m for m in self.preferred_models if m != model]

        # --- API 키 로드/검증 ---
        OPENAI_API_KEY = os.getenv('CAPSTONE_OPENAI_API')
        if not OPENAI_API_KEY:
            raise RuntimeError("환경변수 CAPSTONE_OPENAI_API 필요")

        # --- 공통 헤더 ---
        self.headers = {
            "Authorization" : f"Bearer {OPENAI_API_KEY}",
            "Content-Type" : "application/json",
        }

        # --- JSON 스키마: Opinion(next_close, reason) ---
        self.schema_obj_opinion = {
            "type": "object",
            "properties": {
                "next_close": {"type": "number", "description": "예상되는 다음 거래일 종가"},
                "reason":     {"type": "string",  "description": "근거 요약 (한국어 4~5문장)"},
            },
            "required": ["next_close", "reason"],
            "additionalProperties": False,
        }

        # --- JSON 스키마: Rebuttal(stance, message) ---
        self.schema_obj_rebuttal = {
            "type": "object",
            "properties": {
                "stance": {
                    "type": "string",
                    "enum": ["REBUT", "SUPPORT"],
                    "description": "다른 에이전트 의견에 대한 본인 입장",
                },
                "message": {"type": "string", "description": "근거 요약 (한국어 4~5문장)"},
            },
            "required": ["stance", "message"],
            "additionalProperties": False,
        }

    # --- 로깅 헬퍼 ---
    def _p(self, msg: str):
        if self.verbose:
            print(f"[{self.agent_id}] {msg}")

    # 1) 데이터 수집: API/크롤러 등으로 예측 입력 수집
    def searcher(self, ticker: str) -> StockData:
        """티커 기반 원천 데이터 수집 → StockData 반환(구현 필요)"""
        self._p(f"searcher(ticker={ticker})")
        raise NotImplementedError

    # 2) 타깃 산출: 수집 데이터를 바탕으로 1차 예측치 생성
    def predicter(self, stock_data: StockData) -> Target:
        """입력 데이터를 바탕으로 Target(next_close) 생성(구현 필요)"""
        self._p("predicter(stock_data)")
        raise NotImplementedError

    # 3) Opinion 메시지 빌드 (LLM 호출용)
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """LLM(system/user) 메시지 생성(구현 필요)"""
        raise NotImplementedError

    # 4) Rebuttal 메시지 빌드 (LLM 호출용)
    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        """LLM(system/user) 메시지 생성(구현 필요)"""
        raise NotImplementedError

    # ---------- 공용 파이프라인 ----------
    def reviewer_draft(self, ticker: str) -> Opinion:
        """(1) searcher → (2) predicter → (3) LLM(JSON Schema)로 reason 생성 → Opinion 반환"""
        # 1) 데이터 수집
        self.stockdata = self.searcher(ticker)

        # 2) 예측값 생성
        target = self.predicter(self.stockdata)

        # 3) LLM 호출(reason 생성)
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)
        msg_sys = self._msg("system", sys_text)
        msg_user = self._msg("user",   user_text)

        parsed = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj_opinion)
        reason = parsed.get("reason") or "(사유 생성 실패: 미입력)"

        # 4) Opinion 기록/반환 (항상 최신 값 append)
        opinion = Opinion(agent_id=self.agent_id, target=target, reason=reason)
        self.opinions.append(opinion) # 최신 오피니언 기록용
        return opinion

    def reviewer_rebut(self,
                       round_num: int,
                       my_lastest: Opinion,
                       others_latest: Dict[str, Opinion],
                       stock_data: StockData) -> Dict[str, Rebuttal]:
        """라운드별·상대별 Rebuttal 생성 후 저장하여 반환
        - 입력:
            round_num     : 라운드 번호
            my_lastest    : 나의 최신 Opinion
            others_latest : {상대 에이전트ID: 상대 Opinion}
            stock_data    : 공용 컨텍스트(스냅샷)
        - 출력:
            {상대 에이전트ID: Rebuttal} (해당 라운드 버킷)
        """
        self._p(f"reviewer_rebut(round={round_num})")

        for other_id, other_op in others_latest.items():
            # 1) LLM 메시지 구성
            sys_text, user_text = self._build_messages_rebuttal(my_lastest, other_id, other_op, stock_data)
            msg_sys  = self._msg("system", sys_text)
            msg_user = self._msg("user",   user_text)

            # 2) LLM 호출 → JSON 파싱
            parsed  = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj_rebuttal)
            stance  = parsed.get("stance")  or "(사유 생성 실패: 미입력)"
            message = parsed.get("message") or "(사유 생성 실패: 미입력)"

            # 3) Rebuttal 생성/저장(라운드 리스트에 누적)
            rebuttal = Rebuttal(from_agent_id=self.agent_id,
                                to_agent_id=other_id,
                                stance=stance,
                                message=message)
            self.rebuttals[round_num].append(rebuttal)

        # 해당 라운드의 Rebuttal 리스트 반환
        return self.rebuttals[round_num]

    def reviewer_revise(self,
                        my_lastest: Opinion,
                        others_latest: Dict[str, Opinion],
                        received_rebuttals: List[Rebuttal],
                        stock_data: StockData) -> Opinion:
        """반박/지지 결과를 반영해 Opinion 조정 (LLM 기반)"""
        self._p("reviewer_revise(LLM)")

        # 1) 메시지 빌드
        sys_text = (
            "너는 여러 에이전트의 의견과 반박/지지 결과를 종합하여 "
            "내 기존 Opinion(next_close, reason)을 필요 시 조정하는 애널리스트다. "
            "반환은 JSON(next_close:number, reason:string)만 허용한다."
        )
        ctx = {
            "my_last": {
                "next_close": float(my_lastest.target.next_close),
                "reason": my_lastest.reason,
            },
            "others": {
                oid: {"next_close": float(op.target.next_close), "reason": op.reason}
                for oid, op in others_latest.items()
            },
            "received_rebuttals": [
                {
                    "from": r.from_agent_id,
                    "stance": r.stance,
                    "message": r.message
                }
                for r in received_rebuttals
            ],
            "snapshot": {
                "last_price": float(stock_data.last_price or 0.0),
                "currency": stock_data.currency,
            },
        }
        user_text = "아래 컨텍스트를 참고하여 JSON만 반환하라:\n" + json.dumps(ctx, ensure_ascii=False)

        msg_sys  = self._msg("system", sys_text)
        msg_user = self._msg("user", user_text)

        # 2) LLM 호출 → opinion 스키마 그대로 사용
        parsed = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj_opinion)

        new_next   = parsed.get("next_close", my_lastest.target.next_close)
        new_reason = parsed.get("reason", my_lastest.reason)

        # 3) Opinion 업데이트 & 기록
        revised = Opinion(agent_id=self.agent_id,
                        target=Target(new_next),
                        reason=new_reason)
        self.opinions.append(revised)

        # 항상 최신 Opinion 반환
        return self.opinions[-1]

    # ---------- 도우미 ----------
    def _normalize_ticker(self, ticker: str) -> str:
        """티커 문자열 정규화(KRX 숫자 6자리면 .KS 부착 등)"""
        t = ticker.strip().upper()
        if t.isdigit() and len(t) == 6:
            return t + ".KS"
        return t

    def _detect_currency_and_decimals(self, ticker: str) -> tuple[str, int]:
        """티커로 통화코드/소수자리 추론(KRW/JPY=0, 그 외=2)"""
        try:
            info = yf.Ticker(ticker).info
            ccy = (info.get("currency") or "KRW").upper()
        except Exception:
            ccy = "KRW"
        decimals = 0 if ccy in ("KRW", "JPY") else 2
        return ccy, decimals

    @staticmethod
    def _msg(role: str, text: str) -> dict:
        """OpenAI Responses API용 메시지 포맷 변환"""
        return {"role": role, "content": [{"type": "input_text", "text": text}]}

    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """Responses API 호출(여러 모델 폴백) → JSON Schema 준수 응답 파싱
        - msg_sys/msg_user: self._msg(...)로 생성된 메시지
        - schema_obj      : json_schema(Strict) 형식
        - 반환            : dict(LLM가 준 JSON)
        """
        body_base = {
            "input": [msg_sys, msg_user],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Opinion",   # 호출 시 전달된 schema_obj에 맞춰 구조만 검증
                    "strict": True,
                    "schema": schema_obj,
                }
            },
            "temperature": self.temperature,
        }
        last_err = None
        for m in self.preferred_models:
            body = dict(body_base, model=m)
            r = requests.post(OPENAI_URL, json=body, headers=self.headers, timeout=120)
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
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    # ======================= ML 공통 기능 =======================
    
    def ensure_data_dir(self, data_dir: str = "data"):
        """데이터 디렉토리 생성"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def ensure_models_dir(self, models_dir: str = "ml_modules/models"):
        """모델 디렉토리 생성"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def load_model_and_scaler(self, ticker: str, models_dir: str = "ml_modules/models") -> bool:
        """모델과 스케일러 로드"""
        if not self.use_ml_modules:
            return False
            
        model_path = os.path.join(models_dir, f"{ticker}_{self.agent_id.lower()}_model.pt")
        scaler_path = os.path.join(models_dir, f"{ticker}_{self.agent_id.lower()}_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False
        
        try:
            # 모델 로드 (하위 클래스에서 구현)
            if hasattr(self, 'create_ml_model'):
                self.ml_model = self.create_ml_model()
                self.ml_model.load_state_dict(torch.load(model_path))
                self.ml_model.eval()
            
            # 스케일러 로드
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ {self.agent_id} 모델 로드 실패: {str(e)}")
            return False
    
    def save_model_and_scaler(self, ticker: str, models_dir: str = "ml_modules/models") -> bool:
        """모델과 스케일러 저장"""
        if not self.use_ml_modules or self.ml_model is None or self.scaler is None:
            return False
        
        try:
            self.ensure_models_dir(models_dir)
            
            # 모델 저장
            model_path = os.path.join(models_dir, f"{ticker}_{self.agent_id.lower()}_model.pt")
            torch.save(self.ml_model.state_dict(), model_path)
            
            # 스케일러 저장
            scaler_path = os.path.join(models_dir, f"{ticker}_{self.agent_id.lower()}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ {self.agent_id} 모델 저장 실패: {str(e)}")
            return False
    
    def predict_with_uncertainty(self, X: torch.Tensor, num_samples: int = 10) -> Tuple[float, float]:
        """Monte Carlo Dropout으로 불확실성과 함께 예측"""
        if self.ml_model is None:
            return 0.0, 1.0
        
        self.ml_model.train()  # Dropout 활성화
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.ml_model(X)
                predictions.append(pred.item())
        
        self.ml_model.eval()  # Dropout 비활성화
        
        mean_pred = np.mean(predictions)
        uncertainty = np.var(predictions)
        
        return mean_pred, uncertainty
    
    def peer_correction(self, prediction: float, peer_predictions: Dict[str, float]) -> float:
        """동료 에이전트들의 예측을 바탕으로 보정"""
        if len(peer_predictions) == 0:
            return prediction
        
        # 동료들의 평균 예측
        peer_mean = np.mean(list(peer_predictions.values()))
        
        # β 값에 따른 보정
        alpha = 0.1  # 학습률
        corrected_prediction = prediction + alpha * self.beta_value * (peer_mean - prediction)
        
        return corrected_prediction
    
    def update_beta(self, uncertainty: float, accuracy: float):
        """β 신뢰도 업데이트 (EMA 방식)"""
        # 불확실성 기반 신뢰도 계산
        confidence = 1.0 / (1.0 + uncertainty)
        
        # 정확도 기반 신뢰도 계산
        accuracy_confidence = accuracy
        
        # 종합 신뢰도
        new_beta = (confidence + accuracy_confidence) / 2.0
        
        # EMA 업데이트
        lambda_ema = 0.9
        self.beta_value = (
            lambda_ema * self.beta_value + 
            (1 - lambda_ema) * new_beta
        )

