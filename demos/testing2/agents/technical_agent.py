import json
import numpy as np
import pandas as pd
import yfinance as yf
import os
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple
# 간단한 프롬프트 정의
SEARCHER_PROMPTS = {
    "technical": "기술적 분석을 위한 차트 데이터를 수집하세요."
}
PREDICTER_PROMPTS = {
    "technical": {
        "system": "당신은 전문적인 기술적 분석가입니다. 기술적 지표를 바탕으로 주가를 예측하세요.",
        "user_template": "다음 기술적 데이터를 바탕으로 주가를 예측해주세요:\n{context}"
    }
}
OPINION_PROMPTS = {
    "technical": {
        "system": "당신은 전문적인 기술적 분석가입니다. 기술적 관점에서 의견을 제시하세요.",
        "user": "다음 데이터를 바탕으로 의견을 제시해주세요:\n{context}"
    }
}
REBUTTAL_PROMPTS = {
    "technical": {
        "system": "당신은 전문적인 기술적 분석가입니다. 다른 관점에 대해 반박하세요.",
        "user": "다음 의견에 대해 반박해주세요:\n{context}"
    }
}
REVISION_PROMPTS = {
    "technical": "의견을 수정하세요."
}
from agents.technical_modules import TechnicalModuleManager

class TechnicalAgent(BaseAgent):
    def __init__(self, 
                 agent_id: str = "TechnicalAgent",
                 use_ml_modules: bool = False,
                 fred_api_key: Optional[str] = None,
                 model_path: Optional[str] = None,
                 **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        
        # ML 모듈 설정
        self.use_ml_modules = use_ml_modules
        if self.use_ml_modules:
            self.ml_manager = TechnicalModuleManager(
                use_ml_searcher=True,
                use_ml_predictor=True,
                fred_api_key=fred_api_key or os.getenv('FRED_API_KEY'),
                model_path=model_path or "model_artifacts/final_best.keras"
            )
        else:
            self.ml_manager = None
    
    # ------------------------------------------------------------------
    # 1) 데이터 수집 
    # ------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        df = yf.download(ticker, period="5d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()

        schema_tech = {
            "type": "object",
            "properties": {
                "trend":    {"type": "string", "enum": ["UP", "DOWN", "SIDEWAYS"]},
                "strength": {"type": "number"},
                "signals":  {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "summary":  {"type": "string"},
            },
            "required": ["trend", "strength", "signals", "evidence", "summary"],
            "additionalProperties": False,
        }

        sys_text = SEARCHER_PROMPTS["technical"]["system"]
        user_text = SEARCHER_PROMPTS["technical"]["user_template"].format(
            ticker=ticker, 
            current_price=last_price, 
            currency=currency
        )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            schema_tech
        )

        # ML 모듈 사용 여부에 따른 데이터 수집
        if self.use_ml_modules and self.ml_manager:
            # ML 모듈을 사용한 향상된 기술적 분석 데이터 수집
            ml_technical_data = self.ml_manager.get_enhanced_technical_data(ticker, last_price)
            
            # ML 결과를 기술적 데이터에 추가
            parsed["ml_signals"] = ml_technical_data.get('signals', {})
            parsed["ml_confidence"] = ml_technical_data.get('confidence', 0.0)
            parsed["ml_indicators"] = ml_technical_data.get('indicators', {})
            
            # ML 결과를 GPT 프롬프트에 포함하여 재분석
            ml_context = f"""
ML 모델 분석 결과:
- 기술적 신호: {ml_technical_data.get('signals', {})}
- 신뢰도: {ml_technical_data.get('confidence', 0.0):.2f}
- 수집된 뉴스: {ml_technical_data.get('news_count', 0)}개
- 기술적 지표: RSI, MA, 볼린저밴드 등 계산 완료
"""

            # ML 컨텍스트를 포함한 재분석
            user_text_with_ml = SEARCHER_PROMPTS["technical"]["user_template"].format(
                ticker=ticker, 
                current_price=last_price, 
                currency=currency
            ) + f"\n\n{ml_context}"

            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text_with_ml),
                schema_tech
            )

        self.stockdata = StockData(
            sentimental={},
            fundamental={},
            technical=parsed,
            last_price=last_price,
            currency=currency
        )
        self.current_ticker = ticker  # 현재 티커 저장
        return self.stockdata
    # ------------------------------------------------------------------
    # 2) 1차 예측 (LLM-only)
    #    - 기술 요약(트렌드/강도/신호) + 현재가 앵커로 next_close 산출
    # ------------------------------------------------------------------
    def predicter(self, stock_data: StockData) -> Target:
        # 현재 가격 정보 가져오기
        ticker = getattr(self, 'current_ticker', 'UNKNOWN')
        df = yf.download(ticker, period="1d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()
        
        # 기술적 분석가 특성: 공격적, 현재가 대비 ±15% 범위
        min_price = last_price * 0.85
        max_price = last_price * 1.15
        
        ctx = {
            "technical_summary": stock_data.technical,
            "current_price": last_price,
            "currency": currency,
            "prediction_range": f"{min_price:.2f} - {max_price:.2f} {currency}",
            "agent_character": "공격적인 기술적 분석가로서 차트 패턴과 모멘텀에 기반한 적극적인 예측을 제공합니다."
            }
        sys_text = PREDICTER_PROMPTS["technical"]["system"]
        user_text = PREDICTER_PROMPTS["technical"]["user_template"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            self.schema_obj_opinion
        )
        return Target(next_close=float(parsed.get("next_close", 0.0)))
    
    # ------------------------------------------------------------------
    # 3) Opinion 메시지 빌드 (기술 관점)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()
        last = float(stock_data.last_price or 0.0)

        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_price": last,
            "technical_summary": stock_data.technical or {},
            "our_prediction": float(target.next_close),
        }

        system_text = OPINION_PROMPTS["technical"]["system"]
        user_text   = OPINION_PROMPTS["technical"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text

    # ------------------------------------------------------------------
    # 4) Rebuttal/Revision (기술 관점 문구)
    # ------------------------------------------------------------------
    def _build_messages_rebuttal(self,
                                my_opinion: Opinion,
                                target_agent: str,
                                target_opinion: Opinion,
                                stock_data: StockData) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()

        ctx = {
            "ticker": t,
            "currency": ccy,
            "technical_summary": stock_data.technical or {},
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:2000],
            },
            "other": {
                "agent_id": target_agent,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
            }
        }

        system_text = REBUTTAL_PROMPTS["technical"]["system"]
        user_text   = REBUTTAL_PROMPTS["technical"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text


    def _build_messages_revision(self,
                                my_lastest: Opinion,
                                others_latest: Dict[str, Opinion],
                                received_rebuttals: List[Rebuttal],
                                stock_data: StockData) -> tuple[str, str]:
        ccy = (stock_data.currency or "USD").upper()

        me = {
            "agent_id": my_lastest.agent_id,
            "next_close": float(my_lastest.target.next_close),
            "reason": str(my_lastest.reason)[:2000],
        }
        peers = [{
            "agent_id": aid,
            "next_close": float(op.target.next_close),
            "reason": str(op.reason)[:2000],
        } for aid, op in (others_latest or {}).items()]
        feedback = [{
            "from": r.from_agent_id,
            "to":   r.to_agent_id,
            "stance": r.stance,
            "message": str(r.message)[:500],
        } for r in (received_rebuttals or [])]

        ctx = {
            "me": me,
            "peers": peers,
            "feedback": feedback,
            "technical_summary": stock_data.technical or {},
            "currency": ccy
        }

        system_text = REVISION_PROMPTS["technical"]["system"]
        user_text   = REVISION_PROMPTS["technical"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text
    
    def _update_prompts(self, prompt_configs: Dict[str, str]) -> None:
        """프롬프트 설정 업데이트 (main.py에서 호출)"""
        global PREDICTER_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
        
        # predicter 프롬프트 업데이트
        if "predicter_system" in prompt_configs:
            PREDICTER_PROMPTS["technical"]["system"] = prompt_configs["predicter_system"]
        
        # rebuttal 프롬프트 업데이트
        if "rebuttal_system" in prompt_configs:
            REBUTTAL_PROMPTS["technical"]["system"] = prompt_configs["rebuttal_system"]
        
        # revision 프롬프트 업데이트
        if "revision_system" in prompt_configs:
            REVISION_PROMPTS["technical"]["system"] = prompt_configs["revision_system"]
