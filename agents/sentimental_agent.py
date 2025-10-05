import json
import numpy as np
import pandas as pd
import yfinance as yf
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple
from prompts import SEARCHER_PROMPTS, PREDICTER_PROMPTS, OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
class SentimentalAgent(BaseAgent):
    # ------------------------------------------------------------------
    # 1) 데이터 수집
    # ------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        # 현재가와 통화 가져오기
        df = yf.download(ticker, period="5d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()

        # 스키마 정의
        schema_sent = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "positives": {"type": "array", "items": {"type": "string"}},
                "negatives": {"type": "array", "items": {"type": "string"}},
                "evidence":  {"type": "array", "items": {"type": "string"}},
                "summary":   {"type": "string"},
            },
            "required": ["sentiment", "positives", "negatives", "evidence", "summary"],
            "additionalProperties": False,
        }

        sys_text = SEARCHER_PROMPTS["sentimental"]["system"]
        user_text = SEARCHER_PROMPTS["sentimental"]["user_template"].format(
            ticker=ticker, 
            current_price=last_price, 
            currency=currency
        )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            schema_sent
        )

        self.stockdata = StockData(
            sentimental=parsed,
            fundamental={},
            technical={},
            last_price=last_price,
            currency=currency
        )
        self.current_ticker = ticker  # 현재 티커 저장
        return self.stockdata

    # ------------------------------------------------------------------
    # 2) 1차 예측
    # ------------------------------------------------------------------
    def predicter(self, stock_data: StockData) -> Target:
        # 현재 가격 정보 가져오기
        ticker = getattr(self, 'current_ticker', 'UNKNOWN')
        df = yf.download(ticker, period="1d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()
        
        # 센티멘탈 분석가 특성: 중립적, 현재가 대비 ±10% 범위
        min_price = last_price * 0.90
        max_price = last_price * 1.10
        
        ctx = {
            "sentimental_summary": stock_data.sentimental,
            "current_price": last_price,
            "currency": currency,
            "prediction_range": f"{min_price:.2f} - {max_price:.2f} {currency}",
            "agent_character": "중립적인 센티멘탈 분석가로서 시장 심리와 여론에 기반한 균형 잡힌 예측을 제공합니다."
            }
        sys_text = PREDICTER_PROMPTS["sentimental"]["system"]
        user_text = PREDICTER_PROMPTS["sentimental"]["user_template"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            self.schema_obj_opinion
        )
        return Target(next_close=float(parsed.get("next_close", 0.0)))
    
    # ------------------------------------------------------------------
    # 3) LLM 메시지 빌드(Opinion): 다음날 종가와 근거를 JSON으로 요구
    #    - 시스템: 역할/출력형식 고정
    #    - 사용자: 컨텍스트(JSON 직렬화 가능 타입만)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_close": float(stock_data.last_price or 0.0),
            "signals": {k: float(v) for k, v in (stock_data.technical or {}).items()},
            "our_prediction": float(target.next_close),
            "format_rule": f"숫자는 소수 {decimals}자리, 통화 {ccy}"
        }

        system_text = OPINION_PROMPTS["sentimental"]["system"]
        user_text   = OPINION_PROMPTS["sentimental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text


    # ------------------------------------------------------------------
    # 4) LLM 메시지 빌드(Rebuttal): 내/상대 의견 비교 → REBUT/SUPPORT + message
    #    - 시스템: 출력키는 'stance'와 'message' (스키마와 일치)
    #    - 사용자: 숫자는 float로, 텍스트는 문자열로 제한
    # ------------------------------------------------------------------
    def _build_messages_rebuttal(self,
                                my_opinion: Opinion,
                                target_agent: str,
                                target_opinion: Opinion,
                                stock_data: StockData) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        ctx = {
            "ticker": t,
            "me": {
                "agent_id": self.agent_id,
                "next_close": round(float(my_opinion.target.next_close), decimals),
                "reason": str(my_opinion.reason)[:2000],
            },
            "other": {
                "agent_id": target_agent,
                "next_close": round(float(target_opinion.target.next_close), decimals),
                "reason": str(target_opinion.reason)[:2000],
            },
            "snapshot": {
                "last_price": float(stock_data.last_price or 0.0),
                "currency": ccy,
                "signals": {
                    "technical":   {k: float(v) for k, v in (stock_data.technical   or {}).items()},
                    "sentimental": (stock_data.sentimental or {}),
                    "fundamental": (stock_data.fundamental or {}),
                },
            },
        }

        system_text = REBUTTAL_PROMPTS["sentimental"]["system"]
        user_text   = REBUTTAL_PROMPTS["sentimental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text

    
    
    def _build_messages_revision(self,
                                my_lastest: Opinion,
                                others_latest: Dict[str, Opinion],
                                received_rebuttals: List[Rebuttal],
                                stock_data: StockData) -> tuple[str, str]:
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        me = {
            "agent_id": my_lastest.agent_id,
            "next_close": float(my_lastest.target.next_close),
            "reason": str(my_lastest.reason)[:2000],
        }
        peers = [{
            "agent_id": str(aid),
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
            "snapshot": {
                "last_price": float(stock_data.last_price or 0.0),
                "currency": ccy,
                "signals": {
                    "technical":   {k: float(v) for k, v in (stock_data.technical   or {}).items()},
                    "sentimental": (stock_data.sentimental or {}),
                    "fundamental": (stock_data.fundamental or {}),
                },
            },
        }

        system_text = REVISION_PROMPTS["sentimental"]["system"]
        user_text   = REVISION_PROMPTS["sentimental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text

    
    # ------------------------------------------------------------------
    # RSI 계산: 단순 이동평균 버전(EMA 아님)
    # ------------------------------------------------------------------
    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(period).mean()
        avg_loss = down.rolling(period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        return 100 - (100 / (1 + rs))
    
    def _update_prompts(self, prompt_configs: Dict[str, str]) -> None:
        """프롬프트 설정 업데이트 (main.py에서 호출)"""
        global PREDICTER_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
        
        # predicter 프롬프트 업데이트
        if "predicter_system" in prompt_configs:
            PREDICTER_PROMPTS["sentimental"]["system"] = prompt_configs["predicter_system"]
        
        # rebuttal 프롬프트 업데이트
        if "rebuttal_system" in prompt_configs:
            REBUTTAL_PROMPTS["sentimental"]["system"] = prompt_configs["rebuttal_system"]
        
        # revision 프롬프트 업데이트
        if "revision_system" in prompt_configs:
            REVISION_PROMPTS["sentimental"]["system"] = prompt_configs["revision_system"]
