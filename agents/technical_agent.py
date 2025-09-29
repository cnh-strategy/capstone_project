import json
import numpy as np
import pandas as pd
import yfinance as yf
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple

class TechnicalAgent(BaseAgent):
    # ------------------------------------------------------------------
    # 1) 데이터 수집 (LLM-only 지향 + 앵커용 현재가)
    #    - 최근 가격 힌트만 얇게 가져오고(없어도 OK), 나머지 기술요약은 LLM이 생성
    # ------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        t = self._normalize_ticker(ticker)
        self._p(f"[Technical.searcher-LLM] {t}")

        # 현재가/통화 힌트 (실패해도 진행)
        last_price, ccy = None, "USD"
        try:
            df = yf.download(t, period="10d", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                last_price = float(df["Close"].dropna().iloc[-1])
            try:
                info = yf.Ticker(t).info
                ccy = (info.get("currency") or ccy).upper()
            except Exception:
                pass
        except Exception:
            pass

        # LLM에게 기술적 요약을 요청하는 스키마
        schema_technical = {
            "type": "object",
            "properties": {
                "trend":    {"type": "string", "enum": ["UP", "DOWN", "SIDEWAYS"]},
                "strength": {"type": "number", "description": "-1.0(약한 하락) ~ +1.0(강한 상승)"},
                "signals":  {"type": "array", "items": {"type": "string"}, "description": "MACD/RSI/MA/볼린저 등 신호 요약"},
                "summary":  {"type": "string", "description": "기술적 관점 핵심 요약 (한국어 4~6문장)"},
                "evidence": {"type": "string", "description": "근거(예: 골든크로스, 과매수, 거래량 급증 등)"},
            },
            # strict 모드 대비: properties에 있는 키는 모두 required에 포함
            "required": ["trend", "strength", "signals", "summary", "evidence"],
            "additionalProperties": False,
        }

        system_text = (
            "너는 '기술적 분석 전문가'다. 최근 가격 흐름을 가정하고 MA/RSI/MACD/볼린저/거래량 관점으로 "
            "상승/하락/횡보(trend)와 강도(strength)를 정리하라. 과도한 확신은 피하고, "
            "일관된 논리와 대표적 신호명을 사용하라. 반환은 JSON만 허용한다."
        )
        user_text = "컨텍스트:\n" + json.dumps({
            "ticker": t,
            "last_price_hint": last_price,
            "currency": ccy,
            "instruction": "예: 골든크로스/데드크로스, RSI 과매수/과매도, MACD 시그널 교차, 볼린저 %b, 거래량 급증 등.",
            "output": {"trend": "UP|DOWN|SIDEWAYS", "strength": "-1.0~+1.0", "signals": ["..."], "summary_ko": "4~6문장"}
        }, ensure_ascii=False)

        parsed = self._ask_with_fallback(
            self._msg("system", system_text),
            self._msg("user",   user_text),
            schema_technical
        )

        technical = {
            "trend":    parsed.get("trend", "SIDEWAYS"),
            "strength": float(parsed.get("strength", 0.0)),
            "signals":  list(parsed.get("signals", [])),
            "summary":  parsed.get("summary", ""),
            "evidence": parsed.get("evidence", ""),
        }

        self._last_ticker = t
        sd = StockData(
            sentimental={},
            fundamental={},
            technical=technical,     # ← 기술 요약을 여기 담아 전달
            last_price=last_price,   # 앵커용
            currency=ccy
        )
        self.stockdata = sd
        return sd

    # ------------------------------------------------------------------
    # 2) 1차 예측 (LLM-only)
    #    - 기술 요약(트렌드/강도/신호) + 현재가 앵커로 next_close 산출
    # ------------------------------------------------------------------
    def predicter(self, stock_data: StockData) -> Target:
        self._p("[Technical.predicter-LLM] next_close from technical summary")

        ccy = (stock_data.currency or "USD").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2
        last = float(stock_data.last_price or 0.0)

        sys_text = (
            "너는 '기술적 분석 전문가'다. 아래 기술 요약(trend/strength/signals)과 현재가를 기준으로 "
            "다음 거래일 종가(next_close)를 예측하고, 이유(reason)를 한국어 3~4문장으로 제시하라. "
            "규칙: (1) 현재가를 앵커로 판단, (2) 강한 촉발 신호(교차/돌파/볼륨 급증 등)가 없으면 ±3% 내 보수적 판단, "
            "(3) 큰 변동을 예측할 땐 구체 신호를 이유에 명시. 반환은 JSON(next_close:number, reason:string)만."
        )
        ctx = {
            "ticker": getattr(self, "_last_ticker", "UNKNOWN"),
            "currency": ccy,
            "last_price": last,
            "technical_summary": stock_data.technical,  # trend/strength/signals/summary/evidence
        }
        user_text = "컨텍스트:\n" + json.dumps(ctx, ensure_ascii=False)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user",   user_text),
            self.schema_obj_opinion
        )

        raw_next = parsed.get("next_close", last)
        try:
            pred = float(raw_next)
        except Exception:
            pred = last

        next_close = round(pred, decimals)
        return Target(next_close=next_close)

    # ------------------------------------------------------------------
    # 3) Opinion 메시지 빌드 (기술 관점)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()
        last = float(stock_data.last_price or 0.0)

        system_text = (
            "너는 '기술적 분석 전문가'다. 제공된 기술 요약(trend/strength/signals)과 현재가를 바탕으로 "
            "예측치(next_close)에 대한 근거(reason)를 한국어 4~5문장으로 작성하라. "
            "이유에는 현재가 대비 예상 %변화와 촉발 신호(예: 골든크로스, 과매수 해소, 볼륨 스파이크)를 포함하라. "
            "반환은 JSON(next_close:number, reason:string)만 허용한다."
        )
        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_price": last,
            "technical_summary": stock_data.technical or {},
            "our_prediction": float(target.next_close),
            "guideline": "보수적 기본(±3%), 강한 신호가 있을 때만 범위 초과를 정당화."
        }
        user_text = "아래 컨텍스트를 참고하여 JSON으로만 반환:\n" + json.dumps(ctx, ensure_ascii=False)
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
        system_text = (
            "당신은 '기술적 분석 전문가'다. 두 의견의 기술 가정(추세/신호 해석, 강도, 변동 범위)이 "
            "현실적이고 일관적인지 비교해 'REBUT' 또는 'SUPPORT'를 결정하고, 근거를 한국어 4~5문장으로 요약하라. "
            "반환은 JSON({'stance':'REBUT|SUPPORT','message':string})만 허용한다."
        )
        ctx = {
            "ticker": t,
            "currency": ccy,
            "technical_summary": stock_data.technical or {},
            "me":    {"next_close": float(my_opinion.target.next_close),    "reason": my_opinion.reason[:2000]},
            "other": {"next_close": float(target_opinion.target.next_close), "reason": target_opinion.reason[:2000]},
        }
        user_text = "컨텍스트:\n" + json.dumps(ctx, ensure_ascii=False)
        return system_text, user_text

    def _build_messages_revision(self,
                                 my_lastest: Opinion,
                                 others_latest: Dict[str, Opinion],
                                 received_rebuttals: List[Rebuttal],
                                 stock_data: StockData) -> tuple[str, str]:
        ccy = (stock_data.currency or "USD").upper()
        system_text = (
            "너는 '기술적 분석 전문가'다. 아래 컨텍스트(내 의견, 동료 의견, 반박/지지, 기술 요약)를 종합해 "
            "다음 거래일 종가(next_close)와 근거(reason)를 업데이트하라. "
            "추세/강도/신호의 일관성을 우선 고려하되, 과도한 변경은 피하라. "
            "반환은 JSON만 허용한다: {\"next_close\": number, \"reason\": string}"
        )
        me = {"agent_id": my_lastest.agent_id,
              "next_close": float(my_lastest.target.next_close),
              "reason": my_lastest.reason[:2000]}
        peers = [{
            "agent_id": aid,
            "next_close": float(op.target.next_close),
            "reason": op.reason[:2000]
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
        user_text = "컨텍스트:\n" + json.dumps(ctx, ensure_ascii=False)
        return system_text, user_text
