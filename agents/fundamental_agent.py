import json
import numpy as np
import pandas as pd
import yfinance as yf
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple

class FundamentalAgent(BaseAgent):
    # ------------------------------------------------------------------
    # 1) 데이터 수집 (LLM-only 지향 + 앵커용 현재가)
    #    - 최근 3년 관점의 기초 펀더멘털 스냅샷을 LLM으로 생성
    #    - 현재가/통화만 얇게 주입(없어도 진행 가능)
    # ------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        t = self._normalize_ticker(ticker)
        self._p(f"[Fundamental.searcher-LLM] {t}")

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

        # LLM에게 3년 펀더멘털 요약을 요청하는 스키마
        schema_fund = {
            "type": "object",
            "properties": {
                "quality":   {"type": "string", "description": "사업/경쟁력/진입장벽/리텐션 등 질적 요약"},
                "growth":    {"type": "string", "description": "최근 3년 성장 동인(매출/사용자/신제품/지역)"},
                "profit":    {"type": "string", "description": "수익성 흐름(흑전/적지/마진 개선/악화)"},
                "leverage":  {"type": "string", "description": "재무 건전성(부채/현금흐름/유동성)"},
                "valuation": {"type": "string", "description": "밸류에이션 관점(동종업계 대비 고평가/저평가 논리)"},
                "summary":   {"type": "string", "description": "핵심 한글 요약 4~6문장"},
                "evidence":  {"type": "string", "description": "근거/이벤트(실적 시즌, 가이던스, M&A, 규제 등)"},
            },
            # strict=True에 맞춤: properties에 있는 키 전부 required로 지정
            "required": ["quality", "growth", "profit", "leverage", "valuation", "summary", "evidence"],
            "additionalProperties": False,
        }

        system_text = (
            "너는 '펀더멘털(가치) 분석 전문가'다. 특정 종목의 최근 3년 사업/실적/현금흐름/재무지표/리스크를 "
            "질적·정성적으로 요약하라. 실제 수치가 없으면 임의로 만들지 말고, 방향(개선/악화)과 논리만 제시하라. "
            "동종업계 대비 밸류에이션 시각(고평가/저평가의 논리)도 포함하라. 반환은 JSON만 허용한다."
        )
        user_text = "컨텍스트:\n" + json.dumps({
            "ticker": t,
            "last_price_hint": last_price,
            "currency": ccy,
            "instruction": "최근 3년의 질적 변화 중심(성장/수익성/현금흐름/부채/규제/제품/경쟁). 특정 수치/분기값은 추정하지 말 것.",
            "output": {"fields": ["quality","growth","profit","leverage","valuation","summary","evidence"]}
        }, ensure_ascii=False)

        parsed = self._ask_with_fallback(
            self._msg("system", system_text),
            self._msg("user",   user_text),
            schema_fund
        )

        fundamental = {
            "quality":   parsed.get("quality", ""),
            "growth":    parsed.get("growth", ""),
            "profit":    parsed.get("profit", ""),
            "leverage":  parsed.get("leverage", ""),
            "valuation": parsed.get("valuation", ""),
            "summary":   parsed.get("summary", ""),
            "evidence":  parsed.get("evidence", ""),
        }

        self._last_ticker = t
        sd = StockData(
            sentimental={},
            fundamental=fundamental,  # ← 펀더멘털 요약을 여기 담아 전달
            technical={},
            last_price=last_price,    # 앵커용
            currency=ccy
        )
        self.stockdata = sd
        return sd

    # ------------------------------------------------------------------
    # 2) 1차 예측 (LLM-only)
    #    - 펀더멘털 요약 + 현재가 앵커로 next_close 산출
    # ------------------------------------------------------------------
    def predicter(self, stock_data: StockData) -> Target:
        self._p("[Fundamental.predicter-LLM] next_close from fundamentals summary")

        ccy = (stock_data.currency or "USD").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2
        last = float(stock_data.last_price or 0.0)

        sys_text = (
            "너는 '펀더멘털(가치) 분석 전문가'다. 아래 최근 3년 요약과 현재가를 기준으로 "
            "다음 거래일 종가(next_close)를 예측하고, 이유(reason)를 한국어 3~4문장으로 제시하라. "
            "규칙: (1) 현재가를 앵커로 판단, (2) 단기 이벤트가 명시되지 않으면 ±2~3% 내 보수적 판단, "
            "(3) 큰 변동을 예측할 땐 근거 이벤트를 이유에 명확히 적을 것. 반환은 JSON(next_close:number, reason:string)만."
        )
        ctx = {
            "ticker": getattr(self, "_last_ticker", "UNKNOWN"),
            "currency": ccy,
            "last_price": last,
            "fundamental_summary": stock_data.fundamental,  # quality/growth/profit/leverage/valuation/summary/evidence
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
    # 3) Opinion 메시지 빌드 (가치 관점)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()
        last = float(stock_data.last_price or 0.0)

        system_text = (
            "너는 '펀더멘털(가치) 분석 전문가'다. 제공된 최근 3년 요약과 현재가를 바탕으로 "
            "예측치(next_close)에 대한 근거(reason)를 한국어 4~5문장으로 작성하라. "
            "이유에는 현재가 대비 예상 %변화와 그 근거(가이던스/마진 추세/현금흐름/밸류에이션 관점)를 포함하라. "
            "반환은 JSON(next_close:number, reason:string)만 허용한다."
        )
        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_price": last,
            "fundamental_summary": stock_data.fundamental or {},
            "our_prediction": float(target.next_close),
            "guideline": "보수적 기본(±2~3%), 강한 이벤트가 있을 때만 범위 초과."
        }
        user_text = "아래 컨텍스트를 참고하여 JSON으로만 반환:\n" + json.dumps(ctx, ensure_ascii=False)
        return system_text, user_text

    # ------------------------------------------------------------------
    # 4) Rebuttal/Revision (가치 관점 문구)
    # ------------------------------------------------------------------
    def _build_messages_rebuttal(self,
                                 my_opinion: Opinion,
                                 target_agent: str,
                                 target_opinion: Opinion,
                                 stock_data: StockData) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()
        system_text = (
            "당신은 '펀더멘털(가치) 분석 전문가'다. 두 의견의 가치 가정(성장/마진/현금흐름/레버리지/밸류에이션)과 "
            "단기 이벤트 해석이 일관적인지 비교해 'REBUT' 또는 'SUPPORT'를 결정하고, 근거를 한국어 4~5문장으로 요약하라. "
            "반환은 JSON({'stance':'REBUT|SUPPORT','message':string})만 허용한다."
        )
        ctx = {
            "ticker": t,
            "currency": ccy,
            "fundamental_summary": stock_data.fundamental or {},
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
            "너는 '펀더멘털(가치) 분석 전문가'다. 아래 컨텍스트(내 의견, 동료 의견, 반박/지지, 펀더멘털 요약)를 종합해 "
            "다음 거래일 종가(next_close)와 근거(reason)를 업데이트하라. "
            "밸류에이션 논리와 이벤트의 단기 반영 가능성을 균형 있게 고려하라. "
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
            "fundamental_summary": stock_data.fundamental or {},
            "currency": ccy
        }
        user_text = "컨텍스트:\n" + json.dumps(ctx, ensure_ascii=False)
        return system_text, user_text
