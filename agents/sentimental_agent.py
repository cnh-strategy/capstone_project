import json
import numpy as np
import pandas as pd
import yfinance as yf
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple
class SentimentalAgent(BaseAgent):
    # ------------------------------------------------------------------
    # 1) 데이터 수집: 티커 기준 시세를 내려받아 기술적 신호를 계산
    #    - 반환 포맷은 프로젝트 공용 StockData(dataclass) 사용
    # ------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        t = self._normalize_ticker(ticker)
        self._p(f"[searcher] {t}")

        # 최근 3개월, 1일봉 다운로드 (배당/액면 등 auto_adjust=False 원시가 유지)
        df = yf.download(t, period="3mo", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError(f"가격 데이터를 불러오지 못했습니다: {t}")
        df = df.dropna().copy()

        # --- 기본 신호들 계산 ---
        # 모멘텀: 5/20일 수익률
        df["mom_5"]  = df["Close"] / df["Close"].shift(5)  - 1
        df["mom_20"] = df["Close"] / df["Close"].shift(20) - 1
        # RSI(14)
        df["rsi_14"] = self._rsi(df["Close"], 14)
        # 거래량 정규화(20일 평균 대비 배수, 이상값 클리핑)
        df["vol_norm20"] = (df["Volume"] / df["Volume"].rolling(20).mean()).clip(0, 5)
        df = df.dropna()

        # FutureWarning 방지: iloc[-1] → .item()으로 스칼라화
        last_close = float(df["Close"].iloc[-1].item())
        ccy, _ = self._detect_currency_and_decimals(t)

        # 마지막 컨텍스트(디버깅/프롬프트용)
        self._last_df = df
        self._last_ticker = t

        # StockData(공용 포맷)로 반환
        return StockData(
            sentimental={},  # TODO: 감성 점수 주입 예정
            fundamental={},  # TODO: 재무 요약 주입 예정
            technical={
                "mom_5":      float(df["mom_5"].iloc[-1]),
                "mom_20":     float(df["mom_20"].iloc[-1]),
                "rsi_14":     float(df["rsi_14"].iloc[-1]),
                "vol_norm20": float(df["vol_norm20"].iloc[-1]),
            },
            last_price=last_close,
            currency=ccy
        )

    # ------------------------------------------------------------------
    # 2) 1차 예측: 간단한 규칙으로 next_close 산출(Target)
    #    - 모멘텀, RSI를 조합해 기대수익률을 계산 후 클리핑
    # ------------------------------------------------------------------
    def predicter(self, stock_data: StockData) -> Target:
        sig  = stock_data.technical
        last = float(stock_data.last_price or 0.0)

        momentum = 0.6 * sig.get("mom_5", 0.0) + 0.4 * sig.get("mom_20", 0.0)
        rsi      = float(sig.get("rsi_14", 50.0))
        rsi_adj  = 0.5 if rsi > 70 else (1.5 if rsi < 30 else 1.0)  # 과매수/과매도 가중

        k = 0.5  # 민감도
        expected_return = float(np.clip(k * momentum * rsi_adj, -0.05, 0.05))  # ±5% 제한
        decimals = 0 if (stock_data.currency or "KRW").upper() in ("KRW", "JPY") else 2
        next_close = round(last * (1 + expected_return), decimals)

        self._p(f"[predicter] last={last:.4f}, mom={momentum:.4f}, rsi={rsi:.1f} → next={next_close}")
        return Target(next_close=next_close)

    # ------------------------------------------------------------------
    # 3) LLM 메시지 빌드(Opinion): 다음날 종가와 근거를 JSON으로 요구
    #    - 시스템: 역할/출력형식 고정
    #    - 사용자: 컨텍스트(JSON 직렬화 가능 타입만)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        system_text = (
            "너는 주식 애널리스트다. 입력된 신호(모멘텀/RSI/거래량)와 최근 종가를 바탕으로 "
            "다음 거래일 종가(next_close)에 대한 근거(reason)를 한국어 4~5문장으로 작성한다. "
            "반환은 JSON(next_close:number, reason:string)만 허용한다."
        )
        # ⚠️ JSON 직렬화 가능한 원시 타입만 사용
        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_close": float(stock_data.last_price or 0.0),
            "signals": {k: float(v) for k, v in (stock_data.technical or {}).items()},
            "our_prediction": float(target.next_close),
            "format_rule": f"숫자는 소수 {decimals}자리, 통화 {ccy}"
        }
        user_text = "아래 컨텍스트를 참고하여 next_close와 reason을 JSON으로만 반환해라.\n" + json.dumps(ctx, ensure_ascii=False)
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

        system_text = (
            "당신은 '감성분석 투자 전문가'다. 두 의견의 논리와 감성 가정(이벤트 해석, 투자자 정서 추정)이 "
            f"내 의견(next_close, reason)과 {target_agent}의 의견(next_close, reason)을 비교 평가하고 "
            "'REBUT'(반박) 또는 'SUPPORT'(지지) 중 하나를 결정하라. "
            "판단 근거는 한국어 4~5문장으로 작성한다. "
            "반환은 JSON({'stance':'REBUT|SUPPORT','message':string})만 허용한다."
        )

        # 안전한 직렬화를 위해 원시 타입으로 변환
        me_next  = float(my_opinion.target.next_close)
        oth_next = float(target_opinion.target.next_close)

        ctx = {
            "ticker": t,
            "me": {
                "agent_id": self.agent_id,
                "next_close": round(me_next, decimals),
                "reason": str(my_opinion.reason)[:2000],
            },
            "other": {
                "agent_id": target_agent,
                "next_close": round(oth_next, decimals),
                "reason": str(target_opinion.reason)[:2000],
            },
            # 스냅샷(선택): LLM 힌트용 컨텍스트
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
        user_text = "다음 컨텍스트를 평가하여 JSON 객체만 반환하세요:\n" + json.dumps(ctx, ensure_ascii=False)
        return system_text, user_text
    
    
    def _build_messages_revision(self,
                             my_lastest: Opinion,
                             others_latest: Dict[str, Opinion],
                             received_rebuttals: List[Rebuttal],
                             stock_data: StockData) -> tuple[str, str]:
        """LLM에게 최종 next_close/이유 수정을 맡기기 위한 system/user 메시지"""
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        # 입력을 모두 원시 타입으로 직렬화
        me = {
            "agent_id": my_lastest.agent_id,
            "next_close": float(my_lastest.target.next_close),
            "reason": str(my_lastest.reason)[:2000],
        }
        peers = []
        for aid, op in (others_latest or {}).items():
            peers.append({
                "agent_id": str(aid),
                "next_close": float(op.target.next_close),
                "reason": str(op.reason)[:2000],
            })
        feedback = [{
            "from": r.from_agent_id,
            "to":   r.to_agent_id,
            "stance": r.stance,
            "message": str(r.message)[:500],
        } for r in (received_rebuttals or [])]

        system_text = (
            "너는 주식 애널리스트다. 아래 컨텍스트(내 의견, 동료 의견, 받은 반박/지지, 신호 스냅샷)를 종합해 "
            "내 다음 거래일 종가 예측(next_close)과 근거(reason)를 **업데이트**한다. "
            "규칙:\n"
            f"- 숫자는 소수 {decimals}자리로 반올림.\n"
            "- 동료 의견을 맹목 추종하지 말 것. SUPPORT/REBUT 비중과 기술 신호를 함께 고려.\n"
            "- 과도한 변경은 금지(기존 대비 ±7% 이내 권장). 필요 시 근거를 분명히 작성.\n"
            "반환은 JSON만 허용한다: {\"next_close\": number, \"reason\": string}"
        )

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
        user_text = "아래 컨텍스트를 반영하여 수정된 next_close와 reason을 JSON으로만 반환하라:\n" + json.dumps(ctx, ensure_ascii=False)
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
