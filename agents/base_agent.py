import os
import time
import json
import requests
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from datetime import datetime, timezone

# .env 파일 로드 (환경변수 로드)
load_dotenv()

# 환경 변수에서 API Key 및 기타 기본 설정 불러오기
OPENAI_API_KEY = os.getenv("CAPSTONE_OPENAI_API")
OPENAI_URL = "https://api.openai.com/v1/responses"
UA = "Mozilla/5.0"  # User-Agent 헤더

# 베이스 에이전트 클래스 정의
class BaseAgent:
    def __init__(
        self,
        model: str | None = None,                     # 강제 사용할 모델 지정 (기본 없음)
        preferred_models: list[str] | None = None,    # 후보 모델 리스트
        temperature: float = 0.2,                     # 생성 temperature 설정
        bounds_tolerance: float = 0.15,               # 가격 허용 오차 비율 (±15%)
        reask_on_inconsistent: bool = True,           # 예측값이 비정상일 경우 재질문 여부
        price_period: str = "3mo",                    # 가격 통계 추출 범위
        use_price_snapshot: bool = True               # yfinance로 price_period 기간 가격 스탭샷(최고가, 최저가) 가져 
    ):
        # API 키 로드 및 유효성 검사
        OPENAI_API_KEY = os.getenv("CAPSTONE_OPENAI_API")
        if not OPENAI_API_KEY:
            raise RuntimeError("환경변수 OPENAI_API_KEY가 필요합니다.")

        # API 요청 헤더 세팅
        self.headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # 주요 하이퍼파라미터 저장
        self.temperature = temperature
        self.bounds_tolerance = bounds_tolerance
        self.reask_on_inconsistent = reask_on_inconsistent
        self.price_period = price_period
        self.use_price_snapshot = use_price_snapshot 

        # 사용할 모델 순위 설정
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [m for m in self.preferred_models if m != model]

        # 출력값 스키마 정의 (모든 Agent가 공유 가능)
        self.schema_obj = {
            # 최상위 응답은 객체(JSON 오브젝트)여야 함
            "type": "object",
            
            # 객체 어떤 속성들이 존재할 수 있는지를 정의
            "properties": {
                "buy_price":  {"type": "number", "description": "오늘 기준 목표 매수가"},
                "sell_price": {"type": "number", "description": "오늘 기준 목표 매도가"},
                "reason":     {"type": "string", "description": "근거 요약 (한국어 4~5문장)"}
            },
            # 객체는 아래 속성을 반드시 포함할 것
            "required": ["buy_price", "sell_price", "reason"],  
            
            # 그외 속성은 포함하지 않음
            "additionalProperties": False
        }

    # ticker 정규화 코드
    def _normalize_ticker(self, ticker: str) -> str:
        # ticker 양옆 공백제거, 대문자로 변경
        t = ticker.strip().upper()
        
        # 한국 6자리면 기본 KOSPI
        if t.isdigit() and len(t) == 6:
            return t + ".KS"  
        return t

    # 통화 단위와 소수점 자리 결정
    def _detect_currency_and_decimals(self, ticker: str) -> tuple[str, int]:
        try:
            info = yf.Ticker(ticker).info
            ccy = (info.get("currency") or "KRW").upper()
        except Exception:
            ccy = "KRW"
        decimals = 0 if ccy in ("KRW", "JPY") else 2
        return ccy, decimals

    # OpenAI API 호출 with fallback (모델 우선순위 순회)
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        body_base = {
            "input": [msg_sys, msg_user],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ValuationTargets",
                    "strict": True,
                    "schema": self.schema_obj
                }
            },
            "temperature": self.temperature
        }

        last_err = None
        for m in self.preferred_models:
            body = dict(body_base, model=m)
            r = requests.post(OPENAI_URL, json=body, headers=self.headers, timeout=120)
            if r.ok:
                return r.json()
            if r.status_code in (400, 404):
                last_err = (r.status_code, r.text)
                continue
            r.raise_for_status()
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    # API 응답에서 가격/근거 추출 및 형식 변환
    def _parse_result(self, resp_json: dict, decimals: int) -> tuple[float, float, str]:
        txt = resp_json.get("output_text")
        if not txt:
            txt = resp_json["output"][0]["content"][0]["text"]
        obj = json.loads(txt)
        buy = round(float(obj["buy_price"]), decimals)
        sell = round(float(obj["sell_price"]), decimals)
        reason = obj["reason"]
        return buy, sell, reason

    # 예측값 정합성 검사 (범위와 순서 확인)
    def _sanity_check(self, buy: float, sell: float, price_stats: dict) -> bool:
        if not price_stats:
            return True
        last = price_stats["close_last"]
        lo, hi = price_stats["close_min_3mo"], price_stats["close_max_3mo"]
        tol = self.bounds_tolerance
        span_ok = (lo*(1-tol) <= buy <= hi*(1+tol)) and (lo*(1-tol) <= sell <= hi*(1+tol))
        order_ok = sell >= buy
        # 추가로, 극단적으로 현재가와 동떨어지면 경고
        # (옵션) abs(buy-last)/last <= 50% 등도 넣을 수 있음
        return span_ok and order_ok

    # 예측값을 허용 범위 내로 보정 (클리핑)
    def _clip_to_bounds(self, buy: float, sell: float, price_stats: dict, decimals: int) -> tuple[float, float]:
        lo, hi = price_stats["close_min_3mo"], price_stats["close_max_3mo"]
        tol = self.bounds_tolerance
        lo_b = lo * (1 - tol)
        hi_b = hi * (1 + tol)
        buy2 = min(max(buy, lo_b), hi_b)         # buy를 lo_b ~ hi_b 범위로 클리핑
        sell2 = min(max(sell, buy2), hi_b)       # sell은 buy 이상 hi_b 이하
        return round(buy2, decimals), round(sell2, decimals)

    # 특정 티커의 과거 3개월 가격 데이터 통계 반환
    def _get_price_snapshot(self, ticker: str) -> dict:
        st = yf.Ticker(ticker)
        hist = st.history(period=self.price_period)
        if hist.empty:
            raise ValueError(f"{ticker}에 대한 가격 기록이 없습니다.")
        return {
            "close_last": float(hist["Close"].iloc[-1]),
            "close_mean_3mo": float(hist["Close"].mean()),
            "close_min_3mo": float(hist["Close"].min()),
            "close_max_3mo": float(hist["Close"].max()),
            "vol_mean_3mo": float(hist["Volume"].mean()),
        }
        
    def _add_constraints(self, user_message: dict, price_stats: dict | None, decimals: int) -> dict:
        """
        LLM이 비합리적인 값을 내놨을 때 재질문 메시지에 제약을 추가하는 메서드.
        - price_stats가 있으면: buy/sell 모두 허용 범위 내 + sell ≥ buy
        - price_stats가 없으면: sell ≥ buy만
        """
        if not price_stats:
            extra = "\n\n추가 제약: sell_price ≥ buy_price 조건을 반드시 지켜라."
            return {"role": "user", "content": user_message["content"] + extra}

        lo, hi = price_stats["close_min_3mo"], price_stats["close_max_3mo"]
        tol = self.bounds_tolerance
        lo_ok = round(lo * (1 - tol), decimals)
        hi_ok = round(hi * (1 + tol), decimals)

        extra = (
            f"\n\n추가 제약: buy_price와 sell_price 모두 {lo_ok}~{hi_ok} 범위 내에서 제시하고, "
            "sell_price ≥ buy_price 조건을 반드시 지켜라."
        )
        return {"role": "user", "content": user_message["content"] + extra}