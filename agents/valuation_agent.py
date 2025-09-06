# BaseAgent 임포트
from .base_agent import BaseAgent

# 필요라이브러리 임포트
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


class ValuationAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # 에이전트의 메인 실행 메서드: 티커 기반 예측 수행
    def run(self, ticker: str) -> list:
        # 1. 티커 정규화 (예: 005930 → 005930.KS)
        tkr = self._normalize_ticker(ticker)

        # 2. 통화 단위 및 소수점 자릿수 판별 (KRW → 0자리, USD → 2자리)
        currency, decimals = self._detect_currency_and_decimals(tkr)

        # 3. 최근 3개월 가격 스냅샷 가져오기 (사용 여부는 self.use_price_snapshot에 따름)
        price = self._get_price_snapshot(tkr) if self.use_price_snapshot else None

        # 4. 시스템 메시지 및 유저 메시지 생성
        context = self._build_context(tkr, price)
        msg_sys, msg_user = self._build_messages(context, currency, decimals)

        # 5. GPT API 요청 → 응답 JSON 파싱
        result = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj)
        buy, sell, reason = self._parse_result(result, decimals)

        # 6. 정합성 검사 및 보정 (범위 초과 or 매도가 < 매수가 → 재질문 or 클리핑)
        if not self._sanity_check(buy, sell, price):
            if self.reask_on_inconsistent:
                # 제약 추가하여 재질문
                msg_user2 = self._add_constraints(msg_user, price, decimals)
                result = self._ask_with_fallback(msg_sys, msg_user2, self.schema_obj)
                buy, sell, reason = self._parse_result(result, decimals)

                # 재질문도 정합성 미충족 시 클리핑 보정
                if not self._sanity_check(buy, sell, price):
                    buy, sell = self._clip_to_bounds(buy, sell, price, decimals)
            else:
                # 재질문 비활성화 시 바로 클리핑
                buy, sell = self._clip_to_bounds(buy, sell, price, decimals)

        # 7. 최종 결과 반환
        return [buy, sell, reason]

    # 프롬프트 Context 구성: GPT에게 전달할 가격 정보 및 실행 시간 포함
    def _build_context(self, ticker: str, price: dict | None) -> str:
        lines = [
            f"[RUN_AT_UTC] {datetime.now(timezone.utc).isoformat()}",
            f"[TICKER] {ticker}"
        ]
        if price:
            lines.append(f"[PRICE_3MO] {price}")
        return "\n".join(lines)

    # GPT 메시지 포맷 생성: 시스템 + 유저 요청
    def _build_messages(self, context: str, currency: str, decimals: int) -> tuple[dict, dict]:
        sys = (
            "너는 기업의 펀더멘털 정보를 기반으로"
            "다음 거래일의 목표 매수/매도가를 제시하는 애널리스트다. "
            "매수 목표액 도달시 구매, 매도 목표액 도달시 판매, 매도 목표액 미도달시 종가에 전부 판매한다"
            "수익을 극대화 할 수 있도록 매수/매도가를 제시해라"
            f"통화는 {currency}이며, 숫자는 소수 {decimals}자리로 제시한다. "
            "결과는 JSON 객체로만 반환한다."
            "매매 전략은 "
        )
        user = (
            "입력값: 최근 3개월 가격 데이터"
            "요구사항:\n"
            "1) 오늘 기준 buy_price(number), sell_price(number) 예측 (금일 목표)\n"
            "2) reason(string) 4~5문장으로 요약\n"
            "3) 다른 텍스트 없이 JSON 객체만 반환\n"
            "4) 한국말로 설명할 것 \n\n"
            f"{context}"
        )
        return {"role": "system", "content": sys}, {"role": "user", "content": user}
    
if __name__ == "__main__":
    agent = ValuationAgent()
    result = agent.run("RZLV")
    print(result)
