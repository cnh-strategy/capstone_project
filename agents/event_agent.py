import requests
import xml.etree.ElementTree as ET
import yfinance as yf
from datetime import datetime, timezone, timedelta
from agents.base_agent import BaseAgent   # BaseAgent 상속

class EventAgent(BaseAgent):
    """
    EventAgent는 뉴스/매크로/가격 데이터를 기반으로
    단기 매수/매도가를 예측하는 분석 에이전트
    """

    def __init__(self, news_lang="ko", news_country="KR", max_news=20, use_macro=True, **kwargs):
        """
        생성자
        - news_lang: 뉴스 검색 언어
        - news_country: 뉴스 검색 국가
        - max_news: 가져올 뉴스 개수
        - use_macro: 거시경제 데이터(FRED) 사용 여부
        - kwargs: BaseAgent 공통 인자들
        """
        super().__init__(**kwargs)  # BaseAgent 초기화
        self.news_lang = news_lang
        self.news_country = news_country
        self.max_news = max_news
        self.use_macro = use_macro

        # FRED 기본 지표 (미국 CPI, 실업률, 금리)
        self.fred_series_map = {
            "CPI(US)": "CPIAUCSL",
            "Unemployment(US)": "UNRATE",
            "PolicyRate(US)": "FEDFUNDS",
        }

    # ---------- Public ----------
    def run(self, ticker: str) -> list:
        """
        메인 실행 함수
        1) 뉴스/매크로/가격 데이터 수집
        2) context + 메시지 생성
        3) GPT 요청 및 응답 파싱
        4) 결과 정합성 검사 및 보정
        """
        tkr = self._normalize_ticker(ticker)

        # 1. 데이터 수집
        news = self._fetch_google_news(tkr)
        macro = self._fetch_macro_snapshot(self.fred_series_map) if self.use_macro else {}
        price = self._get_price_snapshot(tkr) if self.use_price_snapshot else None

        # 2. 컨텍스트 생성
        currency, decimals = self._detect_currency_and_decimals(tkr)
        context = self._build_context(tkr, news, macro, price)

        # 3. 메시지 생성 → GPT 호출
        msg_sys, msg_user = self._build_messages(context, currency, decimals)
        result = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj)

        # 4. 응답 파싱
        buy, sell, reason = self._parse_result(result, decimals)

        # 5. 정합성 검사 및 보정
        if not self._sanity_check(buy, sell, price):
            if self.reask_on_inconsistent:
                msg_user2 = self._add_constraints(msg_user, price, decimals)
                result = self._ask_with_fallback(msg_sys, msg_user2, self.schema_obj)
                buy, sell, reason = self._parse_result(result, decimals)
                if not self._sanity_check(buy, sell, price):
                    buy, sell = self._clip_to_bounds(buy, sell, price, decimals)
            else:
                buy, sell = self._clip_to_bounds(buy, sell, price, decimals)

        return [buy, sell, reason]

    # ---------- Data: News / Macro ----------
    def _fetch_google_news(self, query: str) -> list:
        """
        구글 뉴스 RSS에서 특정 쿼리 관련 뉴스 가져오기
        - query: 검색어 (보통 ticker/기업명)
        - 반환: 뉴스 리스트 [{title, summary, url}, ...]
        """
        base = "https://news.google.com/rss/search"
        params = {"q": query, "hl": self.news_lang, "gl": self.news_country, "ceid": f"{self.news_country}:{self.news_lang}"}
        r = requests.get(base, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        items = []
        root = ET.fromstring(r.content)
        for item in root.findall(".//item")[:self.max_news]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            items.append({"title": title, "summary": desc[:400], "url": link})
        return items

    def _fetch_macro_snapshot(self, series_map: dict) -> dict:
        """
        FRED API를 이용해 주요 거시경제 지표 최신 값 가져오기
        - series_map: {이름: 시리즈ID}
        - 반환: {이름: 값}
        """
        snap = {}
        for name, sid in series_map.items():
            val = self._fetch_fred_latest(sid)
            if val: snap[name] = val
        return snap

    def _fetch_fred_latest(self, series_id: str):
        """
        FRED API 호출하여 특정 지표의 최신 데이터 반환
        - series_id: FRED 지표 코드
        - 반환: {"date": 날짜, "value": 값}
        """
        from os import getenv
        FRED_API_KEY = getenv("FRED_API_KEY")
        if not FRED_API_KEY:
            return None
        base = "https://api.stlouisfed.org/fred/series/observations"
        start = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
        params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json", "observation_start": start}
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs: return None
        last = obs[-1]
        return {"date": last["date"], "value": last["value"]}

    # ---------- Context / Prompt ----------
    def _build_context(self, ticker: str, news: list, macro: dict, price: dict | None) -> str:
        """
        뉴스 + 매크로 + 가격 정보를 문자열 context로 합치기
        """
        lines = [
            f"[RUN_AT_UTC] {datetime.now(timezone.utc).isoformat()}",
            f"[TICKER] {ticker}"
        ]
        if price:
            lines.append(f"[PRICE_3MO] {price}")
        if macro:
            lines.append(f"[MACRO] {macro}")
        if news:
            lines.append("[NEWS]")
            for n in news:
                lines.append(f"- {n['title']} :: {n['summary']} (url: {n['url']})")
        return "\n".join(lines)

    def _build_messages(self, context: str, currency: str, decimals: int) -> tuple[dict, dict]:
        """
        GPT에게 전달할 system, user 메시지 구성
        - system: 역할 지시 (애널리스트 역할)
        - user: 데이터와 요구사항
        """
        sys = (
            "너는 최신 뉴스, 거시경제 지표, 최근 가격 데이터를 바탕으로 "
            "다음 거래일의 목표 매수/매도가를 제시하는 애널리스트다. "
            "매수 목표액 도달시 구매, 매도 목표액 도달시 판매, 매도 목표액 미도달시 종가에 전부 판매한다"
            "수익을 극대화 할 수 있도록 매수/매도가를 제시해라"
            f"통화는 {currency}, 숫자는 소수 {decimals}자리로 제시한다. "
            "결과는 JSON 객체로만 반환한다."
        )
        user = (
            "입력값: 뉴스 요약 + 매크로 지표 + 최근 가격 데이터\n"
            "요구사항:\n"
            "1) buy_price(number), sell_price(number) 예측 (금일 목표)\n"
            "2) reason(string) 4~5문장 (출처 요약 포함)\n"
            "3) JSON 객체만 반환\n"
            "4) 한국어 설명\n\n"
            f"{context}"
        )
        return {"role": "system", "content": sys}, {"role": "user", "content": user}
