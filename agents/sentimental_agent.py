import requests
import time
import yfinance as yf
from datetime import datetime, timezone

from agents.base_agent import BaseAgent  # BaseAgent 상속

class SentimentalAgent(BaseAgent):
    """
    SentimentalAgent는 인터넷 커뮤니티(HackerNews/Reddit) 여론 + 가격 데이터를 기반으로
    단기 매수/매도가를 예측하는 분석 에이전트
    """

    def __init__(self, hours: int = 24, max_posts: int = 100, **kwargs):
        """
        생성자
        - hours: 몇 시간 내 게시글만 검색할지
        - max_posts: 최대 가져올 게시글 개수
        - kwargs: BaseAgent 공통 인자들
        """
        super().__init__(**kwargs)  # BaseAgent 초기화
        self.hours = hours
        self.max_posts = max_posts

    # ---------- Public ----------
    def run(self, ticker: str) -> list:
        """
        메인 실행 함수
        1) 커뮤니티 글 수집
        2) 최근 가격 스냅샷
        3) context + 메시지 생성
        4) GPT 요청 및 응답 파싱
        5) 정합성 검사 및 보정
        """
        tkr = self._normalize_ticker(ticker)

        # 1. 커뮤니티 글 수집
        query = self._build_query(tkr)
        posts = self._gather_posts(query)

        # 2. 최근 가격 스냅샷
        price = self._get_price_snapshot(tkr)

        # 3. context/message 생성
        currency, decimals = self._detect_currency_and_decimals(tkr)
        context = self._build_context(tkr, posts, price)
        msg_sys, msg_user = self._build_messages(context, currency, decimals)

        # 4. GPT 호출 → 응답 파싱
        result = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj)
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

    # ---------- Data Helpers ----------
    def _build_query(self, ticker: str) -> str:
        """
        ticker에서 기업명 추출하여 검색 쿼리 생성
        예: "AAPL" → "Apple OR AAPL"
        """
        try:
            info = yf.Ticker(ticker).info
            name = (info.get("shortName") or info.get("longName") or "").strip()
        except Exception:
            name = ""
        if name and name.lower() != ticker.lower():
            return f"{name} OR {ticker}"
        return ticker

    def _since_ts(self) -> int:
        """
        몇 시간 전 timestamp를 반환
        HackerNews/Reddit 검색 시 사용
        """
        return int(time.time() - self.hours * 3600)

    def _fetch_hn(self, query: str) -> list:
        """
        HackerNews에서 최근 글 검색
        """
        url = "https://hn.algolia.com/api/v1/search"
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": f"created_at_i>{self._since_ts()}",
            "hitsPerPage": self.max_posts
        }
        r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        items = []
        for h in r.json().get("hits", []):
            items.append({
                "source": "HackerNews",
                "title": h.get("title") or "",
                "text": (h.get("title") or "") + " " + (h.get("story_text") or ""),
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "score": h.get("points") or 0,
                "created_at": h.get("created_at") or ""
            })
        return items

    def _fetch_reddit(self, query: str) -> list:
        """
        Reddit에서 최근 글 검색
        """
        url = "https://www.reddit.com/search.json"
        params = {"q": query, "t": "day", "limit": str(self.max_posts), "sort": "new"}
        try:
            r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if not r.ok:
                return []
            data = r.json()
        except Exception:
            return []
        items = []
        for c in data.get("data", {}).get("children", []):
            d = c.get("data", {})
            items.append({
                "source": "Reddit",
                "title": d.get("title") or "",
                "text": (d.get("selftext") or "")[:1000],
                "url": "https://www.reddit.com" + (d.get("permalink") or ""),
                "score": d.get("score") or 0,
                "created_at": datetime.fromtimestamp(d.get("created_utc", 0)).isoformat()
            })
        return items

    def _gather_posts(self, query: str) -> list:
        """
        HackerNews + Reddit 글 수집 후 통합
        - 점수/최신순 정렬
        - 중복 제거
        """
        posts = []
        try: posts += self._fetch_hn(query)
        except Exception: pass
        try: posts += self._fetch_reddit(query)
        except Exception: pass

        # 점수/최신 우선 정렬
        posts = sorted(posts, key=lambda x: (x["score"], x["created_at"]), reverse=True)

        # 중복 제거
        seen, uniq = set(), []
        for p in posts:
            key = p["url"] or p["title"]
            if key in seen: continue
            seen.add(key); uniq.append(p)
            if len(uniq) >= self.max_posts:
                break
        return uniq

    # ---------- Context / Prompt ----------
    def _build_context(self, ticker: str, posts: list, price: dict) -> str:
        """
        수집한 커뮤니티 글 + 가격 스냅샷을 context 문자열로 합치기
        """
        lines = [f"[TICKER] {ticker}", f"[PRICE_SNAPSHOT] {price}"]
        for p in posts:
            line = f"- ({p['source']}) {p['title']} :: {p['text']} (url: {p['url']})"
            lines.append(line)
            if sum(len(x) for x in lines) > 7000:  # 최대 길이 제한
                break
        return "\n".join(lines)

    def _build_messages(self, context: str, currency: str, decimals: int) -> tuple[dict, dict]:
        """
        GPT에 보낼 system/user 메시지 구성
        """
        sys = (
            "너는 인터넷 커뮤니티 여론과 최근 가격 데이터를 바탕으로 "
            "다음 거래일의 목표 매수/매도가를 제시하는 애널리스트다. "
            "매수 목표액 도달시 구매, 매도 목표액 도달시 판매, 매도 목표액 미도달시 종가에 전부 판매한다"
            "수익을 극대화 할 수 있도록 매수/매도가를 제시해라"
            f"통화는 {currency}, 숫자는 소수 {decimals}자리로 제시한다. "
            "근거 수치와 예측 수치가 논리적으로 일치해야 한다. "
            "결과는 JSON 객체로만 반환한다."
        )
        user = (
            "입력값: 커뮤니티 글 요약 + 최근 3개월 가격 스냅샷\n"
            "요구사항:\n"
            "1) buy_price(number), sell_price(number) 예측 (금일 목표)\n"
            "2) reason(string) 4~5문장 (출처 요지 포함)\n"
            "3) JSON 객체만 반환\n"
            "4) 한국말 설명\n\n"
            f"{context}"
        )
        return {"role": "system", "content": sys}, {"role": "user", "content": user}
