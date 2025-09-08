import requests
import yfinance as yf
import json
from .base_agent import BaseAgent   # BaseAgent 상속 (LLM 호출 포함)

class FundamentalAgent(BaseAgent):
    """
    FMP(Financial Modeling Prep)와 Yahoo Finance 데이터를 각각 활용하여
    매수/매도 가격과 최종 투자의견(Buy/Hold/Sell)을 산출하는 에이전트.

    - Input : ticker (예: 'AAPL', '005930.KQ')
    - Output: {
        "fmp": { buy_price, sell_price, opinion, reason, raw },
        "yahoo": { buy_price, sell_price, opinion, reason, raw }
      }
    ** raw: FMP 또는 Yahoo에서 직접 받아온 최근 3년치 재무 데이터 요약본
    """

    def __init__(self, ticker: str, fmp_api_key: str, check_years: int = 3, use_llm: bool = False, **kwargs):
        super().__init__(**kwargs)  # BaseAgent 초기화 (LLM 사용 가능)
        self.ticker = ticker
        self.fmp_api_key = fmp_api_key
        self.check_years = check_years
        self.use_llm = use_llm      #llm 사용 혹은 미사용 결정 가능

    # -----------------------------
    # 데이터 수집
    # -----------------------------
    def get_fmp_ratios(self, years):
        try:
            fmp_url = (
                f"https://financialmodelingprep.com/api/v3/ratios/"
                f"{self.ticker}?period=annual&limit={years}&apikey={self.fmp_api_key}"
            )
            fmp_datas = requests.get(fmp_url).json()
            return [
                {
                    "year": fmp_data.get("date"),
                    "pe": fmp_data.get("peRatio"),
                    "pbr": fmp_data.get("priceToBookRatio"),
                    "roe": fmp_data.get("returnOnEquity"),
                    "eps": fmp_data.get("eps"),
                }
                for fmp_data in fmp_datas
            ]
        except Exception as e:
            print(f"[get_fmp_ratios]Error: {e}")

    def get_yahoo_ratios(self, years: int):
        try:
            stock = yf.Ticker(self.ticker)
            earnings = stock.earnings
            if earnings is None or earnings.empty:
                return []

            info = stock.info
            shares = info.get("sharesOutstanding")
            roe = info.get("returnOnEquity")
            pe = info.get("trailingPE")
            pbr = info.get("priceToBook")

            yahoo_result = []
            for year in earnings.index[-years:]:
                net_income = earnings.loc[year, "Earnings"]
                revenue = earnings.loc[year, "Revenue"]
                eps = net_income / shares if shares else None
                yahoo_result.append({
                    "year": str(year),
                    "revenue": revenue,
                    "net_income": net_income,
                    "eps": eps,
                    "roe": roe,
                    "pe": pe,
                    "pbr": pbr
                })
            return yahoo_result
        except Exception as e:
            print(f"[get_yahoo_ratios]Error: {e}")

    # -----------------------------
    # 보조 계산 (룰 기반)
    # -----------------------------
    def calculate_growth(self, eps_values: list) -> float:
        eps_values = [v for v in eps_values if v is not None]
        if len(eps_values) >= 2 and eps_values[0] > 0 and eps_values[-1] > 0:
            years = len(eps_values) - 1
            return (eps_values[-1] / eps_values[0]) ** (1 / years) - 1
        return 0

    def calculate_roe_avg(self, roe_values: list) -> float:
        roe_values = [v for v in roe_values if v is not None]
        return sum(roe_values) / len(roe_values) if roe_values else 0

    def judge(self, eps_values, roe_values, pe_latest, pbr_latest, open_price, close_price):
        try:
            growth = self.calculate_growth(eps_values)
            roe_avg = self.calculate_roe_avg(roe_values)

            if growth > 0.05 and roe_avg > 0.1 and pe_latest and pe_latest < 15 and pbr_latest and pbr_latest < 2:
                opinion = "Buy"
                buy_price = open_price
                sell_price = close_price * 1.05
                reason = f"EPS 성장률 {growth:.1%}, ROE 평균 {roe_avg:.1%}, PER {pe_latest:.1f}, PBR {pbr_latest:.1f} → 저평가 + 성장성 우수."
            elif (growth <= 0 or roe_avg <= 0.05 or (pe_latest and pe_latest >= 30) or (pbr_latest and pbr_latest >= 4)):
                opinion = "Sell"
                buy_price = open_price
                sell_price = close_price * 0.95
                reason = f"EPS 성장률 {growth:.1%}, ROE 평균 {roe_avg:.1%}, PER {pe_latest}, PBR {pbr_latest} → 성장/수익성 부족, 고평가 위험."
            else:
                opinion = "Hold"
                buy_price = open_price
                sell_price = close_price
                reason = f"EPS 성장률 {growth:.1%}, ROE 평균 {roe_avg:.1%}, PER {pe_latest}, PBR {pbr_latest} → 일부 긍정적이나 확신 부족."

            return {"buy_price": buy_price, "sell_price": sell_price, "opinion": opinion, "reason": reason}
        except Exception as e:
            print(f"[judge]Error: {e}")
            return {"buy_price": None, "sell_price": None, "opinion": "Error", "reason": str(e)}


    # -----------------------------
    # 메인 실행
    # -----------------------------
    def fundamental_main_analyze(self, open_price: float, close_price: float) -> dict:
        try:
            fmp_data = self.get_fmp_ratios(self.check_years)
            yahoo_data = self.get_yahoo_ratios(self.check_years)

            if not fmp_data or not yahoo_data:
                return {"fmp": {}, "yahoo": {}}

            if self.use_llm:
                # ★ 통화/자리수 자동 탐지 (BaseAgent 메서드 활용)
                currency, decimals = self._detect_currency_and_decimals(self.ticker)

                context = {
                    "fmp": fmp_data,
                    "yahoo": yahoo_data,
                    "open_price": open_price,
                    "close_price": close_price
                }

                msg_sys = {
                    "role": "system",
                    "content": (
                        "너는 전문 주식 애널리스트다. "
                        "주어진 재무 데이터(FMP, Yahoo)와 현재 주가(open/close)를 기반으로 "
                        "금일 기준 목표 매수/매도가를 산출한다. "
                        "모든 출력은 JSON 형식으로만 반환해야 한다."
                    )
                }

                msg_user = {
                    "role": "user",
                    "content": (
                        f"입력 데이터:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                        "요구사항:\n"
                        "1) buy_price(number): 오늘 기준 추천 매수가. open_price와 close_price를 고려해 제시.\n"
                        "2) sell_price(number): 오늘 기준 목표 매도가. 반드시 sell_price ≥ buy_price.\n"
                        "3) reason(string): 4~5문장 한국어 설명. EPS 성장률, ROE, PER, PBR 및 데이터 출처(FMP/Yahoo)를 반영해라.\n"
                        "4) JSON 객체만 반환. 여분의 텍스트, 설명, 마크다운 불가.\n"
                        f"5) 통화 단위는 {currency}, 숫자는 소수점 {decimals}자리까지 반올림."
                    )
                }

                result = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj)
                buy, sell, reason = self._parse_result(result, decimals)
                return {"llm": {"buy_price": buy, "sell_price": sell, "reason": reason, "raw": context}}

            else:
                fmp_sorted = sorted(fmp_data, key=lambda x: x["year"])
                eps_fmp = [d["eps"] for d in fmp_sorted]
                roe_fmp = [d["roe"] for d in fmp_sorted]
                pe_fmp = fmp_sorted[-1].get("pe")
                pbr_fmp = fmp_sorted[-1].get("pbr")
                fmp_result = self.judge(eps_fmp, roe_fmp, pe_fmp, pbr_fmp, open_price, close_price)
                fmp_result["raw"] = fmp_sorted

                eps_yahoo = [d["eps"] for d in yahoo_data]
                roe_yahoo = [d["roe"] for d in yahoo_data if d["roe"] is not None]
                pe_yahoo = yahoo_data[-1].get("pe")
                pbr_yahoo = yahoo_data[-1].get("pbr")
                yahoo_result = self.judge(eps_yahoo, roe_yahoo, pe_yahoo, pbr_yahoo, open_price, close_price)
                yahoo_result["raw"] = yahoo_data

                return {"fmp": fmp_result, "yahoo": yahoo_result}
        except Exception as e:
            print(f"[fundamental_main_analyze]Error: {e}")
            return {"fmp": {}, "yahoo": {}, "error": str(e)}



# 사용 예시
# # 룰 기반 사용
# agent1 = FundamentalAgent("AAPL", "YOUR_FMP_API_KEY", check_years=3, use_llm=False)
# print(agent1.fundamental_main_analyze(open_price=150, close_price=155))
#
# # LLM 사용
# agent2 = FundamentalAgent("AAPL", "YOUR_FMP_API_KEY", check_years=3, use_llm=True)
# print(agent2.fundamental_main_analyze(open_price=150, close_price=155))
