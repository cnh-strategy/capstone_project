import requests
import yfinance as yf

class FundamentalAgent:
    """
    FundamentalAgent
    ----------------
    FMP(Financial Modeling Prep)와 Yahoo Finance 데이터를 각각 활용하여
    매수/매도 가격과 최종 투자의견(Buy/Hold/Sell)을 산출하는 에이전트.

    - Input : ticker (예: 'AAPL', '005930.KQ')
    - Output: {
        "fmp": { buy_price, sell_price, opinion, reason, raw },
        "yahoo": { buy_price, sell_price, opinion, reason, raw }
      }
    """

    def __init__(self, ticker: str, fmp_api_key: str, check_fmp_data: int = 3):
        """
        :param ticker: 종목 티커 (예: 'AAPL', '005930.KQ')
        :param fmp_api_key: FMP API Key
        :param check_fmp_data: FMP에서 가져올 연간 데이터 개수 (기본값=3년)
        """
        self.ticker = ticker
        self.fmp_api_key = fmp_api_key
        self.check_fmp_data = check_fmp_data

    # -----------------------------
    # 데이터 수집
    # -----------------------------
    def get_fmp_ratios(self):
        """
        FMP API에서 최근 n년 연간 재무 비율 데이터 가져오기
        - PER, PBR, ROE, EPS 포함
        - raw 데이터 그대로 보관 → 결과에 포함
        """
        url = (
            f"https://financialmodelingprep.com/api/v3/ratios/"
            f"{self.ticker}?period=annual&limit={self.check_fmp_data}&apikey={self.fmp_api_key}"
        )
        data = requests.get(url).json()
        return [
            {
                "year": d.get("date"),
                "pe": d.get("peRatio"),
                "pbr": d.get("priceToBookRatio"),
                "roe": d.get("returnOnEquity"),
                "eps": d.get("eps"),
            }
            for d in data
        ]

    def get_yahoo_ratios(self):
        """
        Yahoo Finance(yfinance)에서 최근 3년 데이터 가져오기
        - earnings: 매출, 순이익 (최근 4년치)
        - EPS: net_income ÷ sharesOutstanding 으로 직접 계산
        - ROE, PER, PBR: info 속성에서 가져오기
        """
        stock = yf.Ticker(self.ticker)
        earnings = stock.earnings
        shares = stock.info.get("sharesOutstanding", None)

        result = []
        for year in earnings.index[-3:]:
            net_income = earnings.loc[year, "Earnings"]
            revenue = earnings.loc[year, "Revenue"]
            eps = net_income / shares if shares else None
            roe = stock.info.get("returnOnEquity")
            pe = stock.info.get("trailingPE")
            pbr = stock.info.get("priceToBook")
            result.append({
                "year": str(year),
                "revenue": revenue,
                "net_income": net_income,
                "eps": eps,
                "roe": roe,
                "pe": pe,
                "pbr": pbr
            })
        return result

    # -----------------------------
    # 보조 계산
    # -----------------------------
    def calculate_growth(self, eps_values: list) -> float:
        """
        EPS 성장률(CAGR) 계산
        - None 값 제거
        - 값이 2개 이상일 때만 CAGR 계산
        - 과거 → 최신 순서 가정
        """
        eps_values = [v for v in eps_values if v is not None]
        if len(eps_values) >= 2 and eps_values[0] > 0 and eps_values[-1] > 0:
            years = len(eps_values) - 1
            return (eps_values[-1] / eps_values[0]) ** (1 / years) - 1
        return 0

    def calculate_roe_avg(self, roe_values: list) -> float:
        """
        ROE 평균 계산
        - None 값 제외
        - 값이 없으면 0 반환
        """
        roe_values = [v for v in roe_values if v is not None]
        return sum(roe_values) / len(roe_values) if roe_values else 0

    def judge(self, eps_values, roe_values, pe_latest, pbr_latest, open_price, close_price):
        """
        모든 지표(EPS 성장률, ROE, PER, PBR)를 반영하여
        매수/매도 가격 및 최종 투자의견(Buy/Hold/Sell) 판단
        """
        growth = self.calculate_growth(eps_values)
        roe_avg = self.calculate_roe_avg(roe_values)

        # 강력 매수 조건: 성장성 + 수익성 + 저평가
        if growth > 0.05 and roe_avg > 0.1 and pe_latest and pe_latest < 15 and pbr_latest and pbr_latest < 2:
            opinion = "Buy"
            buy_price = open_price
            sell_price = close_price * 1.05  # 목표가: 종가 대비 +5%
            reason = (
                f"EPS 성장률 {growth:.1%}, ROE 평균 {roe_avg:.1%}, "
                f"PER {pe_latest:.1f}, PBR {pbr_latest:.1f} → 저평가 + 성장성 우수."
            )

        # 매도 조건: 성장성 부재 또는 고평가
        elif (growth <= 0 or roe_avg <= 0.05 or (pe_latest and pe_latest >= 30) or (pbr_latest and pbr_latest >= 4)):
            opinion = "Sell"
            buy_price = open_price
            sell_price = close_price * 0.95  # 손절: 종가 대비 -5%
            reason = (
                f"EPS 성장률 {growth:.1%}, ROE 평균 {roe_avg:.1%}, "
                f"PER {pe_latest}, PBR {pbr_latest} → 성장/수익성 부족, 고평가 위험."
            )

        # 중립 조건: 일부 긍정적이지만 불확실
        else:
            opinion = "Hold"
            buy_price = open_price
            sell_price = close_price  # 보수적 청산
            reason = (
                f"EPS 성장률 {growth:.1%}, ROE 평균 {roe_avg:.1%}, "
                f"PER {pe_latest}, PBR {pbr_latest} → 일부 긍정적이나 확신 부족."
            )

        return {"buy_price": buy_price, "sell_price": sell_price, "opinion": opinion, "reason": reason}

    # -----------------------------
    # 메인 실행
    # -----------------------------
    def analyze(self, open_price: float, close_price: float) -> dict:
        """
        FMP와 Yahoo 데이터를 각각 분석하여 결과 이중 출력
        - 각 결과에는 raw 데이터도 함께 포함
        """
        # FMP 결과
        fmp_data = self.get_fmp_ratios()
        fmp_data_sorted = sorted(fmp_data, key=lambda x: x["year"])  # 과거→최신 정렬
        eps_fmp = [d["eps"] for d in fmp_data_sorted]
        roe_fmp = [d["roe"] for d in fmp_data_sorted]
        pe_fmp = next((d["pe"] for d in fmp_data_sorted if d["pe"] is not None), None)
        pbr_fmp = next((d["pbr"] for d in fmp_data_sorted if d["pbr"] is not None), None)
        fmp_result = self.judge(eps_fmp, roe_fmp, pe_fmp, pbr_fmp, open_price, close_price)
        fmp_result["raw"] = fmp_data_sorted

        # Yahoo 결과
        yahoo_data = self.get_yahoo_ratios()
        eps_yahoo = [d["eps"] for d in yahoo_data]
        roe_yahoo = [d["roe"] for d in yahoo_data if d["roe"] is not None]
        pe_yahoo = yahoo_data[-1].get("pe")
        pbr_yahoo = yahoo_data[-1].get("pbr")
        yahoo_result = self.judge(eps_yahoo, roe_yahoo, pe_yahoo, pbr_yahoo, open_price, close_price)
        yahoo_result["raw"] = yahoo_data

        # 최종 이중 출력
        return {"fmp": fmp_result, "yahoo": yahoo_result}


# -------------------------
# 사용 예시
# -------------------------
# agent = FundamentalAgent("AAPL", "YOUR_FMP_API_KEY", check_fmp_data=3)
# result = agent.analyze(open_price=150, close_price=155)
# print(result)
