#### 데이터 수집 및 csv 파일로 저장

import datetime
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import glob
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()  # .env 파일에서 환경변수 로드

api_key = os.getenv('FINN_API_KEY')  # FINN_API_KEY 환경변수 가져오기

# Nasdaq 100 기업 목록 (영문명)
nasdaq100_eng = {
    "NVIDIA": "NVDA", "Microsoft": "MSFT", "Apple": "AAPL", "Alphabet Class C": "GOOG",
    "Alphabet Class A": "GOOGL", "Amazon.com": "AMZN", "Meta Platforms": "META",
    "Broadcom": "AVGO", "Tesla": "TSLA", "Netflix": "NFLX", "Costco Wholesale": "COST",
    "Palantir Technologies Class A": "PLTR", "ASML Holding ADR": "ASML",
    "T-Mobile US": "TMUS", "Cisco Systems": "CSCO", "Advanced Micro Devices": "AMD",
    "AstraZeneca ADR": "AZN", "Linde": "LIN", "PepsiCo": "PEP", "AppLovin Class A": "APP",
    "Intuit": "INTU", "Shopify": "SHOP", "Booking Holdings": "BKNG",
    "Pinduoduo ADR": "PDD", "Qualcomm": "QCOM", "Texas Instruments": "TXN",
    "Intuitive Surgical": "ISRG", "Micron Technology": "MU", "Amgen": "AMGN",
    "Adobe": "ADBE", "Arm Holdings ADR": "ARM", "Gilead Sciences": "GILD",
    "Honeywell International": "HON", "Lam Research": "LRCX", "Palo Alto Networks": "PANW",
    "Applied Materials": "AMAT", "Comcast": "CMCSA", "Analog Devices": "ADI",
    "KLA": "KLAC", "Automatic Data Processing": "ADP", "MercadoLibre": "MELI",
    "Intel": "INTC", "Synopsys": "SNPS", "DoorDash": "DASH",
    "CrowdStrike Holdings": "CRWD", "Vertex Pharmaceuticals": "VRTX",
    "Cadence Design Systems": "CDNS", "Starbucks": "SBUX",
    "Constellation Energy": "CEG", "MicroStrategy": "MSTR",
    "O'Reilly Automotive": "ORLY", "Cintas": "CTAS", "Mondelez International": "MDLZ",
    "Thomson Reuters": "TRI", "Airbnb": "ABNB", "Marriott International": "MAR",
    "Autodesk": "ADSK", "PayPal Holdings": "PYPL", "Monster Beverage": "MNST",
    "Workday": "WDAY", "Fortinet": "FTNT", "CSX": "CSX",
    "Regeneron Pharmaceuticals": "REGN", "American Electric Power": "AEP",
    "Marvell Technology": "MRVL", "Axon Enterprise": "AXON",
    "NXP Semiconductors": "NXPI", "Roper Technologies": "ROP", "Fastenal": "FAST",
    "IDEXX Laboratories": "IDXX", "PACCAR": "PCAR", "Datadog": "DDOG",
    "Ross Stores": "ROST", "Paychex": "PAYX", "Atlassian": "TEAM", "Copart": "CPRT",
    "Take-Two Interactive": "TTWO", "Baker Hughes": "BKR", "Zscaler": "ZS",
    "Exelon": "EXC", "Xcel Energy": "XEL",
    "Coca-Cola Europacific Partners": "CCEP", "Electronic Arts": "EA",
    "Diamondback Energy": "FANG", "Verisk Analytics": "VRSK",
    "Keurig Dr Pepper": "KDP", "CoStar Group": "CSGP", "Charter Communications": "CHTR",
    "GE HealthCare Technologies": "GEHC", "Microchip Technology": "MCHP",
    "Cognizant Technology Solutions": "CTSH", "Kraft Heinz": "KHC",
    "Old Dominion Freight Line": "ODFL", "Dexcom": "DXCM",
    "Warner Bros. Discovery": "WBD", "Trade Desk": "TTD", "CDW": "CDW",
    "Biogen": "BIIB", "ON Semiconductor": "ON", "Lululemon Athletica": "LULU",
    "GlobalFoundries": "GFS"
}

class SentimentalSearcher:
    """
    Finnhub API와 yfinance 라이브러리를 사용하여 뉴스 및 주가 데이터를 수집합니다.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.nasdaq_100_tickers = list(nasdaq100_eng.values())
        self.today = datetime.date.today()
        self.one_year_ago = self.today - datetime.timedelta(days=365)
        self.today_str = self.today.strftime("%Y-%m-%d")
        self.one_year_ago_str = self.one_year_ago.strftime("%Y-%m-%d")

    def collect_news_data(self):
        """뉴스 데이터를 수집하고 CSV 파일로 저장합니다."""
        all_news_data = []
        print("\n--- 1. 뉴스 데이터 수집 ---")
        for ticker in self.nasdaq_100_tickers:
            print(f"{ticker} 기업 뉴스 데이터 수집 중...")
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={self.one_year_ago_str}&to={self.today_str}&token={self.api_key}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                news_data = response.json()
                if news_data:
                    for news in news_data:
                        news['symbol'] = ticker
                    all_news_data.extend(news_data)
                else:
                    print(f"  -> {ticker} 기업의 뉴스가 없습니다.")
            except requests.exceptions.RequestException as e:
                print(f"  -> HTTP 오류가 발생했습니다: {e}")
                continue
            time.sleep(0.5)

        news_df = pd.DataFrame(all_news_data)
        if not news_df.empty:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file_name = f"finnhub_nasdaq100_news_{timestamp}.csv"
            news_df.to_csv(csv_file_name, index=False, encoding='utf-8-sig')
            print(f"\n뉴스 데이터 수집 완료! 총 {len(news_df)}건을 {csv_file_name}에 저장했습니다.")
            return csv_file_name
        else:
            print("\n뉴스 데이터 수집 실패: 데이터가 없습니다.")
            return None

    def collect_stock_data(self):
        """주가 데이터를 수집하고 CSV 파일로 저장합니다."""
        all_stock_data = []
        print("\n--- 2. 주가 데이터 수집 ---")
        for ticker in self.nasdaq_100_tickers:
            print(f"{ticker} 주가 데이터 수집 중...")
            data = yf.download(ticker, start=self.one_year_ago, end=self.today, auto_adjust=True, progress=False)
            if data.empty:
                print(f"  -> 경고: {ticker}에 대한 데이터가 없습니다.")
                continue
            data = data.reset_index()
            data['symbol'] = ticker
            data = data.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            data['date'] = pd.to_datetime(data['date']).dt.date
            data = data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
            all_stock_data.append(data)

        if all_stock_data:
            stock_df = pd.concat(all_stock_data, ignore_index=True)
            csv_file_name = "nasdaq100_stock_data.csv"
            stock_df.to_csv(csv_file_name, index=False, encoding='utf-8-sig')
            print(f"\n주가 데이터 수집 완료! 총 {len(stock_df)}건을 {csv_file_name}에 저장했습니다.")
            return csv_file_name
        else:
            print("\n주가 데이터 수집 실패: 데이터가 없습니다.")
            return None

if __name__ == "__main__":
    if not api_key:
        print("오류: FINN_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    else:
        searcher = SentimentalSearcher(api_key)
        news_file = searcher.collect_news_data()
        stock_file = searcher.collect_stock_data()
        print("\n데이터 수집 완료!")
