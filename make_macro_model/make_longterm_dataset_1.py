import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 공통 설정
# ============================================================
OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2020-01-01"
END_DATE = '2024-12-31'


# ============================================================
# MacroSentimentAgent — 시장·거시경제 시계열 기반
# ============================================================
class MacroSentimentAgentDataset:
    def __init__(self):
        self.macro_tickers = {
            "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
            "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
            "^VIX": "^VIX",
            "DX-Y.NYB": "DX-Y.NYB",
            "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
            "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F",
          #  "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
        }
        self.data = None

    def fetch_data(self):
        """다중 티커 데이터 다운로드"""
        df = yf.download(
            tickers=list(self.macro_tickers.values()),
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            group_by="ticker",
            auto_adjust=False
        )

        # ✅ pandas 버전/구조 관계없이 일관된 포맷으로 변환
        # MultiIndex 구조일 경우 (티커별로 OHLCV 존재)
        if isinstance(df.columns, pd.MultiIndex):
            # 구조를 (날짜, 티커, 값) 형태로 변환
            df = df.stack(level=0)
            df.index.names = ["Date", "Ticker"]
            df.sort_index(inplace=True)

            # 컬럼 이름 평탄화
            df.columns = [col for col in df.columns]
            df = df.unstack(level="Ticker")
            df.columns = ["_".join(col).strip() for col in df.columns.values]
        else:
            # 단일 인덱스 구조인 경우 그대로 사용
            df.index.name = "Date"

        self.data = df
        print(f"[MacroSentimentAgent] Data shape: {df.shape}, Columns: {len(df.columns)}")
        return df

    def add_features(self):
        """수익률, 금리차, 위험심리 등 계산"""
        df = self.data.copy()

        # 각 자산의 1일 수익률
        for ticker in self.macro_tickers.values():
            if (ticker, "Close") in df.columns:
                df[(ticker, "ret_1d")] = df[(ticker, "Close")].pct_change()

        # 금리 스프레드 (10년 - 3개월)
        if ("^TNX", "Close") in df.columns and ("^IRX", "Close") in df.columns:
            df[("macro", "Yield_spread")] = df[("^TNX", "Close")] - df[("^IRX", "Close")]

        # 시장 위험심리 (SPY - DXY - VIX)
        if ("SPY", "ret_1d") in df.columns and ("DX-Y.NYB", "ret_1d") in df.columns and ("^VIX", "ret_1d") in df.columns:
            df[("macro", "Risk_Sentiment")] = (
                    df[("SPY", "ret_1d")] - df[("DX-Y.NYB", "ret_1d")] - df[("^VIX", "ret_1d")]
            )

        self.data = df
        return df

    def save_csv(self):
        path = os.path.join(OUTPUT_DIR, "macro_data/macro_sentiment.csv")
        self.data.to_csv(path, index=True)
        print(f"[MacroSentimentAgent] Saved {path}")


    def close_price_fetch(self, ticker_name):
        # 여러 종목의 일별 종가 불러오기 (2020-01-01 ~ 2024-12-31)
        df_prices = yf.download(
            ticker_name,
            start="2020-01-01",
            end="2025-01-03"
        )["Close"]

        # CSV 저장
        df_prices.to_csv(f"data/macro_data/daily_closePrice_{ticker_name}.csv")

        print("저장 완료:", df_prices.shape, "rows")

# ============================================================
# 실행 예시
# ============================================================
if __name__ == "__main__":
    # ② MacroSentimentAgent
    macro_agent = MacroSentimentAgentDataset()
    macro_agent.fetch_data()
    macro_agent.close_price_fetch('NVDA')
    macro_agent.add_features()
    macro_agent.save_csv()