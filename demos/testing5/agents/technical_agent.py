# agents/technical_agent.py
import yfinance as yf
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent, StockData, Target

class TechnicalAgent(BaseAgent):
    """가격/거래량 기반 기술적 지표 에이전트"""

    def searcher(self, ticker: str) -> StockData:
        df = yf.download(ticker, period="1y", interval="1d")
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["RSI"] = self._rsi(df["Close"])
        df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()

        summary = {
            "last_close": df["Close"].iloc[-1],
            "ma20": df["MA20"].iloc[-1],
            "rsi": df["RSI"].iloc[-1],
            "macd": df["MACD"].iloc[-1],
            "trend": "UP" if df["Close"].iloc[-1] > df["MA20"].iloc[-1] else "DOWN",
        }

        self.stockdata = StockData(technical=summary, last_price=summary["last_close"])
        return self.stockdata

    def train(self, data: pd.DataFrame):
        """단순 기술 지표 회귀 모델"""
        from sklearn.linear_model import LinearRegression
        X = data[["ma20", "rsi", "macd"]].fillna(0)
        y = data["next_close"]
        self.model = LinearRegression().fit(X, y)
        return True

    def predict(self, features: pd.DataFrame) -> Target:
        if self.model is None:
            raise ValueError("Model not trained.")
        pred = float(self.model.predict(features.fillna(0))[0])
        return Target(next_close=pred)

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14):
        delta = series.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.rolling(period).mean() / down.rolling(period).mean()
        return 100 - (100 / (1 + rs))
