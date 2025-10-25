# agents/sentimental_agent.py
import pandas as pd
import numpy as np
import yfinance as yf
from agents.base_agent import BaseAgent, StockData, Target

class SentimentalAgent(BaseAgent):
    """여론/뉴스 감성 기반 에이전트"""

    def searcher(self, ticker: str) -> StockData:
        """간단히 뉴스 제목 긍/부정 샘플링 (임시 더미 버전)"""
        df = yf.download(ticker, period="1y", interval="1d")
        last_price = df["Close"].iloc[-1]

        # 가짜 여론 점수 (예시)
        sentiment_score = np.random.uniform(-1, 1)
        summary = {
            "sentiment_score": sentiment_score,
            "positivity": "positive" if sentiment_score > 0 else "negative",
            "sample_news": ["Market expects rebound", "Investors show optimism"],
        }

        self.stockdata = StockData(sentimental=summary, last_price=last_price)
        return self.stockdata

    def train(self, data: pd.DataFrame):
        """감성 점수 기반 단순 모델"""
        from sklearn.linear_model import LinearRegression
        X = data[["sentiment_score"]].fillna(0)
        y = data["next_close"]
        self.model = LinearRegression().fit(X, y)
        return True

    def predict(self, features: pd.DataFrame) -> Target:
        if self.model is None:
            raise ValueError("Model not trained.")
        pred = float(self.model.predict(features.fillna(0))[0])
        return Target(next_close=pred)
