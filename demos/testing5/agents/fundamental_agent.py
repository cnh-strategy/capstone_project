# agents/fundamental_agent.py
import yfinance as yf
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent, StockData, Target

class FundamentalAgent(BaseAgent):
    """기업 재무 기반 예측 에이전트"""

    def searcher(self, ticker: str) -> StockData:
        """재무제표 및 밸류 지표 수집"""
        tkr = yf.Ticker(ticker)
        info = tkr.info
        df = tkr.financials.T

        # 단순 요약
        summary = {
            "revenue": df["Total Revenue"].iloc[-1] if "Total Revenue" in df.columns else np.nan,
            "net_income": df["Net Income"].iloc[-1] if "Net Income" in df.columns else np.nan,
            "debt_to_equity": info.get("debtToEquity"),
            "pe_ratio": info.get("trailingPE"),
            "price_to_book": info.get("priceToBook"),
            "currency": info.get("currency", "USD"),
        }

        self.stockdata = StockData(fundamental=summary, currency=summary["currency"])
        return self.stockdata

    def train(self, data: pd.DataFrame):
        """회사의 재무 비율 기반 간단한 회귀 모델 학습 (임시 더미 버전)"""
        from sklearn.linear_model import LinearRegression

        X = data[["revenue", "net_income", "debt_to_equity", "pe_ratio"]].fillna(0)
        y = data["next_close"]

        self.model = LinearRegression().fit(X, y)
        return True

    def predict(self, features: pd.DataFrame) -> Target:
        """학습된 모델 기반 다음 종가 예측"""
        if self.model is None:
            raise ValueError("Model not trained.")
        pred = float(self.model.predict(features.fillna(0))[0])
        return Target(next_close=pred)
