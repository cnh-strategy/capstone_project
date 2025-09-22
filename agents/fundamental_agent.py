import os
from datetime import date

import numpy as np
import requests
import yfinance as yf
import pandas as pd
import joblib
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import lightgbm as lgb
from base_agent import BaseAgent

# .env 로드
load_dotenv()


class FundamentalAgent(BaseAgent):
    """
    FundamentalAgent
    - LightGBM(pkl) 기반 종가 예측
    """

    def __init__(self, ticker: str, fmp_api_key: str, check_years: int = 3, use_llm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ticker = ticker.upper()
        self.fmp_api_key = fmp_api_key
        self.check_years = check_years
        self.use_llm = use_llm

        # LightGBM 모델 로드
        self.lgbm_model = joblib.load("lgbm_close_predictor.pkl")
        if isinstance(self.lgbm_model, lgb.LGBMRegressor):
            self.lgbm_booster = self.lgbm_model.booster_
        else:
            self.lgbm_booster = self.lgbm_model
        print(type(self.lgbm_model))
        self.feature_order = self.lgbm_model.feature_name_

        #날짜 정리
        # 오늘 날짜
        self.today = date.today()
        # 오늘 기준 1년 전
        self.one_year_ago = today - relativedelta(years=1)
        self.one_year_ago_str = self.one_year_ago.strftime("%Y-%m-%d")

        self.variance_history = []  # 최근 분산값들을 기록

    # =========================================================
    # (A) LightGBM 기반 예측에 필요한 데이터 수집
    # =========================================================
    def get_market_data(self, ref_date: str):
        tickers = {
            "NASDAQ": "^IXIC",
            "S&P500": "^GSPC",
            "DOWJONES": "^DJI",
            "DXY": "DX-Y.NYB",
            "VIX": "^VIX",
            "US10Y": "^TNX"
        }
        market = {}

        for name, ticker in tickers.items():
            df = yf.download(ticker, start= self.one_year_ago_str, end=ref_date)
            if not df.empty:
                market[name] = df["Close"].iloc[-1]
                if len(df) > 1:
                    if name in ["NASDAQ", "S&P500", "DOWJONES", "DXY"]:
                        market[f"{name}_ret1"] = df["Close"].pct_change().iloc[-1]
                    elif name in ["US10Y", "VIX"]:
                        market[f"d{name}"] = df["Close"].diff().iloc[-1]
            else:
                market[name] = 0
                print(f"[get_market_data] df:{df} = {market[name]}")
        return market


    def get_technicals(self, symbol: str, ref_date: str):
        df = yf.download(symbol, start=self.one_year_ago_str, end=ref_date, group_by="column")

        if df.empty:
            return {"Close": 0, "ret_1": 0, "ma_5": 0, "ma_20": 0,
                    "ma_60": 0, "vol_20": 0, "mom_20": 0}

        if isinstance(df["Close"], pd.DataFrame):
            close_series = df["Close"].iloc[:, 0]
        else:
            close_series = df["Close"]

        df["ret_1"] = close_series.pct_change()
        df["ma_5"] = close_series.rolling(5, min_periods=3).mean()
        df["ma_20"] = close_series.rolling(20, min_periods=10).mean()
        df["ma_60"] = close_series.rolling(60, min_periods=20).mean()
        df["vol_20"] = df["ret_1"].rolling(20, min_periods=10).std()
        df["mom_20"] = close_series / df["ma_20"]

        latest = df.iloc[-1]
        return {
            "Close": close_series.iloc[-1],
            "ret_1": latest["ret_1"],
            "ma_5": latest["ma_5"],
            "ma_20": latest["ma_20"],
            "ma_60": latest["ma_60"],
            "vol_20": latest["vol_20"],
            "mom_20": latest["mom_20"],
        }

    # [재무제표 가져오기] Yahoo에서 값이 없을 경우 FMP API로 보완
    def get_latest_fundamentals(self):
        fundamentals = {
            "pe": 0, "pbr": 0, "roe": 0,
            "market_cap": 0, "debt_to_equity": 0, "current_ratio": 0,
            "operating_cashflow": 0, "free_cashflow": 0, "profit_margin": 0
        }

        try:
            tkr = yf.Ticker(self.ticker)
            info = tkr.info
            fundamentals.update({
                "pe": info.get("trailingPE", 0),
                "pbr": info.get("priceToBook", 0),
                "roe": info.get("returnOnEquity", 0),
                "market_cap": info.get("marketCap", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "operating_cashflow": info.get("operatingCashflow", 0),
                "free_cashflow": info.get("freeCashflow", 0),
                "profit_margin": info.get("profitMargins", 0),
            })
        except Exception as e:
            print("Yahoo fundamentals 불러오기 실패:", e)

        try:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics/{self.ticker}?limit=1&apikey={self.fmp_api_key}"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    d = data[0]
                    fundamentals.update({
                        "pe": fundamentals["pe"] or d.get("peRatio", 0),
                        "pbr": fundamentals["pbr"] or d.get("pbRatio", 0),
                        "roe": fundamentals["roe"] or d.get("roe", 0),
                        "market_cap": fundamentals["market_cap"] or d.get("marketCap", 0),
                        "debt_to_equity": fundamentals["debt_to_equity"] or d.get("debtToEquity", 0),
                        "current_ratio": fundamentals["current_ratio"] or d.get("currentRatio", 0),
                        "operating_cashflow": fundamentals["operating_cashflow"] or d.get("operatingCashFlowPerShare", 0),
                        "free_cashflow": fundamentals["free_cashflow"] or d.get("freeCashFlowPerShare", 0),
                        "profit_margin": fundamentals["profit_margin"] or d.get("netProfitMargin", 0),
                    })
        except Exception as e:
            print("FMP fundamentals 불러오기 실패:", e)

        return fundamentals

    # 피쳐생성하기
    def _prepare_features(self, fundamentals, market, technicals, symbol: str):
        feature_dict = {
            "Symbol": symbol,
            "Close": technicals["Close"],
            "ret_1": technicals["ret_1"],
            "ma_5": technicals["ma_5"],
            "ma_20": technicals["ma_20"],
            "ma_60": technicals["ma_60"],
            "vol_20": technicals["vol_20"],
            "mom_20": technicals["mom_20"],

            "NASDAQ": market.get("NASDAQ", 0),
            "S&P500": market.get("S&P500", 0),
            "DOWJONES": market.get("DOWJONES", 0),
            "US10Y": market.get("US10Y", 0),
            "VIX": market.get("VIX", 0),
            "DXY": market.get("DXY", 0),

            "NASDAQ_ret1": market.get("NASDAQ_ret1", 0),
            "S&P500_ret1": market.get("S&P500_ret1", 0),
            "DOWJONES_ret1": market.get("DOWJONES_ret1", 0),
            "DXY_ret1": market.get("DXY_ret1", 0),
            "dUS10Y": market.get("dUS10Y", 0),
            "dVIX": market.get("dVIX", 0),

            "pe": fundamentals.get("pe", 0),
            "pbr": fundamentals.get("pbr", 0),
            "roe": fundamentals.get("roe", 0),
            "market_cap": fundamentals.get("market_cap", 0),
            "debt_to_equity": fundamentals.get("debt_to_equity", 0),
            "current_ratio": fundamentals.get("current_ratio", 0),
            "operating_cashflow": fundamentals.get("operating_cashflow", 0),
            "free_cashflow": fundamentals.get("free_cashflow", 0),
            "profit_margin": fundamentals.get("profit_margin", 0),
        }

        df = pd.DataFrame([feature_dict])
        df["Symbol"] = df["Symbol"].astype("category")

        for col in df.columns:
            if col != "Symbol":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        df = df[self.feature_order]
        return df

    def ml_predict_price(self, ref_date="2025-09-18"):
        market = self.get_market_data(ref_date)     #시장 데이터
        technicals = self.get_technicals(self.ticker, ref_date)   #테크니컬 데이터.. 이평선 등
        fundamentals = self.get_latest_fundamentals()   #재무제표 데이터

        # 피쳐 생성
        feature_vector = self._prepare_features(fundamentals, market, technicals, self.ticker)

        # LightGBM 예측 (각 트리별 출력 얻기)
        # 메인 예측값
        pred_price = float(self.lgbm_booster.predict(feature_vector)[0])

        # (2) 트리별 예측값 추출
        tree_preds = self.lgbm_booster.predict(
            feature_vector,
            pred_leaf=False,
            pred_contrib=False,
            raw_score=False,
            num_iteration=-1
        )
        var = np.var(tree_preds)

        # (3) 분산 기록 (히스토리 관리)
        self.variance_history.append(var)
        if len(self.variance_history) > 100:  # 최근 100개만 유지
            self.variance_history.pop(0)

        # (4) min-max scaling
        min_var = min(self.variance_history)
        max_var = max(self.variance_history)
        if max_var == min_var:  # 안정성 처리
            confidence = 0.5
        else:
            confidence = 1 - (var - min_var) / (max_var - min_var)
            confidence = float(np.clip(confidence, 0, 1))

        return {
                "pred_price": pred_price,
                "confidence": confidence
                }


if __name__ == "__main__":
    FMP_API_KEY = os.getenv("FMP_API_KEY")

    # 오늘 날짜 (YYYY-MM-DD 형태)
    today = date.today()           # date 객체
    today_str = today.strftime("%Y-%m-%d")  # 문자열로 변환

    ticker = input("조회할 Ticker를 입력하세요: ").strip()

    fund_agent = FundamentalAgent(ticker, FMP_API_KEY, check_years=5, use_llm=False)
    pred_price = fund_agent.ml_predict_price(today_str)

    print("\n=== 예측 종가 ===")
    print(pred_price)
