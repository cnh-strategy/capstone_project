import os
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
import lightgbm as lgb
from agents.fundamental_reviewer import FunReviewer
from base_agent import BaseAgent

# .env 로드
load_dotenv()


class FundamentalAgent(BaseAgent):
    """
    FundamentalAgent
    - FMP + Yahoo Finance 데이터 수집/정제/교차검증
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
        # Booster 직접 꺼내기
        if isinstance(self.lgbm_model, lgb.LGBMRegressor):
            self.lgbm_booster = self.lgbm_model.booster_
        else:
            self.lgbm_booster = self.lgbm_model
        print(type(self.lgbm_model))
        self.feature_order = self.lgbm_model.feature_name_

    # =========================================================
    # (A) 연도별 재무제표 수집 (FMP + Yahoo) - Reviewer용
    # =========================================================
    def get_fmp_ratios(self, years):
        try:
            url = f"""https://financialmodelingprep.com/api/v3/key-metrics/{self.ticker}?
                                        period=annual&limit={years}&apikey={self.fmp_api_key}"""
            datas = requests.get(url).json()
            if not isinstance(datas, list):
                return []
            return [
                {
                    "year": d.get("date"),
                    "pe": d.get("peRatio"),
                    "pbr": d.get("pbRatio"),
                    "roe": d.get("roe"),
                    "eps": d.get("eps"),
                    "reliability": "high"
                }
                for d in datas
            ]
        except Exception:
            return []

    def get_yahoo_ratios(self, years: int):
        try:
            stock = yf.Ticker(self.ticker)
            income_stmt = stock.income_stmt
            if income_stmt is None or income_stmt.empty:
                return []

            info = stock.info
            shares = info.get("sharesOutstanding")
            roe = info.get("returnOnEquity")
            pe = info.get("trailingPE")
            pbr = info.get("priceToBook")

            yahoo_result = []
            for year in income_stmt.columns[-years:]:
                net_income = income_stmt.loc["Net Income", year]
                eps = net_income / shares if shares else None
                yahoo_result.append({
                    "year": str(year),
                    "net_income": net_income,
                    "eps": eps,
                    "roe": roe,
                    "pe": pe,
                    "pbr": pbr,
                    "reliability": "high"
                })
            return yahoo_result
        except Exception:
            return []

    # 결측치 보완, 신뢰도(reliability) 추가
    def _clean_data(self, data_raw_list: list[dict], key_fields: list[str]):
        df = pd.DataFrame(data_raw_list)

        # None → np.nan 변환
        df = df.replace({None: np.nan})

        # 결측치 보간 처리 (앞뒤 값으로 메움)
        df = df.ffill().bfill()

        # reliability 기본값 설정
        if "reliability" not in df.columns:
            df["reliability"] = "high"
        else:
            df["reliability"] = df["reliability"].fillna("high")

        # 중요한 필드가 결측치라면 신뢰도를 낮게 평가
        for col in key_fields:
            if col in df.columns:
                df.loc[df[col].isna(), "reliability"] = "low"

        return df


    # FMP와 Yahoo의 데이터를 서로 비교해서 값이 크게 차이나는 경우 신뢰도를 낮춤
    def _cross_check(self, fmp_df, yahoo_df, key_fields: list[str]):

        for key in key_fields:
            if key in fmp_df.columns and key in yahoo_df.columns:
                # 숫자형으로 변환 (문자/None → NaN)
                fmp_series = pd.to_numeric(fmp_df[key], errors="coerce")
                yahoo_series = pd.to_numeric(yahoo_df[key], errors="coerce")

                # 차이율 계산 (NaN 자동 무시)
                diff = (fmp_series - yahoo_series).abs() / (fmp_series.abs() + 1e-6)

                mask_diff = diff > 0.05   # 차이 큰 값
                mask_yahoo_missing = yahoo_series.isna()  # Yahoo 값 없음
                mask_fmp_missing = fmp_series.isna()      # FMP 값 없음

                # (1) 차이가 큰 경우 → Yahoo 값으로 교체
                fmp_df.loc[mask_diff & ~mask_yahoo_missing, key] = yahoo_series[mask_diff & ~mask_yahoo_missing]

                # (2) Yahoo 값이 NaN인데 FMP 값이 존재 → Yahoo를 FMP 값으로 채움
                yahoo_df.loc[mask_yahoo_missing & ~mask_fmp_missing, key] = fmp_series[mask_yahoo_missing & ~mask_fmp_missing]

                # (3) FMP 값이 NaN인데 Yahoo 값이 존재 → FMP를 Yahoo 값으로 채움
                fmp_df.loc[mask_fmp_missing & ~mask_yahoo_missing, key] = yahoo_series[mask_fmp_missing & ~mask_yahoo_missing]

                # (4) 신뢰도 조정
                unreliable_mask = mask_diff | (mask_yahoo_missing & ~mask_fmp_missing) | (mask_fmp_missing & ~mask_yahoo_missing)
                fmp_df.loc[unreliable_mask, "reliability"] = "low"
                yahoo_df.loc[unreliable_mask, "reliability"] = "low"

        return fmp_df.to_dict(orient="records"), yahoo_df.to_dict(orient="records")


    # =========================================================
    # (B) LightGBM 기반 예측에 필요한 데이터 수집
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
            df = yf.download(ticker, start="2024-01-01", end=ref_date)
            if not df.empty:
                market[name] = df["Close"].iloc[-1]
                if len(df) > 1:
                    if name in ["NASDAQ", "S&P500", "DOWJONES", "DXY"]:
                        market[f"{name}_ret1"] = df["Close"].pct_change().iloc[-1]
                    elif name in ["US10Y", "VIX"]:
                        market[f"d{name}"] = df["Close"].diff().iloc[-1]
            else:
                market[name] = 0
        return market

    def get_technicals(self, symbol: str, ref_date: str):
        df = yf.download(symbol, start="2024-01-01", end=ref_date, group_by="column")

        if df.empty:
            return {"Close": 0, "ret_1": 0, "ma_5": 0, "ma_20": 0,
                    "ma_60": 0, "vol_20": 0, "mom_20": 0}

        # 혹시 멀티컬럼이면 첫 번째만 사용
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


    def get_latest_fundamentals(self):
        fundamentals = {
            "pe": 0, "pbr": 0, "roe": 0,
            "market_cap": 0, "debt_to_equity": 0, "current_ratio": 0,
            "operating_cashflow": 0, "free_cashflow": 0, "profit_margin": 0
        }

        # Yahoo Finance
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

        # FMP 보완
        try:
            url = f"""https://financialmodelingprep.com/api/v3/key-metrics
                                    /{self.ticker}?limit=1&apikey={self.fmp_api_key}"""
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

    def _prepare_features(self, fundamentals, market, technicals, symbol: str):
        feature_dict = {
            "Symbol": symbol,  # 문자열 그대로 넣음 → 나중에 category 변환
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

        # DataFrame으로 변환
        df = pd.DataFrame([feature_dict])

        # Symbol을 category 타입으로 변환 (LightGBM 학습 시와 동일하게 처리)
        df["Symbol"] = df["Symbol"].astype("category")

        # 나머지 모든 열은 float 변환 (에러 방지)
        for col in df.columns:
            if col != "Symbol":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # feature_order에 맞게 컬럼 순서 맞추기
        df = df[self.feature_order]

        return df



    def ml_predict_price(self, ref_date="2025-09-18"):
        market = self.get_market_data(ref_date)
        technicals = self.get_technicals(self.ticker, ref_date)
        fundamentals = self.get_latest_fundamentals()

        feature_vector = self._prepare_features(fundamentals, market, technicals, self.ticker)

        # 1. 기본 예측
        pred_price = float(self.lgbm_booster.predict(feature_vector)[0])

        # 2. Validation 성능 기반 confidence (예: 학습시 MAPE = 12%)
        val_mape = 0.12
        base_conf = max(0.0, min(1.0, 1 - val_mape))  # 0~1 사이 값 (0.88)

        # 3. 예측 분산 기반 confidence
        try:
            # 각 트리별 leaf output을 이용 (LightGBM Booster 활용)
            booster = self.lgbm_model.booster_
            tree_preds = []
            for tree_idx in range(booster.num_trees()):
                tree_output = booster.predict(feature_vector, start_iteration=tree_idx, num_iteration=tree_idx+1)
                tree_preds.append(tree_output[0])
            preds_array = np.array(tree_preds)

            mean_pred = preds_array.mean()
            std_pred = preds_array.std()

            # 상대적 분산 척도 (0~1 사이로 normalize)
            rel_uncertainty = std_pred / (abs(mean_pred) + 1e-6)
            variance_conf = max(0.0, min(1.0, 1 - rel_uncertainty))
        except Exception as e:
            print("분산 기반 confidence 계산 실패:", e)
            variance_conf = 0.8

        # 4. 최종 confidence (가중 평균)
        confidence = (base_conf * 0.6 + variance_conf * 0.4)

        return pred_price, confidence


    # =========================================================
    # (C) Main 실행
    # =========================================================
    def fundamental_main_analyze(self, ref_date="2025-09-18"):
        # 연도별 재무제표
        fmp_raw = sorted(self.get_fmp_ratios(self.check_years), key=lambda x: x["year"])
        yahoo_raw = sorted(self.get_yahoo_ratios(self.check_years), key=lambda x: x["year"])
        fmp_result, yahoo_result = {}, {}
        if fmp_raw and yahoo_raw:
            fmp_clean = self._clean_data(fmp_raw, ["eps", "roe", "pe", "pbr"])
            yahoo_clean = self._clean_data(yahoo_raw, ["eps", "roe", "pe", "pbr"])
            fmp_result, yahoo_result = self._cross_check(fmp_clean, yahoo_clean, ["eps", "roe", "pe", "pbr"])

        # 최신 예측
        pred_price, confidence = self.ml_predict_price(ref_date)

        return {
            "symbol": self.ticker,
            "fmp": fmp_result,
            "yahoo": yahoo_result,
            "pred_price": pred_price,
            "confidence": confidence
        }


# =========================================================
# 디버깅용
# =========================================================
if __name__ == "__main__":
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    ticker = input("조회할 Ticker를 입력하세요: ").strip()

    fund_agent = FundamentalAgent(ticker, FMP_API_KEY, check_years=5, use_llm=False)
    fund_result = fund_agent.fundamental_main_analyze()

    # reviewer = FunReviewer(use_llm=True)
    # round1_result = reviewer.review_round1(fund_result["fmp"], fund_result["yahoo"])

    print("\n=== FundamentalAgent 결과 ===")
    print(fund_result)

    # print("\n=== Reviewer Round1 결과 ===")
    # print(round1_result)



    # # 4) 다른 에이전트 실행 (현재 더미 형태일 수 있음)
    # s_agent = SentimentalAgent(model=None)
    # e_agent = EventAgent(model=None)
    # v_agent = ValuationAgent(model=None)
    #
    # s_result = s_agent.run(ticker)
    # e_result = e_agent.run(ticker)
    # v_result = v_agent.run(ticker)
    #
    # # 5) Reviewer Round2 실행 (모든 결과 종합)
    # agents_result = {
    #     "fundamental": round1_result,   # Round1 결과 반영
    #     "sentiment": s_result,
    #     "event": e_result,
    #     "valuation": v_result
    # }
    # round2_result = reviewer.review_round2(agents_result, prev_reviewer=round1_result)
    #
    # print("\n=== Reviewer Round2 최종 결과 ===")
    # print(round2_result)
