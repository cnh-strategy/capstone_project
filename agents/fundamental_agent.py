# fundamental_agent.py (Yahoo Finance 기반, 파생피처 보강)

import os
import json
from datetime import date
from typing import Dict, Any, List

import numpy as np
import yfinance as yf
import pandas as pd
import joblib
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import lightgbm as lgb

from base_agent import BaseAgent

load_dotenv()


class FundamentalAgent(BaseAgent):
    """
    FundamentalAgent (Yahoo Finance 버전)
    - 학습 파이프라인(make_market.py, endPrice_market.py)과 동일하게
      Yahoo Finance 데이터 기반으로 펀더멘털 + 시장지표 피처 계산 후 예측
    """

    def __init__(
            self,
            ticker: str,
            check_years: int = 3,
            use_llm: bool = False,
            model_path: str = "final_lgbm.pkl",
            scaler_path: str = "scaler.pkl",
            feature_cols_path: str = "feature_cols.json",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.ticker = ticker.upper()
        self.check_years = check_years
        self.use_llm = use_llm

        # 모델/스케일러/피처목록 로드
        self.lgbm_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_cols_path, "r") as f:
            self.feature_order: List[str] = json.load(f)

        if isinstance(self.lgbm_model, lgb.LGBMRegressor):
            self.lgbm_booster = self.lgbm_model.booster_
        else:
            self.lgbm_booster = self.lgbm_model

        # 날짜 범위
        self.today = date.today()
        self.one_year_ago = self.today - relativedelta(years=1)
        self.one_year_ago_str = self.one_year_ago.strftime("%Y-%m-%d")

    # =========================
    # 시장 데이터 수집/파생
    # =========================
    def _download_index(self, ticker: str, start: str, end: str) -> pd.Series:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = "Close"
        return close

    def get_market_features(self, ref_date: str) -> Dict[str, float]:
        idx_map = {
            "NASDAQ": "^IXIC",
            "S&P500": "^GSPC",
            "DOWJONES": "^DJI",
            "DXY": "DX-Y.NYB",
            "VIX": "^VIX",
            "US10Y": "^TNX",
        }

        feats: Dict[str, float] = {}
        for name, idx_ticker in idx_map.items():
            close = self._download_index(idx_ticker, self.one_year_ago_str, ref_date)
            if close.empty:
                feats[name] = 0.0
                feats[f"{name}_lag1"] = 0.0
                feats[f"{name}_ret"] = 0.0
                feats[f"{name}_ma5"] = 0.0
                feats[f"{name}_ma20"] = 0.0
                feats[f"{name}_vol20"] = 0.0
                continue

            feats[name] = float(close.iloc[-1])
            feats[f"{name}_lag1"] = float(close.shift(1).iloc[-1]) if len(close) > 1 else feats[name]
            ret = close.pct_change()
            feats[f"{name}_ret"] = float(ret.iloc[-1]) if len(ret) > 1 else 0.0
            ma5 = close.rolling(5).mean()
            ma20 = close.rolling(20).mean()
            vol20 = close.pct_change().rolling(20).std()
            feats[f"{name}_ma5"] = float(ma5.iloc[-1]) if not np.isnan(ma5.iloc[-1]) else 0.0
            feats[f"{name}_ma20"] = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else 0.0
            feats[f"{name}_vol20"] = float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else 0.0

        return feats

    # =========================
    # Yahoo Finance 기반 펀더멘털
    # =========================
    def get_yahoo_fundamentals(self) -> Dict[str, float]:
        tkr = yf.Ticker(self.ticker)
        fin = tkr.financials
        bal = tkr.balance_sheet
        cf = tkr.cashflow
        info = tkr.info

        out: Dict[str, float] = {
            "revenue": 0.0, "net_income": 0.0, "operating_income": 0.0, "gross_profit": 0.0,
            "profit_margin": 0.0, "total_assets": 0.0, "total_liabilities": 0.0,
            "current_ratio": 0.0, "debt_to_equity": 0.0, "roe": 0.0,
            "operating_cashflow": 0.0, "capex": 0.0, "free_cashflow": 0.0,
            "eps": 0.0, "pe": 0.0, "forward_pe": 0.0, "pbr": 0.0, "market_cap": 0.0,
            "dividend_yield": 0.0, "beta": 0.0,
            "revenue_growth": 0.0, "net_income_growth": 0.0, "operating_income_growth": 0.0,
        }

        def safe_get(df, key):
            try:
                return float(df.loc[key].iloc[0])
            except Exception:
                return 0.0

        # 손익계산서
        if not fin.empty:
            out["revenue"] = safe_get(fin, "Total Revenue")
            out["net_income"] = safe_get(fin, "Net Income")
            out["operating_income"] = safe_get(fin, "Operating Income")
            out["gross_profit"] = safe_get(fin, "Gross Profit")

        # 재무상태표
        current_assets, current_liab, equity = 0.0, 0.0, 0.0
        if not bal.empty:
            out["total_assets"] = safe_get(bal, "Total Assets")
            out["total_liabilities"] = safe_get(bal, "Total Liabilities")
            current_assets = safe_get(bal, "Total Current Assets")
            current_liab = safe_get(bal, "Total Current Liabilities")
            equity = out["total_assets"] - out["total_liabilities"]
            out["current_ratio"] = current_assets / current_liab if current_liab else 0.0
            out["debt_to_equity"] = out["total_liabilities"] / equity if equity else 0.0
            out["roe"] = out["net_income"] / equity if equity else 0.0

        # 현금흐름표
        if not cf.empty:
            out["operating_cashflow"] = safe_get(cf, "Total Cash From Operating Activities")
            out["capex"] = abs(safe_get(cf, "Capital Expenditures"))

        # info 기반
        out["eps"] = float(info.get("trailingEps", 0.0) or 0.0)
        out["pe"] = float(info.get("trailingPE", 0.0) or 0.0)
        out["forward_pe"] = float(info.get("forwardPE", 0.0) or 0.0)
        out["pbr"] = float(info.get("priceToBook", 0.0) or 0.0)
        out["market_cap"] = float(info.get("marketCap", 0.0) or 0.0)
        out["dividend_yield"] = float(info.get("dividendYield", 0.0) or 0.0)
        out["beta"] = float(info.get("beta", 0.0) or 0.0)

        # 성장률 (최근 2개 분기 비교)
        if not fin.empty and fin.shape[1] > 1:
            try:
                rev_cur, rev_prev = fin.loc["Total Revenue"].iloc[0], fin.loc["Total Revenue"].iloc[1]
                out["revenue_growth"] = (rev_cur - rev_prev) / abs(rev_prev) if rev_prev else 0.0
            except Exception:
                pass
            try:
                ni_cur, ni_prev = fin.loc["Net Income"].iloc[0], fin.loc["Net Income"].iloc[1]
                out["net_income_growth"] = (ni_cur - ni_prev) / abs(ni_prev) if ni_prev else 0.0
            except Exception:
                pass
            try:
                op_cur, op_prev = fin.loc["Operating Income"].iloc[0], fin.loc["Operating Income"].iloc[1]
                out["operating_income_growth"] = (op_cur - op_prev) / abs(op_prev) if op_prev else 0.0
            except Exception:
                pass

        # profit margin
        if out["revenue"]:
            out["profit_margin"] = out["net_income"] / out["revenue"]

        # free cash flow
        out["free_cashflow"] = out["operating_cashflow"] - out["capex"]

        return out

    # =========================
    # 전체 피처 조립
    # =========================
    def build_all_features(self, fundamentals: Dict[str, float], market: Dict[str, float], symbol: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        f, m = fundamentals, market

        # 펀더멘털
        for k, v in f.items():
            out[k] = float(v) if np.isfinite(v) else 0.0

        # 시장 지표
        for k, v in m.items():
            out[k] = float(v) if np.isfinite(v) else 0.0

        # 교호항
        us10y = out.get("US10Y", 0.0)
        spx = out.get("S&P500", 0.0)
        vix = out.get("VIX", 1.0) or 1.0
        dxy = out.get("DXY", 1.0) or 1.0

        out["roe_scaled"] = (out["roe"]) / us10y if (out["roe"] and us10y) else 0.0
        out["pbr_scaled"] = out["pbr"] / spx if spx else 0.0
        out["pm_x_vix"] = out["profit_margin"] * vix
        out["mcap_div_vix"] = out["market_cap"] / vix if vix else 0.0
        out["ocf_div_us10y"] = out["operating_cashflow"] / us10y if us10y else 0.0

        # 환율(DXY) 기반 조정치
        out["revenue_fx_adj"] = out["revenue"] / dxy
        out["net_income_fx_adj"] = out["net_income"] / dxy
        out["marketcap_fx_adj"] = out["market_cap"] / dxy
        out["pe_fx_adj"] = out["pe"] / dxy
        out["pbr_fx_adj"] = out["pbr"] / dxy

        # 성장률 × DXY
        out["revenue_growth_x_dxy"] = out["revenue_growth"] * dxy
        out["net_income_growth_x_dxy"] = out["net_income_growth"] * dxy

        # 심볼 one-hot
        for col in self.feature_order:
            if col.startswith("sym_"):
                out[col] = 1.0 if col == f"sym_{symbol}" else 0.0

        return out

    def _to_dataframe_in_order(self, feat_map: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{c: feat_map.get(c, 0.0) for c in self.feature_order}])

    # =========================
    # 메인 예측
    # =========================
    def ml_predict_price(self, ref_date: str) -> Dict[str, float]:
        market = self.get_market_features(ref_date)
        fundamentals = self.get_yahoo_fundamentals()
        feat_map = self.build_all_features(fundamentals, market, self.ticker)

        X = self._to_dataframe_in_order(feat_map)
        X_scaled = self.scaler.transform(X)
        pred_price = float(self.lgbm_model.predict(X_scaled)[0])

        vix = feat_map.get("VIX", 0.0)
        confidence = float(np.clip(1.0 / (1.0 + max(vix, 0.0) / 20.0), 0.0, 1.0))
        return {"pred_price": pred_price, "confidence": confidence}


if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    ticker = input("조회할 Ticker를 입력하세요: ").strip()
    agent = FundamentalAgent(ticker, check_years=5, use_llm=False)
    out = agent.ml_predict_price(today_str)
    print("\n=== 예측 결과 ===")
    print(out)
