import os
from dotenv import load_dotenv

import requests
import yfinance as yf
import json
import pandas as pd
import numpy as np
from base_agent import BaseAgent  # BaseAgent 상속 (LLM 호출 포함)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# .env 파일 로드 (환경변수 로드)
load_dotenv()

class FundamentalAgent(BaseAgent):
    """
    FMP(Financial Modeling Prep)와 Yahoo Finance 데이터를 각각 활용하여
    매수/매도 가격과 최종 투자의견(Buy/Hold/Sell)을 산출하는 에이전트.

    - Input : ticker (예: 'AAPL', '005930.KQ')
    - Output: {
        "fmp": { buy_price, sell_price, opinion, reason, confidence, raw },
        "yahoo": { buy_price, sell_price, opinion, reason, confidence, raw }
      }
    ** raw: FMP 또는 Yahoo에서 직접 받아온 최근 3년치 재무 데이터 요약본
    """

    def __init__(self, ticker: str, fmp_api_key: str, check_years: int = 3, use_llm: bool = False, **kwargs):
        super().__init__(**kwargs)  # BaseAgent 초기화 (LLM 사용 가능)
        self.ticker = ticker
        self.fmp_api_key = fmp_api_key
        self.check_years = check_years
        self.use_llm = use_llm

    # -----------------------------
    # 데이터 수집
    # -----------------------------
    def get_fmp_ratios(self, years):
        try:
            fmp_url = (
                f"https://financialmodelingprep.com/api/v3/key-metrics/"
                f"{self.ticker}?period=annual&limit={years}&apikey={self.fmp_api_key}"
            )
            fmp_datas = requests.get(fmp_url).json()
            if not isinstance(fmp_datas, list):
                print(f"[get_fmp_ratios]Invalid response: {fmp_datas}")
                return []

            return [
                {
                    "year": fmp_data.get("date"),
                    "pe": fmp_data.get("peRatio"),
                    "pbr": fmp_data.get("pbRatio"),
                    "roe": fmp_data.get("roe"),
                    "eps": fmp_data.get("eps"),
                    "reliability": "high"
                }
                for fmp_data in fmp_datas
            ]

        except Exception as e:
            print(f"[get_fmp_ratios]Error: {e}")
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
        except Exception as e:
            print(f"[get_yahoo_ratios]Error: {e}")
            return []

    # -----------------------------
    # 데이터 정제 및 교차 검증
    # -----------------------------
    def _clean_data(self, data_list: list[dict], key_fields: list[str]) -> list[dict]:
        """결측치 보완 (forward/backward fill), reliability 태깅"""
        df = pd.DataFrame(data_list)
        df = df.fillna(method="ffill").fillna(method="bfill")

        # reliability 태깅
        df["reliability"] = df["reliability"].fillna("high")
        for col in key_fields:
            if col in df.columns:
                df.loc[df[col].isna(), "reliability"] = "low"

        return df.to_dict(orient="records")

    def _cross_check(self, fmp_data: list[dict], yahoo_data: list[dict], keys: list[str]) -> tuple[list[dict], list[dict]]:
        """Yahoo와 FMP 지표 교차 검증, ±5% 이상 차이 나는 값은 제외 및 신뢰도 낮음 태깅"""
        fmp_df, yahoo_df = pd.DataFrame(fmp_data), pd.DataFrame(yahoo_data)

        for key in keys:
            if key in fmp_df.columns and key in yahoo_df.columns:
                diff = abs(fmp_df[key] - yahoo_df[key]) / (fmp_df[key].abs() + 1e-6)
                mask = diff > 0.05
                fmp_df.loc[mask, key] = np.nan
                yahoo_df.loc[mask, key] = np.nan
                fmp_df.loc[mask, "reliability"] = "low"
                yahoo_df.loc[mask, "reliability"] = "low"

        return fmp_df.to_dict(orient="records"), yahoo_df.to_dict(orient="records")

    # -----------------------------
    # 보조 계산 (룰 + ML 기반)
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

    def _prepare_features(self, eps_values, roe_values, pe_latest, pbr_latest):
        growth = self.calculate_growth(eps_values)
        roe_avg = self.calculate_roe_avg(roe_values)
        pe = pe_latest if pe_latest else 0
        pbr = pbr_latest if pbr_latest else 0
        return np.array([[growth, roe_avg, pe, pbr]])

    def ml_predict(self, eps_values, roe_values, pe_latest, pbr_latest):
        """RandomForest + XGBoost 앙상블 예측"""
        X = self._prepare_features(eps_values, roe_values, pe_latest, pbr_latest)

        # 예시용 더미 학습 (실제는 외부 학습 필요)
        dummy_X = np.array([
            [0.10, 0.15, 12, 1.5],  # Buy
            [-0.02, 0.01, 35, 5],   # Sell
            [0.03, 0.08, 20, 3],    # Hold
        ])
        dummy_y = np.array([2, 0, 1])  # 0=Sell,1=Hold,2=Buy

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(dummy_X, dummy_y)
        xgb = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, use_label_encoder=False, eval_metric="mlogloss")
        xgb.fit(dummy_X, dummy_y)

        proba_rf = rf.predict_proba(X)
        proba_xgb = xgb.predict_proba(X)
        proba_avg = (proba_rf + proba_xgb) / 2

        pred_class = int(np.argmax(proba_avg))
        confidence = float(np.max(proba_avg))

        label_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        return label_map[pred_class], confidence

    def judge(self, eps_values, roe_values, pe_latest, pbr_latest, open_price, close_price):
        try:
            # Rule 기반
            growth = self.calculate_growth(eps_values)
            roe_avg = self.calculate_roe_avg(roe_values)

            if growth > 0.05 and roe_avg > 0.1 and pe_latest and pe_latest < 15 and pbr_latest and pbr_latest < 2:
                opinion = "Buy"
                buy_price = open_price
                sell_price = close_price * 1.05
                reason = f"[Rule] EPS {growth:.1%}, ROE {roe_avg:.1%}, PER {pe_latest:.1f}, PBR {pbr_latest:.1f} → 저평가 + 성장성 우수."
            elif (growth <= 0 or roe_avg <= 0.05 or (pe_latest and pe_latest >= 30) or (pbr_latest and pbr_latest >= 4)):
                opinion = "Sell"
                buy_price = open_price
                sell_price = close_price * 0.95
                reason = f"[Rule] EPS {growth:.1%}, ROE {roe_avg:.1%}, PER {pe_latest}, PBR {pbr_latest} → 성장/수익성 부족, 고평가 위험."
            else:
                opinion = "Hold"
                buy_price = open_price
                sell_price = close_price
                reason = f"[Rule] EPS {growth:.1%}, ROE {roe_avg:.1%}, PER {pe_latest}, PBR {pbr_latest} → 일부 긍정적이나 확신 부족."

            # ML 기반
            ml_opinion, confidence = self.ml_predict(eps_values, roe_values, pe_latest, pbr_latest)
            reason += f" [ML] 분류 결과={ml_opinion}, 신뢰도={confidence:.2f}"

            # 최종 의견 조율
            if opinion == ml_opinion:
                final_opinion = opinion
            else:
                final_opinion = "Hold"
                reason += " (Rule-ML 불일치 → Hold로 조정)"

            return {
                "buy_price": buy_price,
                "sell_price": sell_price,
                "opinion": final_opinion,
                "reason": reason,
                "confidence": confidence
            }
        except Exception as e:
            print(f"[judge]Error: {e}")
            return {"buy_price": None, "sell_price": None, "opinion": "Error", "reason": str(e), "confidence": 0.0}

    # -----------------------------
    # 메인 실행
    # -----------------------------
    def fundamental_main_analyze(self, open_price: float, close_price: float) -> dict:
        try:
            fmp_data = self.get_fmp_ratios(self.check_years)
            yahoo_data = self.get_yahoo_ratios(self.check_years)

            if not fmp_data or not yahoo_data:
                return {"fmp": {}, "yahoo": {}}

            fmp_data = self._clean_data(fmp_data, ["eps", "roe", "pe", "pbr"])
            yahoo_data = self._clean_data(yahoo_data, ["eps", "roe", "pe", "pbr"])
            fmp_data, yahoo_data = self._cross_check(fmp_data, yahoo_data, ["eps", "roe", "pe", "pbr"])

            if self.use_llm:
                currency, decimals = self._detect_currency_and_decimals(self.ticker)
                context = {"fmp": fmp_data, "yahoo": yahoo_data, "open_price": open_price, "close_price": close_price}

                msg_sys = {
                    "role": "system",
                    "content": (
                        "너는 전문 주식 애널리스트다. "
                        "주어진 데이터(FMP, Yahoo)와 현재 주가(open/close)를 기반으로 "
                        "목표 매수가/매도가를 산출한다. JSON만 반환."
                    )
                }
                msg_user = {
                    "role": "user",
                    "content": (
                        f"입력 데이터:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                        "요구사항:\n"
                        "1) buy_price(number)\n"
                        "2) sell_price(number, sell>=buy)\n"
                        "3) reason(string, 한국어 4~5문장)\n"
                        f"4) 통화 단위 {currency}, 소수점 {decimals}자리 반올림"
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
FMP_API_KEY = os.getenv("FMP_API_KEY")
agent1 = FundamentalAgent("AAPL", FMP_API_KEY, check_years=3, use_llm=False)
print(agent1.fundamental_main_analyze(open_price=150, close_price=155))

agent2 = FundamentalAgent("AAPL", FMP_API_KEY, check_years=3, use_llm=True)
print(agent2.fundamental_main_analyze(open_price=150, close_price=155))
