import os
import requests
import yfinance as yf
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from agents.fundamental_reviewer import FunReviewer
from base_agent import BaseAgent
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# .env 로드
load_dotenv()

class FundamentalAgent(BaseAgent):
    """
    FundamentalAgent
    - Searcher: FMP, Yahoo Finance 데이터 수집/정제/교차검증
    - Predictor: Rule 기반 + ML(RandomForest, XGBoost) 기반 종가 예측
    """

    def __init__(self, ticker: str, fmp_api_key: str, check_years: int = 3, use_llm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ticker = ticker
        self.fmp_api_key = fmp_api_key
        self.check_years = check_years
        self.use_llm = use_llm

    # -----------------------------
    # 데이터 수집
    # -----------------------------
    def get_fmp_ratios(self, years):
        try:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics/{self.ticker}?period=annual&limit={years}&apikey={self.fmp_api_key}"
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

    # -----------------------------
    # 데이터 정제/교차검증
    # -----------------------------
    #결측치 보완, 신뢰도(reliability) 추가
    def _clean_data(self, data_raw_list: list[dict], key_fields: list[str]):
        data_list = pd.DataFrame(data_raw_list)
        #결측치 보간 처리: 앞뒤 값을 이용해서 빈 칸을 메우기
        df_check = data_list.fillna(method="ffill").fillna(method="bfill")
        #reliability 기본값 설정
        df_check["reliability"] = df_check["reliability"].fillna("high")
        #중요한 필드가 결측치라면 신뢰도를 낮게 평가
        for col in key_fields:
            if col in df_check.columns:
                df_check.loc[df_check[col].isna(), "reliability"] = "low"
        return df_check

    # FMP와 Yahoo의 데이터를 서로 비교해서 값이 크게 차이나는 경우 신뢰도를 낮춤
    def _cross_check(self, fmp_df, yahoo_df, key_fields: list[str]):

        for key in key_fields:
            if key in fmp_df.columns and key in yahoo_df.columns:
                # 차이율 계산 (NaN 무시)
                diff = abs(fmp_df[key] - yahoo_df[key]) / (fmp_df[key].abs() + 1e-6)
                mask_diff = diff > 0.05   # 차이 큰 값
                mask_yahoo_missing = yahoo_df[key].isna()  # Yahoo 값 없음
                mask_fmp_missing = fmp_df[key].isna()      # FMP 값 없음

                # (1) 차이가 큰 경우 → Yahoo 값으로 교체
                fmp_df.loc[mask_diff & ~mask_yahoo_missing, key] = yahoo_df.loc[mask_diff & ~mask_yahoo_missing, key]

                # (2) Yahoo 값이 NaN인데 FMP 값이 존재 → Yahoo를 FMP 값으로 채움
                yahoo_df.loc[mask_yahoo_missing & ~mask_fmp_missing, key] = fmp_df.loc[mask_yahoo_missing & ~mask_fmp_missing, key]

                # (3) FMP 값이 NaN인데 Yahoo 값이 존재 → FMP를 Yahoo 값으로 채움
                fmp_df.loc[mask_fmp_missing & ~mask_yahoo_missing, key] = yahoo_df.loc[mask_fmp_missing & ~mask_yahoo_missing, key]

                # (4) 신뢰도 조정
                #   - 차이가 큰 경우
                #   - Yahoo 값이 없어 FMP로 보완된 경우
                #   - FMP 값이 없어 Yahoo로 보완된 경우
                unreliable_mask = mask_diff | (mask_yahoo_missing & ~mask_fmp_missing) | (mask_fmp_missing & ~mask_yahoo_missing)
                fmp_df.loc[unreliable_mask, "reliability"] = "low"
                yahoo_df.loc[unreliable_mask, "reliability"] = "low"

        return fmp_df.to_dict(orient="records"), yahoo_df.to_dict(orient="records")




    # -----------------------------
    # Predictor (종가 예측)
    # -----------------------------
    # EPS 성장률 계산
    def calculate_growth(self, eps_values: list) -> float:
        eps_values = [v for v in eps_values if v is not None]

        # 첫 해 EPS와 마지막 해 EPS가 모두 양수이고, 2년 이상 데이터가 있는 경우 → CAGR(연평균 성장률) 계산
        if len(eps_values) >= 2 and eps_values[0] > 0 and eps_values[-1] > 0:
            years = len(eps_values) - 1
            return (eps_values[-1] / eps_values[0]) ** (1 / years) - 1
        return 0

    # ROE 평균 수익성 계산
    def calculate_roe_avg(self, roe_values: list) -> float:
        roe_values = [roe_v for roe_v in roe_values if roe_v is not None]
        # 값이 있으면 단순 평균 반환, 없으면 0 반환
        return sum(roe_values) / len(roe_values) if roe_values else 0

    # ML 모델 입력용 feature 벡터 생성
    def _prepare_features(self, eps_values, roe_values, pe_latest, pbr_latest):
        '''
        최종적 모델 입력값(return): EPS 성장률, ROE 평균, 최신 PER, 최신 PBR
        '''
        # 성장률과 수익성 계산
        growth = self.calculate_growth(eps_values)
        roe_avg = self.calculate_roe_avg(roe_values)

        pe = pe_latest if pe_latest else 0
        pbr = pbr_latest if pbr_latest else 0

        return np.array([[growth, roe_avg, pe, pbr]])

    # ML 기반 주가 예측 모듈 (수정 및 고도화 필요!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!>>> pkl 파일로 만들어서 대입해 두기!)
    def ml_predict_price(self, eps_values, roe_values, pe_latest, pbr_latest):
        target_vector = self._prepare_features(eps_values, roe_values, pe_latest, pbr_latest)

        # 2. 더미 학습 데이터 (feature 4개: 성장률, ROE 평균, PER, PBR)
        dummy_X = np.array([
            [0.10, 0.15, 12, 1.5],  # 강한 Buy 시나리오
            [-0.02, 0.01, 35, 5],   # 강한 Sell 시나리오
            [0.03, 0.08, 20, 3],    # 중립 Hold 시나리오
        ])
        dummy_y = np.array([120, 80, 100])  # 각 시나리오에 대응하는 예측 종가 (예시)

        # 3. 랜덤포레스트 모델 학습
        rf = RandomForestRegressor(n_estimators=20, random_state=42)
        rf.fit(dummy_X, dummy_y)

        # 4. XGBoost 모델 학습
        xgb = XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
        xgb.fit(dummy_X, dummy_y)

        # 5. 두 모델 예측 수행
        pred_rf = rf.predict(target_vector)
        pred_xgb = xgb.predict(target_vector)

        # 6. 예측값 평균 → 최종 예측 종가 산출
        pred_price = float((pred_rf + pred_xgb) / 2)

        # 7. 예측 결과 반환 (confidence는 임시 고정값 0.8)
        return pred_price, 0.8

    # 재무지표와 ML 예측을 종합해서 매수/매도 의견을 내는 규칙 기반 평가 함수
    def judge(self, eps_values, roe_values, pe_latest, pbr_latest):
        pred_price, confidence = self.ml_predict_price(eps_values, roe_values, pe_latest, pbr_latest)
        growth = self.calculate_growth(eps_values)
        roe_avg = self.calculate_roe_avg(roe_values)

        if growth > 0.05 and roe_avg > 0.1 and pe_latest and pe_latest < 15 and pbr_latest and pbr_latest < 2:
            opinion = "Buy"
        elif (growth <= 0 or roe_avg <= 0.05 or (pe_latest and pe_latest >= 30) or (pbr_latest and pbr_latest >= 4)):
            opinion = "Sell"
        else:
            opinion = "Hold"

        return {
            "predicted_price": pred_price,
            "opinion": opinion,
            "reason": f"[Rule] EPS 성장 {growth:.1%}, ROE 평균 {roe_avg:.1%}, PER {pe_latest}, PBR {pbr_latest}",
            "confidence": confidence
        }

    # -----------------------------
    # Main 실행
    # -----------------------------
    def fundamental_main_analyze(self) -> dict:
        # 1. API 활용한 데이터 수집 + 연도 정렬
        fmp_raw_data = sorted(self.get_fmp_ratios(self.check_years), key=lambda x: x["year"])
        yahoo_raw_data = sorted(self.get_yahoo_ratios(self.check_years), key=lambda x: x["year"])
        if not fmp_raw_data or not yahoo_raw_data:
            return {"fmp": {}, "yahoo": {}}

        # 2.데이터 정제
        # 결측치 보완, 신뢰도(reliability) 추가
        fmp_clean_data = self._clean_data(fmp_raw_data, ["eps", "roe", "pe", "pbr"])
        yahoo_clean_data = self._clean_data(yahoo_raw_data, ["eps", "roe", "pe", "pbr"])
        # 데이터 교차 검증
        fmp_data, yahoo_data = self._cross_check(fmp_clean_data, yahoo_clean_data, ["eps", "roe", "pe", "pbr"])

        # 4. 최종
        fmp_sorted = sorted(fmp_data, key=lambda x: x["year"])  # 출력 보장 차원에서 다시 확인용 정렬
        eps_fmp = [data["eps"] for data in fmp_sorted]
        roe_fmp = [data["roe"] for data in fmp_sorted]
        pe_fmp = fmp_sorted[-1].get("pe")
        pbr_fmp = fmp_sorted[-1].get("pbr")
        fmp_result = self.judge(eps_fmp, roe_fmp, pe_fmp, pbr_fmp)
        fmp_result["raw"] = fmp_sorted

        yahoo_sorted = sorted(yahoo_data, key=lambda x: x["year"])
        eps_yahoo = [data["eps"] for data in yahoo_sorted]
        roe_yahoo = [data["roe"] for data in yahoo_sorted if data["roe"] is not None]
        pe_yahoo = yahoo_sorted[-1].get("pe")
        pbr_yahoo = yahoo_sorted[-1].get("pbr")
        yahoo_result = self.judge(eps_yahoo, roe_yahoo, pe_yahoo, pbr_yahoo)
        yahoo_result["raw"] = yahoo_sorted

        return {"fmp": fmp_result, "yahoo": yahoo_result}


# 디버깅용
if __name__ == "__main__":
    # 1) 입력 받기
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    ticker = input("조회할 Ticker를 입력하세요: ").strip()

    # 2) Fundamental 실행
    fund_agent = FundamentalAgent(ticker, FMP_API_KEY, check_years=5, use_llm=False)
    fund_result = fund_agent.fundamental_main_analyze()

    # 3) Reviewer Round1 실행
    reviewer = FunReviewer(use_llm=True)
    round1_result = reviewer.review_round1(fund_result["fmp"], fund_result["yahoo"])
    print("\n=== Reviewer Round1 결과 ===")
    print(round1_result)

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
