import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from fun_sub import get_price_history, get_multi_index_prices, get_quarterly_report


class FundamentalAgent:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()

        # 모델, 스케일러, 피처 순서 로드
        self.model = joblib.load("models22/final_lgbm.pkl")
        self.scaler = joblib.load("models22/scaler.pkl")
        with open("models22/feature_cols.json", "r") as f:
            self.feature_order = json.load(f)

    def _build_features(self, target_date: str):
        ref_dt = datetime.strptime(target_date, "%Y-%m-%d")

        # 1. 개별 종목 주가
        df_prices = get_price_history(self.ticker, target_date)

        # 2. 시장 지표
        df_markets = get_multi_index_prices(target_date)

        # 3. 재무제표 (45일 지연 반영)
        df_fund = get_quarterly_report(self.ticker, target_date)
        if df_fund is not None:
            df_fund = pd.DataFrame([df_fund])
        else:
            df_fund = pd.DataFrame(columns=["symbol", "period", "year"])

        # 4. 주가 + 시장 + 재무제표 병합
        panel = df_prices.copy()
        panel.columns = [c.lower() for c in panel.columns]   # 소문자 변환
        panel.rename(columns={"date": "Date"}, inplace=True) # 날짜 복구
        panel = panel.merge(df_markets, on="Date", how="left")
        panel["symbol"] = self.ticker

        # 재무제표 asof merge
        if not df_fund.empty:
            df_fund["period"] = pd.to_datetime(df_fund["period"])
            merged = pd.merge_asof(
                panel.sort_values("Date"),
                df_fund.sort_values("period"),
                left_on="Date", right_on="period",
                by="symbol", direction="backward"
            )
        else:
            merged = panel.copy()

        # 파생변수 추가
        merged = merged.sort_values("Date")
        merged["ret_1"] = merged["close"].pct_change(1)
        for w in [5, 20, 60]:
            merged[f"ma_{w}"] = merged["close"].rolling(w, min_periods=3).mean()
        merged["vol_20"] = merged["ret_1"].rolling(20, min_periods=10).std()
        merged["mom_20"] = merged["close"] / merged["ma_20"]

        # === 추가: 펀더멘털 대비 시총 비율 ===
        if "net_income" in merged.columns and "market_cap" in merged.columns:
            merged["ni_to_mcap"] = merged["net_income"] / merged["market_cap"]
        else:
            merged["ni_to_mcap"] = 0.0

        if "revenue" in merged.columns and "market_cap" in merged.columns:
            merged["rev_to_mcap"] = merged["revenue"] / merged["market_cap"]
        else:
            merged["rev_to_mcap"] = 0.0

        if "operating_cashflow" in merged.columns and "market_cap" in merged.columns:
            merged["ocf_to_mcap"] = merged["operating_cashflow"] / merged["market_cap"]
        else:
            merged["ocf_to_mcap"] = 0.0

        # === 추가: 시장 상대 지표 ===
        if "NASDAQ_ret1" in merged.columns:
            merged["excess_ret"] = merged["ret_1"] - merged["NASDAQ_ret1"]
        else:
            merged["excess_ret"] = 0.0

        if "VIX" in merged.columns:
            merged["rel_vol"] = merged["vol_20"] / merged["VIX"]
        else:
            merged["rel_vol"] = 0.0



        # target_date 이전 가장 가까운 거래일
        row = merged.loc[merged["Date"] <= ref_dt].tail(1).copy()
        if row.empty:
            raise ValueError(f"{target_date} 데이터 없음")

        used_date = row["Date"].iloc[0]
        today_close = row["close"].iloc[0]

        # feature_cols.json 순서에 맞게 정렬
        feat_map = {}
        for col in self.feature_order:
            if col.startswith("sym_"):
                feat_map[col] = 1.0 if col == f"sym_{self.ticker}" else 0.0
            else:
                feat_map[col] = row.iloc[0].get(col, 0.0)

        X = pd.DataFrame([feat_map], columns=self.feature_order).fillna(0)
        return X, used_date, today_close

    def ml_predict_price(self, ref_date: str):
        # 피처 생성
        X, used_date, today_close = self._build_features(ref_date)

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 예측 (모델 출력은 다음날 수익률)
        pred_ret = float(self.model.predict(X_scaled)[0])

        # === 안전 가드: 비정상치 감지 ===
        if abs(pred_ret) > 0.5:  # ±50% 이상이면 경고
            print(f"[경고] 비정상적인 예측 수익률 감지: {pred_ret:.4f} (날짜={ref_date}, 종목={self.ticker})")

        # 내일 종가 복원
        pred_price = today_close * (1 + pred_ret)

        return {
            "ticker": self.ticker,
            "target_date": ref_date,      # 사용자가 요청한 날짜
            "used_date": str(used_date),  # 실제로 사용된 기준 거래일
            "today_close": float(today_close),
            "pred_ret": pred_ret,
            "pred_price": float(pred_price)
        }


if __name__ == "__main__":
    ticker = "MSFT"
    target_date = "2025-01-02"
    agent = FundamentalAgent(ticker)
    out = agent.ml_predict_price(target_date)
    print("\n=== 예측 결과 ===")
    print(out)
