import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

from config.agents import dir_info

'''
예측을 위해 최신 매크로 데이터 수집하는 클래스
몬테 카를로 생성 함수도 존재함
'''

# yfinance 진행률 바 및 경고 메시지 숨기기
warnings.filterwarnings("ignore")

model_dir: str = dir_info["model_dir"]

class MakeDatasetMacro:
    def __init__(self, base_date: datetime, window: int = 40, target_tickers=None):
        self.ticker = None
        self.macro_df = None
        self.macro_tickers = {
            "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
            "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
            "^VIX": "^VIX",
            "DX-Y.NYB": "DX-Y.NYB",
            "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
            "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F"
        }
        self.target_tickers = target_tickers #or ["AAPL", "MSFT", "NVDA"]
        self.base_date = base_date
        self.start_date = base_date - timedelta(days=window + 20)
        self.end_date = base_date
        self.agent_id='MacroAgent'
        self.data = None

    # -------------------------------------------------------------
    # 1. 데이터 수집
    # -------------------------------------------------------------
    def fetch_data(self):
        """매크로 자산 + 개별 티커 데이터 다운로드"""
        print(f"[INFO] Collecting macro features ({len(self.macro_tickers)} tickers)...")
        df_macro = yf.download(
            tickers=list(self.macro_tickers.values()),
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

        # ✅ MultiIndex → 단일 인덱스 변환
        if isinstance(df_macro.columns, pd.MultiIndex):
            df_macro = df_macro.stack(level=0)
            df_macro.index.names = ["Date", "Ticker"]
            df_macro = df_macro.unstack(level="Ticker")
            df_macro.columns = ["_".join(col).strip() for col in df_macro.columns.values]
        df_macro.reset_index(inplace=True)
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        print(f"[MacroAgent] Macro data: {df_macro.shape}")

        # ✅ 개별 주식 데이터 별도 수집
        price_dfs = []
        for t in self.target_tickers:
            self.ticker = t
            try:
                df_t = yf.download(t, start=self.start_date, end=self.end_date, interval="1d", progress=False)
                df_t = df_t.rename(columns={
                    "Open": f"Open_{t}",
                    "High": f"High_{t}",
                    "Low": f"Low_{t}",
                    "Close": f"Close_{t}",
                    "Volume": f"Volume_{t}"
                })
                df_t[f"{t}_ret1"] = df_t[f"Close_{t}"].pct_change()
                df_t[f"{t}_ma5"] = df_t[f"Close_{t}"].rolling(5).mean()
                df_t[f"{t}_ma10"] = df_t[f"Close_{t}"].rolling(10).mean()
                price_dfs.append(df_t)
            except Exception as e:
                print(f"[WARN] {t} 다운로드 실패:", e)

        # ✅ 주식 데이터 병합
        price_df = pd.concat(price_dfs, axis=1)

        # ✅ MultiIndex 평탄화
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = ["_".join([str(c) for c in col if c]) for col in price_df.columns]

        price_df.reset_index(inplace=True)
        price_df = price_df.loc[:, ~price_df.columns.duplicated()]

        print(f"[MacroAgent] Stock data: {price_df.shape}")

        # ✅ 매크로 + 주가 병합
        merged_df = pd.merge(df_macro, price_df, on="Date", how="inner").fillna(method="ffill")
        self.data = merged_df
        print(f"[MacroAgent] Data shape: {merged_df.shape}")
        return merged_df





    # -------------------------------------------------------------
    # 2. 피처 엔지니어링
    # -------------------------------------------------------------
    def add_features(self):
        df = self.data.copy()
        df.set_index("Date", inplace=True)

        # (1) 매크로 피처
        for ticker in self.macro_tickers.values():
            col_name = f"{ticker}_Close"
            if col_name in df.columns:
                df[f"{ticker}_ret_1d"] = df[col_name].pct_change()

        if "^TNX_Close" in df.columns and "^IRX_Close" in df.columns:
            df["Yield_spread"] = df["^TNX_Close"] - df["^IRX_Close"]

        if "SPY_ret_1d" in df.columns and "DX-Y.NYB_ret_1d" in df.columns and "^VIX_ret_1d" in df.columns:
            df["Risk_Sentiment"] = df["SPY_ret_1d"] - df["DX-Y.NYB_ret_1d"] - df["^VIX_ret_1d"]

        # (2) 결측치 처리
        df = df.fillna(method="ffill").fillna(method="bfill")

        # (3) 스케일러 순서 맞추기
        try:
            from joblib import load
            scaler_X = load(f"{model_dir}/scalers/{self.ticker}_{self.agent_id}_xscaler.pkl")
            feature_order = list(scaler_X.feature_names_in_)
            df = df.reindex(columns=feature_order, fill_value=0)
        except Exception as e:
            print("[WARN] feature order sync skipped:", e)
            df = df.reindex(sorted(df.columns), axis=1)

        df.reset_index(inplace=True)
        self.data = df
        print(f"[MacroAgent] Feature engineering complete. Final shape: {df.shape}")
        return self.data


