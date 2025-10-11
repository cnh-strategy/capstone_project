import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class MacroSentimentAgent:
    def __init__(self, base_date: datetime, window: int = 40):
        """
        base_date: 예측 기준일
        window: 최근 데이터 일수
        """
        self.macro_tickers = {
            "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
            "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
            "^VIX": "^VIX",
            "DX-Y.NYB": "DX-Y.NYB",
            "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
            "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F",
            "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
        }
        self.data = None
        self.base_date = base_date
        self.start_date = base_date - timedelta(days=window + 20)  # 여유 20일 확보
        self.end_date = base_date

    # -------------------------------------------------------------
    # 1. 데이터 다운로드
    # -------------------------------------------------------------
    def fetch_data(self):
        """매크로 + 자산 데이터 다운로드"""
        df = yf.download(
            tickers=list(self.macro_tickers.values()),
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=False
        )

        # ✅ MultiIndex → 단일 인덱스로 변환
        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=0)
            df.index.names = ["Date", "Ticker"]
            df.sort_index(inplace=True)
            df.columns = [col for col in df.columns]
            df = df.unstack(level="Ticker")
            df.columns = ["_".join(col).strip() for col in df.columns.values]
        else:
            df.index.name = "Date"

        # ✅ 인덱스 복원 및 타입 보정
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])

        self.data = df
        print(f"[MacroSentimentAgent] Data shape: {df.shape}, Columns: {len(df.columns)}")
        return df

    # -------------------------------------------------------------
    # 2. 피처 생성
    # -------------------------------------------------------------
    def add_features(self):
        """수익률, 금리차, 위험심리 등 계산 + 주요 종목 피처 생성"""
        df = self.data.copy()
        df.set_index("Date", inplace=True)

        # -------------------------------
        # (1) 매크로 피처 생성
        # -------------------------------
        for ticker in self.macro_tickers.values():
            col_name = f"{ticker}_Close"
            if col_name in df.columns:
                df[f"{ticker}_ret_1d"] = df[col_name].pct_change()

        if "^TNX_Close" in df.columns and "^IRX_Close" in df.columns:
            df["Yield_spread"] = df["^TNX_Close"] - df["^IRX_Close"]

        if "SPY_ret_1d" in df.columns and "DX-Y.NYB_ret_1d" in df.columns and "^VIX_ret_1d" in df.columns:
            df["Risk_Sentiment"] = (
                    df["SPY_ret_1d"] - df["DX-Y.NYB_ret_1d"] - df["^VIX_ret_1d"]
            )

        # -------------------------------
        # (2) 추가: 주요 종목 데이터 (AAPL, MSFT, NVDA)
        # -------------------------------
        target_tickers = ["AAPL", "MSFT", "NVDA"]
        price_dfs = []
        for t in target_tickers:
            df_t = yf.download(t, start=self.start_date, end=self.end_date, interval="1d")[["Close"]]
            df_t.columns = [t]
            df_t[f"{t}_ret1"] = df_t[t].pct_change()
            df_t[f"{t}_ma5"] = df_t[t].rolling(5).mean()
            df_t[f"{t}_ma10"] = df_t[t].rolling(10).mean()
            price_dfs.append(df_t)

        price_df = pd.concat(price_dfs, axis=1).fillna(method="bfill")

        # -------------------------------
        # (3) 매크로 + 주가 데이터 병합
        # -------------------------------
        df.reset_index(inplace=True)
        price_df.reset_index(inplace=True)
        merged_df = pd.merge(price_df, df, on="Date", how="inner").fillna(0)

        # -------------------------------
        # (4) Feature 구조 정리
        # -------------------------------
        drop_cols = [c for c in merged_df.columns if c.endswith("_Close") or c in ["AAPL", "MSFT", "NVDA"]]
        merged_df = merged_df.drop(columns=drop_cols, errors="ignore")

        merged_df.columns = [c.replace("_ret_ret", "_ret") for c in merged_df.columns]
        merged_df.columns = [c.replace("__", "_") for c in merged_df.columns]

        # ✅ 학습 시 feature 순서와 동기화
        try:
            from joblib import load
            scaler_X = load("models/scaler_X.pkl")
            feature_order = list(scaler_X.feature_names_in_)
            merged_df = merged_df.reindex(columns=feature_order, fill_value=0)
        except Exception as e:
            print("[WARN] feature order sync skipped:", e)
            merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

        self.data = merged_df
        print(f"[MacroSentimentAgent] Feature engineering complete. Final shape: {merged_df.shape}")
        return merged_df

    # -------------------------------------------------------------
    # 3. 저장
    # -------------------------------------------------------------
    def save_csv(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "macro_sentiment.csv")
        self.data.to_csv(path, index=False)
        print(f"[MacroSentimentAgent] Saved {path}")
