import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yfinance as yf
import pandas as pd

from config.agents import dir_info
from dateutil.relativedelta import relativedelta


save_dir = dir_info["data_dir"]
model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]
OUTPUT_DIR = data_dir


# 데이터셋과 모델 만드는 클래스
class MacroAData:
    def __init__(self,
                 ticker='NVDA'):
        self.merged_df = None
        self.macro_tickers = {
            "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
            "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
            "^VIX": "^VIX",
            "DX-Y.NYB": "DX-Y.NYB",
            "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
            "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F",
            #  "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
        }
        self.data = None

        self.agent_id = 'MacroAgent'
        self.ticker=ticker
        self.model_path = f"{model_dir}/{self.ticker}_{self.agent_id}.pt"
        self.scaler_X_path = f"{model_dir}/scalers/{self.ticker}_{self.agent_id}_xscaler.pkl"
        self.scaler_y_path = f"{model_dir}/scalers/{self.ticker}_{self.agent_id}_yscaler.pkl"

        five_years_ago = datetime.today() - relativedelta(years=5)
        self.start_date = five_years_ago.strftime("%Y-%m-%d")
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    def fetch_data(self):
        """다중 티커 데이터 다운로드"""

        df = yf.download(
            tickers=list(self.macro_tickers.values()),
            start = self.start_date,
            end= self.end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=False
        )

        # ✅ pandas 버전/구조 관계없이 일관된 포맷으로 변환
        # MultiIndex 구조일 경우 (티커별로 OHLCV 존재)
        if isinstance(df.columns, pd.MultiIndex):
            # 구조를 (날짜, 티커, 값) 형태로 변환
            df = df.stack(level=0)
            df.index.names = ["Date", "Ticker"]
            df.sort_index(inplace=True)

            # 컬럼 이름 평탄화
            df.columns = [col for col in df.columns]
            df = df.unstack(level="Ticker")
            df.columns = ["_".join(col).strip() for col in df.columns.values]
        else:
            # 단일 인덱스 구조인 경우 그대로 사용
            df.index.name = "Date"

        self.data = df
        print(f"[MacroAgent] Data shape: {df.shape}, Columns: {len(df.columns)}")
        return df

    def add_features(self):
        """수익률, 금리차, 위험심리 등 계산"""
        df = self.data.copy()

        # 각 자산의 1일 수익률
        for ticker in self.macro_tickers.values():
            if (ticker, "Close") in df.columns:
                df[(ticker, "ret_1d")] = df[(ticker, "Close")].pct_change()

        # 금리 스프레드 (10년 - 3개월)
        if ("^TNX", "Close") in df.columns and ("^IRX", "Close") in df.columns:
            df[("macro", "Yield_spread")] = df[("^TNX", "Close")] - df[("^IRX", "Close")]

        # 시장 위험심리 (SPY - DXY - VIX)
        if ("SPY", "ret_1d") in df.columns and ("DX-Y.NYB", "ret_1d") in df.columns and ("^VIX", "ret_1d") in df.columns:
            df[("macro", "Risk_Sentiment")] = (
                    df[("SPY", "ret_1d")] - df[("DX-Y.NYB", "ret_1d")] - df[("^VIX", "ret_1d")]
            )

        self.data = df
        print(f"[INFO] Feature engineering: {df.shape[0]} rows, {df.shape[1]} features")
        return df

    def save_csv(self):
        path = os.path.join(OUTPUT_DIR, f"{self.ticker}_{self.agent_id}.csv")
        self.data.to_csv(path, index=True)
        print(f"[MacroAgent] Saved {path}")


    def make_close_price(self):
        # 일별 종가 불러오기
        df_prices = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
        )["Close"]

        # CSV 저장
        path = os.path.join(OUTPUT_DIR, "daily_closePrice.csv")
        df_prices.to_csv(path)

        print("저장 완료:", df_prices.shape, "rows")


    #티커 통합 모델 저장
    def model_maker(self):

        # -------------------------------------------------------------
        # 1. 데이터 불러오기
        # -------------------------------------------------------------
        PRICE_CSV_PATH = os.path.join(OUTPUT_DIR, "daily_closePrice.csv")
        MACRO_CSV_PATH = os.path.join(OUTPUT_DIR, f"{self.ticker}_{self.agent_id}.csv")

        macro_df = pd.read_csv(MACRO_CSV_PATH)
        price_df = pd.read_csv(PRICE_CSV_PATH)

        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        price_df['Date'] = pd.to_datetime(price_df['Date'])

        # -------------------------------------------------------------
        # 2. 매크로 피처 확장 (원본 + 변화율)
        # -------------------------------------------------------------
        macro_features = [c for c in macro_df.columns if c != 'Date']
        macro_ret = macro_df[macro_features].pct_change()
        macro_ret.columns = [f"{c}_ret" for c in macro_ret.columns]
        macro_full = pd.concat([macro_df, macro_ret], axis=1)
        macro_full = macro_full.replace([np.inf, -np.inf], np.nan).dropna(subset=['Date'])
        macro_full = macro_full.ffill().bfill()

        # -------------------------------------------------------------
        # 3. Volume 계열 제거 + 상수 피처 제거
        # -------------------------------------------------------------
        remove_patterns = [
            "Volume_^FVX", "Volume_^IRX", "Volume_^TNX",
            "Volume_^VIX", "Volume_DX-Y.NYB",
            "Volume_EURUSD=X", "Volume_USDJPY=X"
        ]
        macro_full = macro_full.drop(
            columns=[c for c in macro_full.columns if any(p in c for p in remove_patterns)],
            errors='ignore'
        )

        constant_cols = [c for c in macro_full.columns if macro_full[c].std() == 0]
        macro_full = macro_full.drop(columns=constant_cols, errors="ignore")

        # -------------------------------------------------------------
        # 4. 주가 기반 피처 생성
        # -------------------------------------------------------------
        t = self.ticker
        price_df[f"{t}_ret1"]  = price_df[t].pct_change()
        price_df[f"{t}_ma5"]   = price_df[t].rolling(5).mean()
        price_df[f"{t}_ma10"]  = price_df[t].rolling(10).mean()
        price_df = price_df.fillna(method='bfill')

        # -------------------------------------------------------------
        # 5. 날짜 기준 병합
        # -------------------------------------------------------------
        merged = pd.merge(price_df, macro_full, on="Date", how="inner").sort_values("Date")
        self.merged_df = merged.reset_index(drop=True)

        # -------------------------------------------------------------
        # 6. Feature 컬럼 선택
        # -------------------------------------------------------------
        macro_cols  = [c for c in macro_full.columns if c != 'Date']
        price_cols  = [c for c in merged.columns if t in c and ('_ret' in c or '_ma' in c)]
        feature_cols = macro_cols + price_cols

        X_all = merged[feature_cols]

        # -------------------------------------------------------------
        # 7. 입력 스케일링 (scaler_X)
        # -------------------------------------------------------------
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_all)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        # -------------------------------------------------------------
        # 8. 타깃 생성
        # -------------------------------------------------------------
        merged[f"{t}_target"] = merged[t].pct_change().shift(-1)
        y_all = merged[[f"{t}_target"]].dropna().reset_index(drop=True)
        X_scaled = X_scaled.iloc[:len(y_all)]   # 길이 맞추기

        # -------------------------------------------------------------
        # 9. 출력 스케일링 (scaler_y)
        # -------------------------------------------------------------
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = scaler_y.fit_transform(y_all)

        # -------------------------------------------------------------
        # 10. 시퀀스 생성
        # -------------------------------------------------------------
        def create_sequences(X, y, window=40):
            Xs, ys = [], []
            for i in range(len(X) - window):
                Xs.append(X.iloc[i:(i + window)].values)
                ys.append(y[i + window])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=40)

        # -------------------------------------------------------------
        # 11. Train/Test split (여기까지만 하고 저장)
        # -------------------------------------------------------------
        split_idx = int(len(X_seq) * 0.8)
        self.X_train, self.X_test = X_seq[:split_idx], X_seq[split_idx:]
        self.y_train, self.y_test = y_seq[:split_idx], y_seq[split_idx:]

        # -------------------------------------------------------------
        # 12. 스케일러 저장
        # -------------------------------------------------------------
        os.makedirs(os.path.dirname(self.scaler_X_path), exist_ok=True)
        scaler_X.feature_names_in_ = np.array(feature_cols)

        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)

        # -------------------------------------------------------------
        # 13. 객체에 유지
        # -------------------------------------------------------------
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        print(f"[OK] MacroAData.model_maker: 데이터 + 시퀀스 + 스케일러 생성 완료")
