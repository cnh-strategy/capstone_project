import os
import joblib
import numpy as np
from keras import Input
from keras.src.saving.saving_api import save_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import pandas as pd

from config.agents import dir_info
from core.macro_classes.nasdaq_100 import nasdaq100_eng

# 종목 리스트 (딕셔너리 값 = 티커)
symbols = list(nasdaq100_eng.values())


save_dir = dir_info["data_dir"]
model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]
OUTPUT_DIR = data_dir

# 데이터셋과 모델 만드는 클래스
class MacroAData:
    """\
    macro_agent = MacroAData()
    macro_agent.fetch_data()
    macro_agent.add_features()
    macro_agent.save_csv()
    macro_agent.make_close_price()

    macro_agent.model_maker()
    """

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

        self.agent_id = 'MacroSentiAgent'
        self.ticker=ticker
        self.model_path = f"{model_dir}/{self.ticker}_{self.agent_id}.keras"
        self.scaler_X_path = f"{model_dir}/scalers/{self.ticker}_{self.agent_id}_xscaler.pkl"
        self.scaler_y_path = f"{model_dir}/scalers/{self.ticker}_{self.agent_id}_yscaler.pkl"

    def fetch_data(self):
        """다중 티커 데이터 다운로드"""
        df = yf.download(
            tickers=list(self.macro_tickers.values()),
            start="2020-01-01",
            end='2024-12-31',
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
        print(f"[MacroSentimentAgent] Data shape: {df.shape}, Columns: {len(df.columns)}")
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
        print(f"[TRACE A] add_features() for self.data:{self.data}")
        print(f"[TRACE A] add_features() for {self.ticker} columns:", df.columns.tolist()[:15])
        return df

    def save_csv(self):
        path = os.path.join(OUTPUT_DIR, f"{self.ticker}_{self.agent_id}.csv")
        self.data.to_csv(path, index=True)
        print(f"[MacroSentimentAgent] Saved {path}")


    def make_close_price(self):
        # 여러 종목의 일별 종가 불러오기 (2020-01-01 ~ 2024-12-31)
        df_prices = yf.download(
            symbols,
            start="2020-01-01",
            end="2025-01-03"
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
        # macro_full = macro_full.replace([np.inf, -np.inf], np.nan).dropna(subset=['Date']).fillna(0)
        macro_full = macro_full.replace([np.inf, -np.inf], np.nan).dropna(subset=['Date'])
        macro_full = macro_full.ffill().bfill()


        # ✅ 거래량 없는 피처 제거 (원본 + 변화율 포함)
        remove_patterns = [
            "Volume_^FVX", "Volume_^IRX", "Volume_^TNX",
            "Volume_^VIX", "Volume_DX-Y.NYB",
            "Volume_EURUSD=X", "Volume_USDJPY=X"
        ]

        macro_full = macro_full.drop(
            columns=[c for c in macro_full.columns if any(p in c for p in remove_patterns)],
            errors='ignore'
        )
        print(f"[INFO] Removed all constant Volume columns matching patterns: {remove_patterns}")
        print(f"[INFO] Remaining Volume columns: {[c for c in macro_full.columns if 'Volume' in c]}")

        # ✅ 분산이 0인 상수 피처 자동 제거
        constant_cols = [c for c in macro_full.columns if macro_full[c].std() == 0]
        if constant_cols:
            print(f"[INFO] Removing {len(constant_cols)} constant columns: {constant_cols}")
            macro_full = macro_full.drop(columns=constant_cols)


        # -------------------------------------------------------------
        # 3. 주가 기반 피처 생성 (각 종목별)
        # -------------------------------------------------------------
        # target_tickers = ['AAPL', 'MSFT', 'NVDA']
        for ticer in [self.ticker]:
            price_df[f"{ticer}_ret1"] = price_df[ticer].pct_change()
            price_df[f"{ticer}_ma5"] = price_df[ticer].rolling(5).mean()
            price_df[f"{ticer}_ma10"] = price_df[ticer].rolling(10).mean()
        price_df = price_df.fillna(method='bfill')

        # -------------------------------------------------------------
        # 4. 날짜 기준 병합
        # -------------------------------------------------------------
        self.merged_df = pd.merge(price_df, macro_full, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
        print(f"[INFO] 병합 후 데이터 shape: {self.merged_df.shape}")

        # -------------------------------------------------------------
        # 5. Feature 선택
        # -------------------------------------------------------------
        macro_cols = [c for c in macro_full.columns if c != 'Date']
        price_cols = [c for c in self.merged_df.columns if any(t in c for t in [self.ticker]) and ('_ret' in c or '_ma' in c)]
        feature_cols = macro_cols + price_cols

        X_all = self.merged_df[feature_cols]

        # -------------------------------------------------------------
        # 6. 입력 스케일링
        # -------------------------------------------------------------
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_all)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        # -------------------------------------------------------------
        # 7. 타깃 (3종목 동시 예측)
        # -------------------------------------------------------------
        for t in [self.ticker]:
            self.merged_df[f"{t}_target"] = self.merged_df[t].pct_change().shift(-1)

        y_all = self.merged_df[[f"{t}_target" for t in [self.ticker]]].dropna().reset_index(drop=True)
        X_scaled = X_scaled.iloc[:len(y_all)]

        # -------------------------------------------------------------
        # 8. 출력 스케일링
        # -------------------------------------------------------------
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = scaler_y.fit_transform(y_all)

        # -------------------------------------------------------------
        # 9. 시퀀스 생성 함수
        # -------------------------------------------------------------
        def create_sequences(X, y, window=40):
            Xs, ys = [], []
            for i in range(len(X) - window):
                Xs.append(X.iloc[i:(i + window)].values)
                ys.append(y[i + window])
            return np.array(Xs), np.array(ys)

        # -------------------------------------------------------------
        # 10. 시퀀스 변환
        # -------------------------------------------------------------
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=40)
        split_idx = int(len(X_seq) * 0.8)

        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # -------------------------------------------------------------
        # 11. 멀티아웃풋 LSTM 모델 정의
        # -------------------------------------------------------------
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),  # 입력 정의
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len([self.ticker]))
        ])

        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mae')

        # -------------------------------------------------------------
        # 12. 학습
        # -------------------------------------------------------------
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=60,
            batch_size=16,
            verbose=1
        )

        # -------------------------------------------------------------
        # 13. 예측 및 복원
        # -------------------------------------------------------------
        preds_scaled = model.predict(X_test)
        preds = scaler_y.inverse_transform(preds_scaled)
        y_test_inv = scaler_y.inverse_transform(y_test)


        # 전체 모델 저장
        save_model(model, f"{self.model_path}")
        joblib.dump(scaler_X, f"{self.scaler_X_path}")
        joblib.dump(scaler_y, f"{self.scaler_y_path}")

        print("[CHECK] macro_full variance summary:")
        numeric_cols = macro_full.select_dtypes(include=[np.number]).columns
        print(macro_full[numeric_cols].var().sort_values().head(10))