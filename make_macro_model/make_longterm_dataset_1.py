import json
import os
from collections import defaultdict
from typing import Optional, List, Dict

import requests
import torch
import yfinance as yf
from typing import Dict, List, Optional, Literal, Tuple, Any
import warnings
from dataclasses import dataclass, field
import joblib
import pandas as pd
import numpy as np
from keras.src.saving import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from agents.dump import CAPSTONE_OPENAI_API
from agents.macro_shap_llm import LLMExplainer, AttributionAnalyzer
from agents.macro_sub import get_std_pred
from debate_ver4.prompts import REBUTTAL_PROMPTS

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 공통 설정
# ============================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)



dir_info = {
    "data_dir": os.path.join(PROJECT_ROOT, "data", "processed"),
    "model_dir": os.path.join(PROJECT_ROOT, "models"),
}
save_dir = dir_info["data_dir"]
model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]


@dataclass
class Target:
    """예측 목표값 + 불확실성 정보 포함
    - next_close: 다음 거래일 종가 예측치
    - uncertainty: Monte Carlo Dropout 기반 예측 표준편차(σ)
    - confidence: 모델 신뢰도 β (정규화된 신뢰도; 선택적)
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None

@dataclass
class Opinion:
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str

@dataclass
class RoundLog:
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    """에이전트 입력 원천 데이터(필요 시 자유 확장)
    - sentimental: 심리/커뮤니티/뉴스 스냅샷
    - fundamental: 재무/밸류에이션 요약
    - technical  : 가격/지표 스냅샷
    - last_price : 최신 종가
    - currency   : 통화코드
    """
    SentimentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    FundamentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    TechnicalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: Optional[str] = None


# ============================================================
# MacroSentimentAgent — 시장·거시경제 시계열 기반
# ============================================================
class MacroSentimentAgentDataset:
    OPENAI_URL = "https://api.openai.com/v1/responses"

    def __init__(self,
                 agent_id="MacroSentiAgent",
                 preferred_models: Optional[List[str]] = None,
                 option_model: Optional[str] = None,
                 verbose: bool = False,
                 temperature: float = 0.2,
                 ticker = None,
                 **kwargs
                 ):
        self.merged_df = None
        self.price_df = None
        self.y_test = None
        self.X_test = None
        self.y_seq = None
        self.X_seq = None
        self.y_train = None
        self.X_train = None
        self.y_scaled = None
        self.y_all = None
        self.X_all = None
        self.feature_cols = None
        self.price_cols = None
        self.macro_cols = None
        self.macro_full = None
        self.temperature = None
        self.target = None
        self.opinions = None
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
        self.ticker_name = ticker

        self.model_path = f"{model_dir}/{ticker}_{agent_id}.h5"
        self.scaler_X_path = f"{model_dir}/scaler_X.pkl"
        self.scaler_y_path = f"{model_dir}/scaler_y.pkl"

        self.tickers = [self.ticker_name] or ["AAPL", "MSFT", "NVDA"]
        # self.target_tickers = target_tickers or ["AAPL", "MSFT", "NVDA"]

        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.macro_df = None
        self.pred_df = None
        self.X_scaled = None

        self.start_date = "2020-01-01"
        self.end_date = '2024-12-31'

        self.window_size = 40
        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if option_model:
            self.preferred_models = [option_model] + [
                m for m in self.preferred_models if m != option_model
            ]

        # 공통 헤더
        self.api_key = CAPSTONE_OPENAI_API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.temperature = temperature # Temperature 설정
        self.verbose = verbose            # 디버깅 모드
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)

    def fetch_data(self):
        """매크로 및 개별 종목 데이터 수집"""
        print(f"[INFO] Collecting macro features ({len(self.macro_tickers)} tickers)...")

        # --------------------------------------------
        # ① 매크로 자산 데이터 수집
        # --------------------------------------------
        df_macro = yf.download(
            tickers=list(self.macro_tickers.values()),
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=False
        )


        # MultiIndex → 단일 컬럼명 통일
        if isinstance(df_macro.columns, pd.MultiIndex):
            # yfinance 컬럼 구조가 (Ticker, Price) 또는 (Price, Ticker)인 경우 자동 처리
            if df_macro.columns.names == ["Ticker", "Price"]:
                df_macro.columns = [f"{t}_{col}" for t, col in df_macro.columns]
            else:
                df_macro.columns = [f"{col}_{t}" for col, t in df_macro.columns]

            # ✅ flatten 이후 순서 보정 (Adj Close_SPY → SPY_Adj Close)
            df_macro.columns = [
                "_".join(reversed(c.split("_"))) if c.split("_")[0] in ["Adj", "Close", "Open", "High", "Low", "Volume"] else c
                for c in df_macro.columns
            ]
        else:
            df_macro.columns = [f"macro_{col}" for col in df_macro.columns]

        df_macro.reset_index(inplace=True)
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        print(f"[INFO] Macro columns after flatten: {df_macro.columns[:10].tolist()}")
        print(f"[MacroSentimentAgent] Macro data loaded: {df_macro.shape}")

        # --------------------------------------------
        # ② 개별 종목 데이터 수집
        # --------------------------------------------
        stock_dfs = []
        tickers = [self.ticker_name] if isinstance(self.ticker_name, str) else (self.ticker_name or ["NVDA"])
        for t in tickers:
            print(f"[INFO] Downloading {t} ...")
            try:
                df_t = yf.download(t, start=self.start_date, end=self.end_date, interval="1d", group_by="ticker")

                # ✅ MultiIndex → 단일 인덱스로 변환
                if isinstance(df_t.columns, pd.MultiIndex):
                    df_t.columns = [f"{t}_{col}" for t, col in df_t.columns]
                    print(f"[INFO] {t} columns after flatten:", df_t.columns.tolist())

                df_t = df_t.rename(columns={
                    f"{t}_Open": f"{t}_Open",
                    f"{t}_High": f"{t}_High",
                    f"{t}_Low": f"{t}_Low",
                    f"{t}_Close": f"{t}_Close",
                    f"{t}_Volume": f"{t}_Volume"
                })

                df_t.reset_index(inplace=True)

                # ✅ 파생 컬럼 생성
                df_t[f"{t}_ret1"] = df_t[f"{t}_Close"].pct_change()
                df_t[f"{t}_ma5"] = df_t[f"{t}_Close"].rolling(5).mean()
                df_t[f"{t}_ma10"] = df_t[f"{t}_Close"].rolling(10).mean()

                stock_dfs.append(df_t)
            except Exception as e:
                print(f"[WARN] {t} download failed: {e}")

        # --------------------------------------------
        # ③ 병합 (outer join으로 공백일 포함)
        # --------------------------------------------
        merged_df = df_macro
        for df_t in stock_dfs:
            # ✅ 혹시라도 MultiIndex가 남아있으면 평탄화
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = ["_".join([str(c) for c in col if c]) for col in df_t.columns]
            merged_df = pd.merge(merged_df, df_t, on="Date", how="outer")


        # --------------------------------------------
        # ④ 결측치 및 정리
        # --------------------------------------------
        merged_df = merged_df.sort_values("Date")
        merged_df = merged_df.fillna(method="ffill").fillna(method="bfill")
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # 중복 제거
        merged_df.reset_index(drop=True, inplace=True)

        self.data = merged_df
        print(f"[MacroSentimentAgent] Combined data shape: {merged_df.shape}")
        return merged_df


    def add_features(self):
        """매크로 데이터 기반 피처 엔지니어링"""
        try:
            if self.data is None:
                raise ValueError("[ERROR] add_features() 호출 전 fetch_data()를 실행해야 합니다.")

            df = self.data.copy()
            df.index.name = "Date"

            # ✅ 이미 Date 컬럼이 있으면 reset_index()를 생략
            if "Date" not in df.columns:
                df.reset_index(inplace=True)
            else:
                df = df.copy()  # 그대로 유지

            # --------------------------------------------
            # (1) 매크로 피처 생성
            # --------------------------------------------
            for ticker in self.macro_tickers.values():
                col_name = f"{ticker}_Close"
                if col_name in df.columns:
                    df[f"{ticker}_ret_1d"] = df[col_name].pct_change()

            # 금리 스프레드
            if "^TNX_Close" in df.columns and "^IRX_Close" in df.columns:
                df["Yield_spread"] = df["^TNX_Close"] - df["^IRX_Close"]

            # 위험심리 지표
            if (
                    "SPY_ret_1d" in df.columns
                    and "DX-Y.NYB_ret_1d" in df.columns
                    and "^VIX_ret_1d" in df.columns
            ):
                df["Risk_Sentiment"] = (
                        df["SPY_ret_1d"] - df["DX-Y.NYB_ret_1d"] - df["^VIX_ret_1d"]
                )

            # --------------------------------------------
            # (2) 결측치 처리
            # --------------------------------------------
            df = df.fillna(method="ffill").fillna(method="bfill")

            # --------------------------------------------
            # (3) 피처 순서 정렬 (스케일러 기준 정렬 제거)
            # --------------------------------------------
            print("[INFO] feature order sync skipped (manual override)")
            df = df.reindex(sorted(df.columns), axis=1)

            # --------------------------------------------
            # (4) reset_index() 중복 제거 — 기존 코드 수정
            # --------------------------------------------
            # 기존 코드에서는 여기서 또 reset_index()를 실행했지만,
            # 이미 Date 컬럼이 있을 경우 중복 오류가 발생하므로 조건부로 수행
            if "Date" not in df.columns:
                df.reset_index(inplace=True)

            # --------------------------------------------
            # (5) 마무리
            # --------------------------------------------
            self.data = df
            self.macro_df = df.copy()   # ✅ macro_predictor에서도 동일 데이터 참조
            print(f"[MacroSentimentAgent] Feature engineering complete. Final shape: {df.shape}")
            return df

        except Exception as e:
            print(f"[add_features]Error: {e}")


    def save_csv(self):
        path = os.path.join(OUTPUT_DIR, "macro_sentiment.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data.to_csv(path, index=True)
        print(f"[MacroSentimentAgent] Saved {path}")


    # -------------------------------------------------------------
    # 1. 티커별 종가 저장
    # -------------------------------------------------------------
    def close_price_fetch(self, ticker_name):
        """티커별 일별 종가 저장 (NVDA_Close 중복, MultiIndex 문제 완벽 방지)"""
        print(f"[INFO] Fetching close prices for {ticker_name} ...")

        df_prices = yf.download(
            tickers=ticker_name,
            start="2020-01-01",
            end="2025-01-03",
            group_by="ticker",
            auto_adjust=False
        )

        # (1) MultiIndex 구조 처리
        if isinstance(df_prices.columns, pd.MultiIndex):
            df_prices.columns = ["_".join(col).strip() for col in df_prices.columns.values]
            close_cols = [c for c in df_prices.columns if "Close" in c]
            if close_cols:
                df_prices = df_prices[close_cols]
                df_prices.columns = [f"{ticker_name}_Close" for _ in close_cols]
            else:
                raise KeyError(f"[ERROR] MultiIndex 구조에서 Close 컬럼을 찾을 수 없습니다: {df_prices.columns.tolist()}")

        # (2) 단일 컬럼 구조
        elif isinstance(df_prices, pd.DataFrame) and "Close" in df_prices.columns:
            df_prices = df_prices[["Close"]].rename(columns={"Close": f"{ticker_name}_Close"})

        # (3) Series 반환
        elif isinstance(df_prices, pd.Series):
            df_prices = df_prices.to_frame(name=f"{ticker_name}_Close")

        else:
            raise KeyError(f"[ERROR] Close 컬럼을 찾을 수 없습니다: {df_prices.columns.tolist()}")

        # 날짜 컬럼 확보
        if "Date" not in df_prices.columns:
            df_prices.reset_index(inplace=True)
            if "index" in df_prices.columns:
                df_prices.rename(columns={"index": "Date"}, inplace=True)

        # 중복 컬럼 정리
        df_prices = df_prices.loc[:, ~df_prices.columns.duplicated()]
        if "Date.1" in df_prices.columns:
            df_prices = df_prices.drop(columns=["Date.1"])

        # 저장
        path = os.path.join(OUTPUT_DIR, f"daily_closePrice_{ticker_name}.csv")
        if os.path.exists(path):
            os.remove(path)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_prices.to_csv(path, index=False)

        print(f"✅ 저장 완료: {df_prices.shape} rows >> {path}")
        return df_prices

    # -------------------------------------------------------------
    # 2. 매크로 + 주가 병합 데이터셋 생성
    # -------------------------------------------------------------
    def make_dataset_seq(self, ticker_name):
        self.ticker_name = ticker_name
        print(f"[INFO] make_dataset_seq() 시작 — {ticker_name}")

        macro_path = os.path.join(OUTPUT_DIR, "macro_sentiment.csv")
        price_path = os.path.join(OUTPUT_DIR, f"daily_closePrice_{ticker_name}.csv")

        # 파일 존재 검증
        if not os.path.exists(macro_path):
            raise FileNotFoundError(f"[ERROR] macro_sentiment.csv가 없습니다 → {macro_path}")
        if not os.path.exists(price_path):
            raise FileNotFoundError(f"[ERROR] {price_path} 파일이 없습니다. close_price_fetch('{ticker_name}') 실행 필요.")

        # 파일 로드
        self.macro_df = pd.read_csv(macro_path)
        self.price_df = pd.read_csv(price_path)

        # 중복 및 결측 처리
        self.price_df = self.price_df.loc[:, ~self.price_df.columns.duplicated()]
        for col in self.price_df.columns:
            if col.endswith(".1"):
                base_col = col.split(".")[0]
                if base_col not in self.price_df.columns:
                    self.price_df.rename(columns={col: base_col}, inplace=True)

        # 날짜 정리
        self.macro_df["Date"] = pd.to_datetime(self.macro_df["Date"])
        self.price_df["Date"] = pd.to_datetime(self.price_df["Date"])

        # -------------------------------------------------------------
        # 매크로 피처 확장
        # -------------------------------------------------------------
        numeric_df = self.macro_df.select_dtypes(include=[np.number])
        macro_features = [c for c in numeric_df.columns if c != "Date"]
        macro_ret = numeric_df[macro_features].pct_change()
        macro_ret.columns = [f"{c}_ret" for c in macro_ret.columns]

        self.macro_full = pd.concat([self.macro_df, macro_ret], axis=1)
        self.macro_full = self.macro_full.replace([np.inf, -np.inf], np.nan).dropna(subset=["Date"]).fillna(0)

        # -------------------------------------------------------------
        # 주가 기반 피처 생성
        # -------------------------------------------------------------
        close_col = [c for c in self.price_df.columns if "Close" in c and ticker_name in c]
        if not close_col:
            close_col = [c for c in self.price_df.columns if "Close" in c]
        if not close_col:
            raise KeyError(f"[ERROR] {ticker_name} 종가 컬럼을 찾을 수 없습니다: {self.price_df.columns.tolist()}")
        close_col = close_col[0]

        self.price_df[f"{ticker_name}_ret1"] = self.price_df[close_col].pct_change()
        self.price_df[f"{ticker_name}_ma5"] = self.price_df[close_col].rolling(5).mean()
        self.price_df[f"{ticker_name}_ma10"] = self.price_df[close_col].rolling(10).mean()
        self.price_df = self.price_df.fillna(method="bfill")

        # -------------------------------------------------------------
        # 병합 수행
        # -------------------------------------------------------------
        print("[INFO] 병합 전 Date 일치화 중...")
        self.macro_df["Date"] = pd.to_datetime(self.macro_df["Date"])
        self.price_df["Date"] = pd.to_datetime(self.price_df["Date"])

        merged_df = pd.merge(self.macro_df, self.price_df, on="Date", how="inner")
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # 중복 제거
        if "Date.1" in merged_df.columns:
            merged_df.drop(columns=["Date.1"], inplace=True)
        if merged_df.empty:
            raise ValueError("[ERROR] macro_df와 price_df 간 공통 Date가 없습니다.")

        # ✅ NVDA_Close_x/y 문제 해결
        rename_map = {}
        for c in merged_df.columns:
            if c.endswith("_Close_x") or c.endswith("_Close_y"):
                rename_map[c] = c.replace("_Close_x", "_Close").replace("_Close_y", "_Close")
        merged_df.rename(columns=rename_map, inplace=True)

        self.merged_df = merged_df

        print(f"[INFO] 병합 후 데이터 shape: {merged_df.shape}")
        print(f"[DEBUG] NVDA 관련 컬럼: {[c for c in merged_df.columns if 'NVDA' in c]}")

        # -------------------------------------------------------------
        # Feature 구성
        # -------------------------------------------------------------
        target_ticker_list = ["AAPL", "MSFT", "NVDA"]
        self.macro_cols = [c for c in self.macro_full.columns if c != "Date"]
        self.price_cols = [c for c in merged_df.columns if any(t in c for t in target_ticker_list) and ("_ret" in c or "_ma" in c)]
        self.feature_cols = self.macro_cols + self.price_cols

        self.X_all = merged_df[self.feature_cols]

        # -------------------------------------------------------------
        # 입력 스케일링
        # -------------------------------------------------------------
        self.scaler_X = StandardScaler()
        self.X_scaled = pd.DataFrame(self.scaler_X.fit_transform(self.X_all), columns=self.feature_cols)

        # -------------------------------------------------------------
        # 타깃
        # -------------------------------------------------------------
        close_target = [c for c in merged_df.columns if f"{ticker_name}_Close" in c]
        if not close_target:
            raise KeyError(f"[ERROR] 타깃 종가 컬럼({ticker_name}_Close)이 없습니다: {merged_df.columns.tolist()}")
        close_target = close_target[0]

        merged_df[f"{ticker_name}_target"] = merged_df[close_target].pct_change().shift(-1)
        self.y_all = merged_df[[f"{ticker_name}_target"]].dropna().reset_index(drop=True)
        self.X_scaled = self.X_scaled.iloc[: len(self.y_all)]

        # -------------------------------------------------------------
        # 출력 스케일링
        # -------------------------------------------------------------
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.y_scaled = self.scaler_y.fit_transform(self.y_all)

        # -------------------------------------------------------------
        # 시퀀스 생성
        # -------------------------------------------------------------
        def create_sequences(X, y, window=40):
            Xs, ys = [], []
            for i in range(len(X) - window):
                Xs.append(X.iloc[i:(i + window)].values)
                ys.append(y[i + window])
            return np.array(Xs), np.array(ys)

        self.X_seq, self.y_seq = create_sequences(self.X_scaled, self.y_scaled, window=40)
        split_idx = int(len(self.X_seq) * 0.8)
        self.X_train, self.X_test = self.X_seq[:split_idx], self.X_seq[split_idx:]
        self.y_train, self.y_test = self.y_seq[:split_idx], self.y_seq[split_idx:]

        # -------------------------------------------------------------
        # 데이터셋 저장
        # -------------------------------------------------------------
        csv_path = os.path.join(OUTPUT_DIR, f"{ticker_name}_{self.__class__.__name__}_dataset.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        merged_df.to_csv(csv_path, index=False)
        print(f"[OK] 병합 완료 및 저장 → {csv_path}")

        return self.X_train, self.X_test, self.y_train, self.y_test, self.X_seq, self.y_seq, self.feature_cols



    def macro_searcher_add_funs(self, X_seq, feature_cols, agent_id=None):
        """
        searcher()의 마지막 단계와 동일한 기능:
        - 최신 윈도우 데이터(X_tensor) 생성
        - feature_dict 구성
        - StockData 초기화 및 값 저장
        """

        # ---------------------------------------------------------
        # 기본 정보 세팅
        # ---------------------------------------------------------
        if agent_id is None:
            agent_id = self.agent_id

        ticker_name = self.ticker_name or "UNKNOWN"

        # StockData 객체 생성
        self.stockdata = StockData()
        self.stockdata.ticker = ticker_name

        # ---------------------------------------------------------
        # 최신 윈도우(X_latest → X_tensor)
        # ---------------------------------------------------------
        X_latest = X_seq[-1:]              # 마지막 시퀀스 (예측용)
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # DataFrame 변환
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)

        # feature_dict 구성
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}

        # agent_id 이름으로 속성 추가 (예: self.stockdata.MacroSentiAgent)
        setattr(self.stockdata, agent_id, feature_dict)

        # ---------------------------------------------------------
        # 종가 및 통화 정보 수집
        # ---------------------------------------------------------
        try:
            data = yf.download(ticker_name, period="1d", interval="1d")
            if not data.empty:
                self.stockdata.last_price = float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"[WARN] yfinance 오류 발생 (가격): {e}")

        try:
            self.stockdata.currency = yf.Ticker(ticker_name).info.get("currency", "USD")
        except Exception as e:
            print(f"[WARN] yfinance 오류 발생 (통화): {e}")
            self.stockdata.currency = "USD"

        print(f"✅ StockData 생성 완료: {ticker_name} / {self.stockdata.currency}")

        return X_tensor, self.stockdata



    # 모델 생성
    # -------------------------------------------------------------
    # 모델 생성 및 학습
    # -------------------------------------------------------------
    def make_lstm_macro_model(self, ticker_name, X_train, y_train):
        print(f'[make_lstm_macro_model] 시작: {ticker_name}')

        # -------------------------------------------------------------
        # 1. 스케일러 생성 및 스케일링
        # -------------------------------------------------------------
        print("[INFO] 스케일링 시작")
        self.scaler_X = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
        self.scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

        X_scaled = self.scaler_X.transform(X_train.reshape(-1, X_train.shape[-1]))
        X_scaled = X_scaled.reshape(X_train.shape)
        y_scaled = self.scaler_y.transform(y_train.reshape(-1, 1))

        print(f"[INFO] X_scaled shape: {X_scaled.shape}, y_scaled shape: {y_scaled.shape}")

        # -------------------------------------------------------------
        # 2. LSTM 모델 정의
        # -------------------------------------------------------------
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mae')

        # -------------------------------------------------------------
        # 3. 학습
        # -------------------------------------------------------------
        print("[INFO] 모델 학습 시작...")
        history = self.model.fit(
            X_scaled, y_scaled,
            validation_split=0.1,
            epochs=60,
            batch_size=16,
            verbose=1
        )

        # -------------------------------------------------------------
        # 4. 모델 및 스케일러 저장
        # -------------------------------------------------------------
        model_path = os.path.join(model_dir, f"{ticker_name}_{self.agent_id}.h5")
        scaler_X_path = os.path.join(model_dir, "scaler_X.pkl")
        scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")

        os.makedirs(model_dir, exist_ok=True)
        self.model.save(model_path)
        joblib.dump(self.scaler_X, scaler_X_path)
        joblib.dump(self.scaler_y, scaler_y_path)

        print(f"✅ 모델 및 스케일러 저장 완료:")
        print(f"   - Model: {model_path}")
        print(f"   - Scaler X: {scaler_X_path}")
        print(f"   - Scaler y: {scaler_y_path}")
        print(f"✅ pretraining finished.\n")

        return self.model




    # -------------------------------------------------------------
    # 1. 모델 및 스케일러 로드
    # -------------------------------------------------------------
    def load_assets(self):
        print("[INFO] 모델 및 스케일러 로드 중...")
        self.model = load_model(self.model_path, compile=False)
        self.scaler_X = joblib.load(self.scaler_X_path)
        self.scaler_y = joblib.load(self.scaler_y_path)
        print("[OK] 모델 및 스케일러 로드 완료")


    #predict
    def macro_predictor(self, X_seq):
        print("[INFO] 예측 수행 중...")

        # 1. 모델 예측
        self.load_assets()
        pred_scaled = self.model.predict(X_seq)
        pred_inv = self.scaler_y.inverse_transform(pred_scaled)

        # 2. 종가 추출
        last_prices = {}
        for t in self.tickers:
            close_candidates = [
                c for c in self.macro_df.columns
                if "Close" in c and (c.startswith(t) or c.endswith(t))
            ]
            if not close_candidates:
                raise ValueError(f"[ERROR] {t}의 종가 컬럼을 찾을 수 없습니다. "
                                 f"현재 컬럼들: {self.macro_df.columns.tolist()}"
                                 )
            last_prices[t] = self.macro_df[close_candidates[0]].iloc[-1]

        # 3. 예측 종가 및 수익률 계산
        records = []
        pred_prices = {}
        for i, t in enumerate(self.tickers):
            pred_ret = float(pred_inv[0][i])
            last_price = float(last_prices[t])
            next_price = last_price * (1 + pred_ret)
            pred_prices[t] = next_price

            records.append({
                "Ticker": t,
                "Last_Close": last_price,
                "Predicted_Close": next_price,
                "Predicted_Return": pred_ret,
                "Predicted_%": pred_ret * 100
            })

            print(f"{t}: 마지막 종가={last_price:.2f} → 예측 종가={next_price:.2f} (예상 수익률 {pred_ret*100:.2f}%)")

        # 4. Monte Carlo Dropout 불확실성
        mean_pred, std_pred, confidence, predicted_price = get_std_pred(
            self.model, X_seq, n_samples=30, scaler_y=self.scaler_y, stockdata=self.stockdata
        )

        # 5. 결과 병합
        for i, r in enumerate(records):
            r["uncertainty"] = float(std_pred[i]) if len(std_pred) > 1 else float(std_pred[-1])
            r["confidence"] = float(confidence[i]) if len(confidence) > 1 else float(confidence[-1])

        pred_df = pd.DataFrame(records).round(4)
        self.pred_df = pred_df
        self.pred_prices = pred_prices

        print("\n================= 예측 결과 (표) =================")
        print(pred_df)

        print("\n================= 예측 결과 (값) =================")
        print(pred_prices)

        # 단일 티커일 경우 target 요약 제공
        self.target = Target(
            next_close=float(pred_df["Predicted_Close"].iloc[-1]),
            uncertainty=float(std_pred[-1]),
            confidence=float(pred_df["confidence"].iloc[-1])
        )


        return self.pred_prices, self.target



    def macro_reviewer_draft(self):
        temporal_summary, causal_summary, interaction_summary = self.make_macro_shap()
        # -------------------------------
        # 4️⃣ llm 생성
        # -------------------------------
        print("\n4️⃣ Generating explanation using LLM...")

        llm  = LLMExplainer()
        feature_summary = feature_df.tail(5).describe().round(3).to_dict()
        explanation = llm.generate_explanation(feature_summary, self.pred_prices,
                                               importance_dict,
                                               temporal_summary, causal_summary,
                                               interaction_summary)

        print(f"\n================= pred_prices:{self.pred_prices} =================")

        print("\n================= LLM Explanation =================")
        print(explanation)
        print("===================================================")

        total_json = {
            'agent_id' : self.agent_id,
            'target' : self.target,
            'reason' : explanation
        }

        stock_data = {
            'temporal_summary' : temporal_summary,
            'causal_summary' : causal_summary,
            'interaction_summary' : interaction_summary

        }

        context = json.dumps({
            "agent_id": self.agent_id,
            "predicted_next_close": round(self.target.next_close, 3),
            "uncertainty_sigma": round(self.target.uncertainty or 0.0, 4),
            "confidence_beta": round(self.target.confidence or 0.0, 4),
            "latest_data": str(stock_data)
        }, ensure_ascii=False, indent=2)

        reason = explanation

        # 4) Opinion 기록/반환 (항상 최신 값 append)
        self.opinions.append(Opinion(agent_id=self.agent_id, target=self.target, reason=reason))

        return total_json, self.opinions[-1]




    def make_macro_shap(self):
        # -------------------------------
        # 3️⃣ SHAP 계산
        # -------------------------------
        # --- (run() 안의 안전 처리) ---
        X_scaled = self.X_scaled.astype(np.float32)
        X_scaled = X_scaled[:, :, :300]
        feature_names = feature_names[:300]

        print("\n3️⃣ Calculating feature importance...")
        analyzer = AttributionAnalyzer(self.model)
        importance_dict, temporal_df, causal_df, interaction_df = analyzer.run_all_shap(X_scaled, feature_names)

        temporal_summary = temporal_df.head().to_dict(orient="records") if temporal_df is not None else []
        causal_summary = causal_df.to_dict(orient="records") if causal_df is not None else []
        if isinstance(interaction_df, pd.DataFrame):
            interaction_summary = interaction_df.iloc[:5, :5].round(3).to_dict()
        else:
            interaction_summary = {}

        return temporal_summary, causal_summary, interaction_summary



    #[base_agent.py]
    def _msg(self, role: str, content: str) -> dict:
        """OpenAI ChatCompletion용 메시지 구조 생성"""
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}


    #[base_agent.py]
    def macro_reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""

        # 메시지 생성 (context 구성은 별도 헬퍼에서)
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )

        # LLM 호출
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"}
                },
                "required": ["stance", "message"],
                "additionalProperties": False
            }
        )

        # 결과 정리 및 기록
        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )

        # 저장
        self.rebuttals[round].append(result)

        # 디버깅 로그
        if self.verbose:
            print(
                f"[{self.agent_id}] rebuttal 생성 → {result.stance} "
                f"({my_opinion.agent_id} → {other_opinion.agent_id})"
            )

        return result



    #[m_agent.py]
    def _build_messages_rebuttal(self,
                                 my_opinion: Opinion,
                                 target_opinion: Opinion,
                                 stock_data: StockData) -> tuple[str, str]:

        t = stock_data.ticker or "UNKNOWN"
        ccy = (stock_data.currency or "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        ctx = {
            "ticker": t,
            "currency": ccy,
            "data_summary": getattr(stock_data, self.agent_id, {}).get("feature_cols", []),
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:2000],
                "uncertainty": float(my_opinion.target.uncertainty),
                "confidence": float(my_opinion.target.confidence),
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
                "uncertainty": float(target_opinion.target.uncertainty),
                "confidence": float(target_opinion.target.confidence),
            }
        }
        # 각 컬럼별 최근 시계열 그대로 포함
        # (최근 7~14일 정도면 LLM이 이해 가능한 범위)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[self.window_size:]  # 최근 14일치 전체 시계열
            else:
                ctx[col] = [values]

        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        user_text   = REBUTTAL_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text



    #[base_agent.py] OpenAI API 호출
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
        payload_base = {
            "input": [msg_sys, msg_user],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Response",
                    "strict": True,
                    "schema": schema_obj,
                }
            },
            "temperature": self.temperature,
        }
        last_err = None
        for model in self.preferred_models:
            payload = dict(payload_base, model=model)
            try:
                r = requests.post(self.OPENAI_URL, headers=self.headers, json=payload, timeout=120)
                if r.ok:
                    data = r.json()
                    # 1) output_text 우선 사용
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}  # JSON 실패 시 원문 텍스트 보존
                    # 2) output 배열에서 텍스트 모으기
                    out = data.get("output")
                    if isinstance(out, list) and out:
                        texts = []
                        for blk in out:
                            for c in blk.get("content", []):
                                if "text" in c:
                                    texts.append(c["text"])
                        joined = "\n".join(t for t in texts if t)
                        if joined.strip():
                            try:
                                return json.loads(joined)
                            except Exception:
                                return {"reason": joined}
                    # 비정상 응답
                    return {}
                # 400/404는 다음 모델로 폴백
                if r.status_code in (400, 404):
                    last_err = (r.status_code, r.text)
                    continue
                # 기타 에러는 즉시 예외
                r.raise_for_status()
            except Exception as e:
                self._p(f"■ 모델 {model} 실패: {e}")
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")













    def macro_reviewer_revise(
            self,
            my_opinion: Opinion,
            others: List[Opinion],
            rebuttals: List[Rebuttal],
            stock_data: StockData,
            fine_tune: bool = True,
            lr: float = 1e-4,
            epochs: int = 20,
    ):
        """
        Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - fine-tuning (수익률 단위)
        - reasoning 생성
        """
        gamma = getattr(self, "gamma", 0.3)               # 수렴율 (0~1)
        delta_limit = getattr(self, "delta_limit", 0.05)  # fine-tuning 보정 한계

        try:
            # ===================================
            # ① β 계산 (불확실성 작을수록 신뢰 높음)
            # ===================================
            my_price = my_opinion.target.next_close
            my_sigma = abs(my_opinion.target.uncertainty or 1e-6)

            other_prices = np.array([o.target.next_close for o in others])
            other_sigmas = np.array([abs(o.target.uncertainty or 1e-6) for o in others])

            all_sigmas = np.concatenate([[my_sigma], other_sigmas])
            all_prices = np.concatenate([[my_price], other_prices])

            inv_sigmas = 1 / (all_sigmas + 1e-6)
            betas = inv_sigmas / inv_sigmas.sum()

            # ===================================
            # ② 논문식 수렴 업데이트
            #     y_i_rev = y_i + γ Σ β_j (y_j - y_i)
            # ===================================
            delta = np.sum(betas[1:] * (other_prices - my_price))
            revised_price = my_price + gamma * delta

        except Exception as e:
            print(f"[{self.agent_id}] revised_target 계산 실패: {e}")
            revised_price = my_opinion.target.next_close
            current_price = getattr(self.stockdata, "last_price", 100.0)
            price_uplimit = current_price * (1 + delta_limit)
            price_downlimit = current_price * (1 - delta_limit)
            revised_price = min(max(revised_price, price_downlimit), price_uplimit)

        # ===================================
        # ③ Fine-tuning (return 단위)
        # ===================================
        loss_value = None
        if fine_tune and hasattr(self, "model"):
            try:
                current_price = getattr(self.stockdata, "last_price", 100.0)
                revised_return = (revised_price / current_price) - 1  # 🔹수익률 변환

                X_input = self.searcher(self.ticker)
                device = next(self.model.parameters()).device
                X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
                y_tensor = torch.tensor([[revised_return]], dtype=torch.float32).to(device)

                self.model.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()

                for _ in range(epochs):
                    optimizer.zero_grad()
                    pred = self.model(X_tensor)
                    delta_loss = pred - y_tensor
                    loss = criterion(pred - delta_loss, y_tensor)
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
                print(f"[{self.agent_id}] fine-tuning 완료: loss={loss_value:.6f}")

            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")

        # ===================================
        # ④ fine-tuning 이후 새 예측 생성
        # ===================================
        try:
            X_latest = self.searcher(self.ticker)
            new_target = self.predict(X_latest)
        except Exception as e:
            print(f"[{self.agent_id}] predict 실패: {e}")
            new_target = my_opinion.target

        # ===================================
        # ⑤ reasoning 생성
        # ===================================
        try:
            sys_text, user_text = self._build_messages_revision(
                my_opinion=my_opinion,
                others=others,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )
        except Exception as e:
            print(f"[{self.agent_id}] _build_messages_revision 실패: {e}")
            sys_text, user_text = (
                "너는 금융 분석가다. 간단히 reason만 생성하라.",
                json.dumps({"reason": "기본 메시지 생성 실패"}),
            )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        )

        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=new_target,
            reason=revised_reason,
        )

        self.opinions.append(revised_opinion)
        print(f"[{self.agent_id}] revise 완료 → new_close={new_target.next_close:.2f}, loss={loss_value}")
        return self.opinions[-1]