import os
import json
from dataclasses import asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.agents import dir_info, agents_info, common_params
from core.macro_classes.macro_llm import GradientAnalyzer
from agents.base_agent import BaseAgent, Target, StockData, Opinion, Rebuttal
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS

# =============================================================================
# 상수 정의 (피처 목록 고정 - searcher와 pretrain 일치)
# =============================================================================
MACRO_TICKERS = {
    "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
    "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
    "^VIX": "^VIX",
    "DX-Y.NYB": "DX-Y.NYB",
    "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
    "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F"
}

# 최종 사용할 피처 리스트 정의 (확장 버전 - OHLCV 모두 포함)
_macro_base_features = []
for t in sorted(MACRO_TICKERS.values()):
    # OHLCV + Return
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        _macro_base_features.append(f"{t}_{col}")
    _macro_base_features.append(f"{t}_ret_1d")

_macro_derived_features = ["Yield_spread", "Risk_Sentiment"]
_stock_features = ["ret1", "ma5", "ma10"]

# 모든 피처 리스트 합치기 및 정렬
FINAL_FEATURES = sorted(_macro_base_features + _macro_derived_features + _stock_features)

model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]


class MacroAgent(BaseAgent, nn.Module):
    def __init__(self,
                 base_date=datetime.today(),
                 window=None,
                 ticker=None,
                 agent_id='MacroAgent',
                 data_dir=None,
                 **kwargs):
        # 1) nn.Module 먼저 초기화
        nn.Module.__init__(self)

        # 2) BaseAgent 초기화
        if data_dir is None:
            data_dir = dir_info.get("data_dir", "data")
        BaseAgent.__init__(self, agent_id=agent_id, ticker=ticker, data_dir=data_dir, **kwargs)

        # Config에서 하이퍼파라미터 가져오기
        cfg = agents_info.get(agent_id, {})

        self.agent_id = agent_id
        self.base_date = base_date
        self.window = int(window) if window is not None else cfg.get("window_size", 40)
        self.window_size = self.window
        self.tickers = [ticker] if ticker else []
        self.ticker = ticker

        # 모델 경로 (ticker가 있으면 설정, 없으면 나중에 searcher에서 설정)
        if ticker:
            self.model_path = os.path.join(model_dir, f"{ticker}_{agent_id}.pt")
            self.scaler_X_path = os.path.join(model_dir, "scalers", f"{ticker}_{agent_id}_xscaler.pkl")
            self.scaler_y_path = os.path.join(model_dir, "scalers", f"{ticker}_{agent_id}_yscaler.pkl")
        else:
            self.model_path = None
            self.scaler_X_path = None
            self.scaler_y_path = None

        # 모델 하이퍼파라미터 설정 (Config 기반)
        self.input_dim = cfg.get("input_dim", len(FINAL_FEATURES)) 
        self.output_dim = len(self.tickers) if self.tickers else 1
        hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
        dropout_rates = cfg.get("dropout_rates", [0.3, 0.3, 0.2])

        # LSTM 레이어 즉시 정의 (TechnicalAgent 패턴)
        self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
        self.drop1 = nn.Dropout(dropout_rates[0])
        self.drop2 = nn.Dropout(dropout_rates[1])
        self.drop3 = nn.Dropout(dropout_rates[2])
        self.fc1 = nn.Linear(hidden_dims[2], 32)
        self.fc2 = nn.Linear(32, self.output_dim)

        # 데이터 관련
        self.scaler_X = None
        self.scaler_y = None
        self.macro_df = None
        self.X_scaled = None
        self.X_raw = None
        self.last_price = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # get_opinion - agent.pretrain()에서 사용됨
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델 forward pass
        입력: x (B, T, F)
        출력: (B, output_dim) - 다음날 수익률(return)
        """
        # LSTM layers
        h1, _ = self.lstm1(x)
        h1 = self.drop1(h1)
        h2, _ = self.lstm2(h1)
        h2 = self.drop2(h2)
        h3, _ = self.lstm3(h2)
        h3 = self.drop3(h3)

        # 마지막 시점만 사용 (batch, seq_len, hidden) -> (batch, hidden)
        h3_last = h3[:, -1, :]

        # Dense layers
        out = torch.relu(self.fc1(h3_last))
        out = self.fc2(out)
        return out

    # =========================================================================
    # 내부 데이터 처리 메서드 (통합됨)
    # =========================================================================
    def _fetch_macro_data(self, start_date, end_date):
        """매크로 데이터 수집"""
        print(f"[INFO] Collecting macro features ({len(MACRO_TICKERS)} tickers)...")
        try:
            df_macro = yf.download(
                tickers=list(MACRO_TICKERS.values()),
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False
            )
        except Exception as e:
            print(f"[WARN] Macro data download failed: {e}")
            return pd.DataFrame()

        # MultiIndex 처리
        if isinstance(df_macro.columns, pd.MultiIndex):
            try:
                df_macro = df_macro.stack(level=0)
                df_macro.index.names = ["Date", "Ticker"]
                df_macro = df_macro.unstack(level="Ticker")
                # (Price, Ticker) -> Ticker_Price (예: SPY_Close)로 변환하여 FINAL_FEATURES와 일치시킴
                df_macro.columns = [f"{col[1]}_{col[0]}" for col in df_macro.columns.values]
            except Exception as e:
                print(f"[WARN] Macro MultiIndex processing failed: {e}")
        
        df_macro.reset_index(inplace=True)
        if "Date" in df_macro.columns:
            df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        
        return df_macro

    def _fetch_stock_data(self, ticker, start_date, end_date):
        """개별 주식 데이터 수집"""
        try:
            df_t = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            
            # MultiIndex 평탄화
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df_t.columns]
            
            # 컬럼명 변경 (단순 이름으로)
            t = ticker
            df_t = df_t.rename(columns={
                "Open": f"Open_{t}", "High": f"High_{t}", "Low": f"Low_{t}",
                "Close": f"Close_{t}", "Volume": f"Volume_{t}"
            })
            
            df_t.reset_index(inplace=True)
            if "Date" in df_t.columns:
                df_t["Date"] = pd.to_datetime(df_t["Date"])
                
            return df_t
        except Exception as e:
            print(f"[WARN] Stock data download failed for {ticker}: {e}")
            return pd.DataFrame()

    def _add_derived_features(self, df):
        """파생 변수 생성"""
        df = df.copy()
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
            
        # 1. 매크로 자산별 1일 수익률
        for ticker in MACRO_TICKERS.values():
            col_name = f"{ticker}_Close"
            if col_name in df.columns:
                df[f"{ticker}_ret_1d"] = df[col_name].pct_change()

        # 2. 금리 스프레드
        if "^TNX_Close" in df.columns and "^IRX_Close" in df.columns:
            df["Yield_spread"] = df["^TNX_Close"] - df["^IRX_Close"]

        # 3. 시장 위험 심리
        if "SPY_ret_1d" in df.columns and "DX-Y.NYB_ret_1d" in df.columns and "^VIX_ret_1d" in df.columns:
            df["Risk_Sentiment"] = df["SPY_ret_1d"] - df["DX-Y.NYB_ret_1d"] - df["^VIX_ret_1d"]
            
        df.reset_index(inplace=True)
        return df

    def _prepare_final_dataset(self, macro_df, stock_df, ticker):
        """데이터 병합 및 최종 피처셋 구성 (FINAL_FEATURES 기준)"""
        t = ticker
        
        # 주가 피처 생성
        stock_df = stock_df.copy()
        close_col = f"Close_{t}"
        if close_col in stock_df.columns:
            stock_df["ret1"] = stock_df[close_col].pct_change()
            stock_df["ma5"] = stock_df[close_col].rolling(5).mean()
            stock_df["ma10"] = stock_df[close_col].rolling(10).mean()
        
        # 병합
        if "Date" in macro_df.columns and "Date" in stock_df.columns:
            merged = pd.merge(stock_df, macro_df, on="Date", how="inner").sort_values("Date")
        else:
            print("[WARN] 'Date' column missing for merge")
            return pd.DataFrame(), pd.DataFrame()

        merged = merged.fillna(method='ffill').fillna(method='bfill').dropna()
        merged = merged.reset_index(drop=True)
        
        # FINAL_FEATURES 기준으로 필터링 및 정렬 (누락된 건 0.0)
        X_final = pd.DataFrame(index=merged.index)
        for feature in FINAL_FEATURES:
            if feature in merged.columns:
                X_final[feature] = merged[feature]
            else:
                X_final[feature] = 0.0
                
        return X_final, merged

    # =========================================================================
    # 메인 메서드: searcher / pretrain
    #  - searcher: MacroAData를 사용해 CSV/원시 피처만 생성 + 최신 윈도우 준비
    #  - pretrain: CSV 기반으로만 스케일링/시퀀싱/학습 수행
    # =========================================================================

    def _ensure_macro_csv(self, ticker: str, rebuild: bool = False) -> None:
        """
        MacroAData 없이, searcher 내부에서 직접:
        - 매크로/주가 데이터를 API로 수집
        - 파생 피처를 생성
        - raw CSV를 생성한다.

        NOTE:
        - raw CSV는 data/raw에 저장된다.
        - processed CSV는 pretrain에서 생성된다.
        """
        csv_path = os.path.join(data_dir, f"{ticker}_{self.agent_id}_dataset.csv")

        if not rebuild and os.path.exists(csv_path):
            return

        print(f"[INFO] MacroAgent.searcher: {ticker} 매크로 데이터셋 CSV 생성 중...")

        # ------------------------------------------------------------------
        # 1) 기간 설정 - config.common_params["period"] 를 사용
        # ------------------------------------------------------------------
        from config.agents import common_params  # 순환 import 방지용 지역 import

        period = common_params.get("period", "2y")
        # yfinance 의 period 파라미터를 그대로 사용 (예: "2y", "1y")

        # ------------------------------------------------------------------
        # 2) 매크로 데이터 수집 (MACRO_TICKERS 기준)
        # ------------------------------------------------------------------
        try:
            df_macro = yf.download(
                tickers=list(MACRO_TICKERS.values()),
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] MacroAgent.searcher: 매크로 데이터 다운로드 실패: {e}")
            df_macro = pd.DataFrame()

        if isinstance(df_macro.columns, pd.MultiIndex):
            # (Date, 티커, OHLCV) -> (Date, Ticker_OHLCV)
            df_macro = df_macro.stack(level=0)
            df_macro.index.names = ["Date", "Ticker"]
            df_macro = df_macro.unstack(level="Ticker")
            df_macro.columns = [f"{col[1]}_{col[0]}" for col in df_macro.columns.values]
        else:
            df_macro.index.name = "Date"

        df_macro = df_macro.reset_index()
        if "Date" in df_macro.columns:
            df_macro["Date"] = pd.to_datetime(df_macro["Date"]).dt.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # 3) 매크로 파생 피처 생성 (1일 수익률, Yield_spread, Risk_Sentiment)
        #    - FINAL_FEATURES 와 일관되도록 이름을 맞춘다.
        # ------------------------------------------------------------------
        df_macro_feat = df_macro.copy()
        if "Date" in df_macro_feat.columns:
            df_macro_feat.set_index("Date", inplace=True)

        # 각 자산의 1일 수익률
        for t in MACRO_TICKERS.values():
            col_close = f"{t}_Close"
            if col_close in df_macro_feat.columns:
                df_macro_feat[f"{t}_ret_1d"] = df_macro_feat[col_close].pct_change()

        # 금리 스프레드
        if "^TNX_Close" in df_macro_feat.columns and "^IRX_Close" in df_macro_feat.columns:
            df_macro_feat["Yield_spread"] = df_macro_feat["^TNX_Close"] - df_macro_feat["^IRX_Close"]

        # 시장 위험심리
        if (
            "SPY_ret_1d" in df_macro_feat.columns
            and "DX-Y.NYB_ret_1d" in df_macro_feat.columns
            and "^VIX_ret_1d" in df_macro_feat.columns
        ):
            df_macro_feat["Risk_Sentiment"] = (
                df_macro_feat["SPY_ret_1d"] - df_macro_feat["DX-Y.NYB_ret_1d"] - df_macro_feat["^VIX_ret_1d"]
            )

        df_macro_feat = df_macro_feat.reset_index()

        # ------------------------------------------------------------------
        # 4) 개별 티커 주가 데이터 수집 (종가 기준) + 주가 피처(ret1, ma5, ma10)
        # ------------------------------------------------------------------
        try:
            df_price = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            # MultiIndex 컬럼(flatten)
            if isinstance(df_price.columns, pd.MultiIndex):
                df_price.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df_price.columns]

            df_price = df_price[["Close"]].rename(columns={"Close": ticker})
            df_price.index.name = "Date"
            df_price = df_price.reset_index()
            df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[WARN] MacroAgent.searcher: 종가 데이터 다운로드 실패({ticker}): {e}")
            df_price = pd.DataFrame(columns=["Date", ticker])

        # 주가 기반 파생 피처 (FINAL_FEATURES 의 stock 피처 이름과 일치하도록)
        if not df_price.empty:
            df_price["ret1"] = df_price[ticker].pct_change()
            df_price["ma5"] = df_price[ticker].rolling(5).mean()
            df_price["ma10"] = df_price[ticker].rolling(10).mean()
            df_price = df_price.fillna(method="bfill")

        # ------------------------------------------------------------------
        # 5) 매크로 + 주가 병합 후, FINAL_FEATURES 기준 피처셋 구성
        # ------------------------------------------------------------------
        if "Date" in df_macro_feat.columns and "Date" in df_price.columns:
            merged = pd.merge(df_price, df_macro_feat, on="Date", how="inner").sort_values("Date")
        else:
            print("[WARN] MacroAgent.searcher: 'Date' 컬럼 누락으로 병합 실패")
            merged = pd.DataFrame(columns=["Date"])

        merged = merged.fillna(method="ffill").fillna(method="bfill").dropna().reset_index(drop=True)

        # FINAL_FEATURES 기준으로 정렬/보정 (누락 피처는 0.0으로 채움)
        X_final = pd.DataFrame(index=merged.index)
        for feature in FINAL_FEATURES:
            if feature in merged.columns:
                X_final[feature] = merged[feature]
            else:
                X_final[feature] = 0.0

        # 개별 종목 종가 컬럼 준비 (마지막 컬럼으로 사용)
        # 형식 통일: 첫 컬럼 Date, 마지막 컬럼 Close
        if ticker in merged.columns:
            merged["Close"] = merged[ticker]
        elif "Close" not in merged.columns:
            # 종가 정보를 찾지 못한 경우 fallback (NaN)
            merged["Close"] = np.nan

        # Date + FINAL_FEATURES + Close 형태로 저장
        out_df = pd.concat(
            [
                merged[["Date"]].reset_index(drop=True),
                X_final.reset_index(drop=True),
                merged[["Close"]].reset_index(drop=True),
            ],
            axis=1,
        )

        # processed 경로
        os.makedirs(data_dir, exist_ok=True)
        out_df.to_csv(csv_path, index=False)

        # raw 경로에도 동일 내용 저장 (TechnicalAgent 패턴과 유사)
        base_root = os.path.dirname(data_dir)  # e.g. "data"
        raw_dir = os.path.join(base_root, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{ticker}_{self.agent_id}_raw.csv")
        out_df.to_csv(raw_path, index=False)

        print(f"[INFO] MacroAgent.searcher: 매크로 데이터셋 CSV 저장 완료 -> {csv_path}")

    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """
        MacroAgent 전용 searcher
        - MacroAData를 사용해 공통 CSV/가격 CSV를 준비
        - CSV 기반으로 전체 시퀀스를 만든 뒤, 최신 윈도우만 반환
        """
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError(f"{agent_id}: ticker가 지정되지 않았습니다.")

        self.ticker = ticker
        if ticker not in self.tickers:
            self.tickers = [ticker]

        # 모델/스케일러 경로 업데이트
        self.model_path = os.path.join(model_dir, f"{ticker}_{agent_id}.pt")
        self.scaler_X_path = os.path.join(model_dir, "scalers", f"{ticker}_{agent_id}_xscaler.pkl")
        self.scaler_y_path = os.path.join(model_dir, "scalers", f"{ticker}_{agent_id}_yscaler.pkl")

        # 1) Raw CSV 보장 (데이터 수집/전처리)
        self._ensure_macro_csv(ticker, rebuild=rebuild)
        
        # 2) Raw CSV에서 최신 window_size만큼 직접 추출
        raw_csv_path = os.path.join(os.path.dirname(self.data_dir), "raw", f"{ticker}_{self.agent_id}_raw.csv")
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
        
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        feature_cols = FINAL_FEATURES
        window_size = self.window
        
        # 피처 추출 (Date, Close 제외)
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        # 최신 window_size만큼 추출
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        self.X_raw = df_raw[feature_cols]
        X_latest = X_all[-window_size:].reshape(1, window_size, -1)  # (1, T, F)
        
        print(f"[OK] MacroAgent searcher - 최신 윈도우 준비 완료: {X_latest.shape}")

        # StockData 구성
        self.stockdata = StockData(ticker=ticker)
        try:
            self.stockdata.last_price = float(df_raw["Close"].iloc[-1])
            self.last_price = self.stockdata.last_price
        except Exception:
            self.stockdata.last_price = None

        # 통화코드
        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception:
            self.stockdata.currency = "USD"

        # feature_dict (마지막 윈도우)
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}

        setattr(self.stockdata, agent_id, feature_dict)
        self.stockdata.feature_cols = feature_cols

        return torch.FloatTensor(X_latest).to(self.device)

    def pretrain(self):
        """MacroAgent 사전학습 루틴 - data/raw CSV에서 직접 로드하여 scaling/window 처리"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        # Config
        cfg = agents_info.get(self.agent_id, {})
        epochs = cfg.get("epochs", 60)
        lr = cfg.get("learning_rate", 0.0005)
        batch_size = cfg.get("batch_size", 16)

        if not self.ticker:
            raise ValueError("MacroAgent.pretrain: ticker가 설정되지 않았습니다.")

        ticker = self.ticker

        # 1) data/raw CSV 로드
        raw_csv_path = os.path.join(os.path.dirname(self.data_dir), "raw", f"{ticker}_{self.agent_id}_raw.csv")
        if not os.path.exists(raw_csv_path):
            print(f"⚙️ {ticker} {self.agent_id} raw CSV not found. Running searcher() to generate it...")
            _ = self.searcher(ticker, rebuild=True)
            if not os.path.exists(raw_csv_path):
                raise FileNotFoundError(f"Raw CSV not found after searcher: {raw_csv_path}")
        
        # raw CSV 읽기 (Date 첫 컬럼, Close 마지막 컬럼)
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        # 2) 피처 컬럼 추출 (Date, Close 제외)
        feature_cols = FINAL_FEATURES
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        # 3) 타겟 생성 (다음날 수익률)
        close_prices = df_raw["Close"].values
        y_all = (close_prices[1:] / close_prices[:-1] - 1.0).reshape(-1, 1).astype(np.float32)
        X_all = X_all[:-1]  # 마지막 행 제외
        
        # 4) Window 처리 (시퀀스 생성)
        window_size = self.window
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        def _create_sequences(X, y, win: int):
            Xs, ys = [], []
            for i in range(len(X) - win):
                Xs.append(X[i : i + win])
                ys.append(y[i + win])
            return np.array(Xs), np.array(ys)
        
        X_seq, y_seq = _create_sequences(X_all, y_all, window_size)
        print(f"[INFO] 시퀀스 생성 완료: {X_seq.shape}, {y_seq.shape}")
        
        if len(X_seq) == 0:
            print("[WARN] MacroAgent.pretrain: 학습용 시퀀스가 없습니다.")
            return

        # 5) Scaling
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_seq.reshape(-1, X_seq.shape[-1]))
        X_scaled = X_scaled.reshape(X_seq.shape)
        
        minmax_range = cfg.get("minmax_scaler_range", (-1, 1))
        scaler_y = MinMaxScaler(feature_range=minmax_range)
        y_scaled = scaler_y.fit_transform(y_seq)
        
        # 스케일러 저장
        os.makedirs(os.path.dirname(self.scaler_X_path), exist_ok=True)
        scaler_X.feature_names_in_ = np.array(feature_cols)
        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        # 6) input_dim 자동 조정
        actual_input_dim = X_seq.shape[-1]
        if actual_input_dim != self.input_dim:
            print(f"[INFO] input_dim 조정: {self.input_dim} -> {actual_input_dim}")
            self.input_dim = actual_input_dim
            hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
            dropout_rates = cfg.get("dropout_rates", [0.3, 0.3, 0.2])

            self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
            self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
            self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
            self.fc1 = nn.Linear(hidden_dims[2], 32)
            self.fc2 = nn.Linear(32, self.output_dim)

        # 7) 학습 준비
        model = self
        model.to(self.device)
        model.train()

        dataset = TensorDataset(
            torch.FloatTensor(X_scaled).to(self.device),
            torch.FloatTensor(y_scaled).to(self.device),
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.L1Loss()

        # 5) 학습 루프
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")

        # 8) 모델 저장
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, self.model_path)
        self.model_loaded = True
        print(f"[OK] MacroAgent 학습 완료. Final Train Loss: {train_loss:.4f}")
        
        # 9) 전처리된 데이터 저장 (선택적)
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_id}_dataset.csv")
        flattened_data = []
        dates_list = df_raw["Date"].values[:-1]  # 마지막 제외
        
        for sample_idx in range(len(X_seq)):
            for time_idx in range(window_size):
                date_idx = sample_idx + time_idx
                row = {
                    'sample_id': sample_idx,
                    'time_step': time_idx,
                    'date': str(dates_list[date_idx]) if date_idx < len(dates_list) else None,
                    'target': float(y_scaled[sample_idx, 0]) if time_idx == window_size - 1 else np.nan,
                }
                for feat_idx, feat_name in enumerate(feature_cols):
                    row[feat_name] = float(X_scaled[sample_idx, time_idx, feat_idx])
                flattened_data.append(row)
        
        dataset_df = pd.DataFrame(flattened_data)
        os.makedirs(self.data_dir, exist_ok=True)
        dataset_df.to_csv(dataset_path, index=False)
        print(f"✅ 전처리된 데이터 저장 완료: {dataset_path}")

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치 로드 (Input Dim 자동 조정 포함)"""
        if model_path is None:
            model_path = self.model_path

        if not os.path.exists(model_path):
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            
            # 1. Input Dim 불일치 확인 및 레이어 재생성
            # lstm1.weight_ih_l0 shape: (4 * hidden_dim, input_dim)
            if "lstm1.weight_ih_l0" in state_dict:
                weight = state_dict["lstm1.weight_ih_l0"]
                saved_input_dim = weight.shape[1]
                
                if saved_input_dim != self.input_dim:
                    print(f"[INFO] 모델 로드 중 input_dim 조정: {self.input_dim} -> {saved_input_dim}")
                    self.input_dim = saved_input_dim
                    
                    # Config 재로드 (hidden_dims 등)
                    cfg = agents_info.get(self.agent_id, {})
                    hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
                    dropout_rates = cfg.get("dropout_rates", [0.3, 0.3, 0.2])
                    
                    # 레이어 재생성
                    self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
                    self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
                    self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
                    self.fc1 = nn.Linear(hidden_dims[2], 32)
                    self.fc2 = nn.Linear(32, self.output_dim)
                    
                    # GPU 이동
                    self.to(self.device)

            self.load_state_dict(state_dict, strict=False)
            self.eval()
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"[{self.agent_id}] load_model 실패: {e}")
            return False

    def predict(self, X, n_samples: Optional[int] = None, current_price: Optional[float] = None, X_last: Optional[np.ndarray] = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)
        """
        # n_samples 설정 (config에서 가져오기)
        if n_samples is None:
            n_samples = common_params.get("n_samples", 30)
        
        # ticker 확인
        if not self.ticker:
            raise ValueError("ticker가 설정되지 않았습니다. 먼저 searcher(ticker)를 호출하세요.")
        
        # 모델 파일 확인
        if not os.path.exists(self.model_path):
            print(f"[{self.agent_id}] 모델이 없어 pretrain()을 실행합니다...")
            self.pretrain()
        else:
            # 모델이 있으면 로드 (이미 로드되었는지 확인)
            if not hasattr(self, "model_loaded") or not self.model_loaded:
                self.load_model(self.model_path)
        
        # 모델 준비 및 스케일러 로드
        if not hasattr(self, "scaler_X") or self.scaler_X is None:
             if os.path.exists(self.scaler_X_path):
                 self.scaler_X = joblib.load(self.scaler_X_path)
                 self.scaler_y = joblib.load(self.scaler_y_path)
             else:
                 # 스케일러가 없으면 pretrain 실행
                 if not self.ticker:
                     raise ValueError("ticker가 설정되지 않았습니다. 먼저 searcher(ticker)를 호출하세요.")
                 print(f"[{self.agent_id}] 스케일러가 없어 pretrain()을 실행합니다...")
                 self.pretrain()
                 # pretrain 후 다시 로드
                 if os.path.exists(self.scaler_X_path):
                     self.scaler_X = joblib.load(self.scaler_X_path)
                     self.scaler_y = joblib.load(self.scaler_y_path)
                 else:
                     raise RuntimeError("pretrain() 후에도 스케일러가 생성되지 않았습니다.")

        # 입력 변환 및 스케일링 (StockData 지원)
        if isinstance(X, StockData):
            sd = X
            X_in = getattr(sd, "X_seq", None)
            if X_in is None:
                # StockData에 X_seq가 없으면 agent_id로 찾기
                X_in = getattr(sd, self.agent_id, None)
                if isinstance(X_in, dict):
                    # dict 형태면 DataFrame으로 변환
                    df = pd.DataFrame(X_in)
                    X_in = df.values
            if X_in is None:
                raise ValueError(f"StockData에 {self.agent_id} 데이터가 없습니다. searcher()를 먼저 호출하세요.")
            if current_price is None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
            X = X_in
        
        if isinstance(X, np.ndarray):
            X_np = X
        elif isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")
        
        # 형태 정규화: (1, T, F) or (T, F) → (T, F)
        if X_np.ndim == 3:
            X_2d = X_np[0]
        else:
            X_2d = X_np
            
        # 데이터 프레임으로 변환 (피처 이름 기준 transform)
        # 학습 시 사용한 feature_cols(= scaler_X.feature_names_in_)를 그대로 사용해야 함
        feature_names = getattr(self.scaler_X, "feature_names_in_", None)
        if feature_names is None:
            # fallback: 기존 FINAL_FEATURES 사용 (구버전 대비)
            feature_names = FINAL_FEATURES
        # 길이가 다를 경우(예: config/상수 변경 후 오래된 모델)에는
        # 현재 입력 X의 차원에 맞춰 앞에서부터 잘라 사용
        if len(feature_names) != X_2d.shape[1]:
            feature_names = list(feature_names)[: X_2d.shape[1]]
        X_df = pd.DataFrame(X_2d, columns=feature_names)
        X_scaled = self.scaler_X.transform(X_df)
        
        X_scaled_np = np.expand_dims(X_scaled, axis=0)
        X_tensor = torch.FloatTensor(X_scaled_np).to(self.device)

        # Monte Carlo Dropout 추론
        self.train() # Dropout 활성화
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = self(X_tensor).cpu().numpy()
                preds.append(y_pred)

        preds = np.stack(preds)  # (samples, batch, output_dim)
        mean_pred = preds.mean(axis=0)
        std_pred = np.abs(preds.std(axis=0))

        # 역변환
        pred_inv = self.scaler_y.inverse_transform(mean_pred)
        std_inv = self.scaler_y.inverse_transform(std_pred)

        sigma = float(std_inv[-1, 0]) if std_inv.ndim > 1 else float(std_inv[-1])
        sigma_min = common_params.get("sigma_min", 1e-6)
        sigma = max(sigma, sigma_min)
        confidence = 1 / (1 + np.log1p(sigma))

        # 가격 계산
        if current_price is None:
            default_price = common_params.get("default_current_price", 100.0)
            current_price = getattr(self.stockdata, 'last_price', None) or self.last_price or default_price

        predicted_return = float(pred_inv[-1, 0]) if pred_inv.ndim > 1 else float(pred_inv[-1])
        # config에서 수익률 클리핑 범위 가져오기
        cfg = agents_info.get(self.agent_id, {})
        return_clip_min = cfg.get("return_clip_min", -0.5)
        return_clip_max = cfg.get("return_clip_max", 0.5)
        predicted_return = np.clip(predicted_return, return_clip_min, return_clip_max)
        
        predicted_price = current_price * (1 + predicted_return)

        target = Target(
            next_close=float(predicted_price),
            uncertainty=sigma,
            confidence=float(confidence),
        )

        return target

    
    def reviewer_draft(self, stock_data: StockData = None, target: Target = None) -> Opinion:
        """(1) searcher → (2) predicter → (3) LLM(JSON Schema)로 reason 생성 → Opinion 반환"""

        # 1) 데이터 수집
        if stock_data is None:
            stock_data = self.stockdata
        
        if stock_data is None:
             # 데이터가 없으면 searcher 호출
             if not self.ticker:
                 raise ValueError("ticker가 설정되지 않았습니다.")
             self.searcher(self.ticker)
             stock_data = self.stockdata

        # 2) 예측값 생성
        if target is None:
            # 데이터가 없으면 searcher 호출
            if getattr(stock_data, self.agent_id, None) is None:
                X_input = self.searcher(self.ticker)
            else:
                # 이미 있으면 복원 (근데 텐서로 복원이 까다로우니 그냥 searcher 다시 부르는게 안전)
                X_input = self.searcher(self.ticker)
            target = self.predict(X_input)

        # 3) GradientAnalyzer를 사용한 해석
        # GradientAnalyzer 실행을 위해 스케일링된 입력 필요
        if self.X_raw is not None and self.scaler_X is not None:
            try:
                # Raw Data(최근 window) -> Scaling
                X_window = self.X_raw.tail(self.window)
                X_scaled = self.scaler_X.transform(X_window)
                X_scaled_np = np.expand_dims(X_scaled, axis=0).astype(np.float32)
                
                feature_names = list(self.scaler_X.feature_names_in_)
                # 너무 많은 피처는 상위 300개로 제한 (GradientAnalyzer 부담 경감)
                if X_scaled_np.shape[2] > 300:
                    X_scaled_np = X_scaled_np[:, :, :300]
                    feature_names = feature_names[:300]

                model = self if isinstance(self, nn.Module) else self.model
                gradient_analyzer = GradientAnalyzer(model, feature_names)
                importance_dict, temporal_df, consistency_df, sensitivity_df, grad_results = gradient_analyzer.run_all_gradients(X_scaled_np)
                
                # 결과 저장
                if stock_data:
                    feature_imp = {
                        'feature_summary': grad_results.get("feature_summary"),
                        'importance_dict': importance_dict,
                        'temporal_summary': temporal_df.head().to_dict(orient="records") if temporal_df is not None else [],
                        'consistency_summary': consistency_df.to_dict(orient="records") if consistency_df is not None else [],
                        'sensitivity_summary': sensitivity_df.to_dict(orient="records") if sensitivity_df is not None else [],
                        'stability_summary': grad_results.get("stability_summary")
                    }
                    
                    # stockdata.agent_id 딕셔너리에 업데이트
                    agent_data = getattr(stock_data, self.agent_id, {})
                    if isinstance(agent_data, dict):
                        agent_data['feature_importance'] = feature_imp
                        agent_data['our_prediction'] = target.next_close
                        agent_data['uncertainty'] = round(target.uncertainty or 0.0, 8)
                        agent_data['confidence'] = round(target.confidence or 0.0, 8)
                        setattr(stock_data, self.agent_id, agent_data)
            except Exception as e:
                print(f"[WARN] GradientAnalyzer 실행 실패: {e}")
        
        # 4) LLM 호출(reason 생성)
        sys_text, user_text = self._build_messages_opinion(stock_data, target)
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}
        )
        reason = parsed.get("reason", "(사유 생성 실패)")
        self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason))
        return self.opinions[-1]

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]}, "message": {"type": "string"}}, "required": ["stance", "message"], "additionalProperties": False}
        )
        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(실패)")
        )
        self.rebuttals[round].append(result)
        return result

    def reviewer_rebuttal(self, my_opinion, other_opinion, round_index):
        return self.reviewer_rebut(my_opinion, other_opinion, round_index)

    def reviewer_revise(self, my_opinion, others, rebuttals, stock_data, fine_tune=True, lr: Optional[float] = None, epochs: Optional[int] = None):
        # Fine-tuning 파라미터 설정 (config에서 가져오기)
        if lr is None:
            lr = common_params.get("fine_tune_lr", 1e-4)
        if epochs is None:
            epochs = agents_info.get(self.agent_id, {}).get("fine_tune_epochs", 5)
        return super().reviewer_revise(my_opinion, others, rebuttals, stock_data, fine_tune, lr, epochs)

    def _build_messages_opinion(self, stock_data, target):
        # ... (기존 로직 복원) ...
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
             # 데이터가 없으면 빈 딕셔너리로 처리 (에러 방지)
             agent_data = {}
             
        stock_data_dict = asdict(stock_data)
        feature_imp = agent_data.get("feature_importance", {})
        
        ctx = {
            "agent_id": self.agent_id,
            "ticker": stock_data_dict.get("ticker", "Unknown"),
            "currency": stock_data_dict.get("currency", "USD"),
            "last_price": stock_data_dict.get("last_price", None),
            "our_prediction": float(target.next_close),
            "uncertainty": float(target.uncertainty or 0.0),
            "confidence": float(target.confidence or 0.0),

            "feature_importance": {
                "feature_summary": feature_imp.get("feature_summary", []),
                "importance_dict": feature_imp.get("importance_dict", []),
                "temporal_summary": feature_imp.get("temporal_summary", []),
                'consistency_summary': feature_imp.get('consistency_summary', []),
                'sensitivity_summary': feature_imp.get('sensitivity_summary', []),
                'stability_summary': feature_imp.get('stability_summary', [])
            },
        }
        
        # feature_importance Top 5 요약 출력 (디버깅용)
        if 'importance_dict' in feature_imp and isinstance(feature_imp['importance_dict'], dict):
            importance_dict = feature_imp['importance_dict']
            try:
                numeric_items = [(k, v) for k, v in importance_dict.items() if isinstance(v, (int, float))]
                if numeric_items:
                    top5 = sorted(numeric_items, key=lambda x: abs(x[1]), reverse=True)[:5]
                    top5_str = ", ".join([f"{str(k)}={v:.2e}" for k, v in top5])
                    print(f"  |  [INFO] Top 5 features: {top5_str}")
            except Exception:
                pass

        # 시계열 데이터 포함 (config에서 일수 가져오기)
        cfg = agents_info.get(self.agent_id, {})
        recent_days = cfg.get("recent_days", 14)
        # agent_data에 저장된 리스트들 중, feature_imp가 아닌 실제 시계열 데이터만 추출
        for col, values in agent_data.items():
            if col == 'feature_importance': continue
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-recent_days:]  # 최근 N일치
            else:
                ctx[col] = [values]
        
        system_text = OPINION_PROMPTS[self.agent_id]["system"]
        user_text = OPINION_PROMPTS[self.agent_id]["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    def _build_messages_rebuttal(self, my_opinion, target_opinion, stock_data):
        t = getattr(stock_data, "ticker", "UNKNOWN")
        ccy = getattr(stock_data, "currency", "USD")
        
        # 상세 로직은 복잡하므로 핵심만 전달하도록 구성
        ctx = {
            "ticker": t,
            "currency": ccy,
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": my_opinion.reason
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": target_opinion.reason
            }
        }
        
        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        user_text = REBUTTAL_PROMPTS[self.agent_id]["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    def _build_messages_revision(self, my_opinion, others, rebuttals, stock_data):
        # ... (간단 복원) ...
        t = getattr(stock_data, "ticker", "UNKNOWN")
        others_summary = [{"agent": o.agent_id, "price": o.target.next_close, "reason": o.reason} for o in others]
        
        ctx = {
            "ticker": t,
            "my_opinion": {"price": my_opinion.target.next_close, "reason": my_opinion.reason},
            "others": others_summary,
            "rebuttals": [r.message for r in (rebuttals or [])]
        }
        
        system_text = REVISION_PROMPTS[self.agent_id]["system"]
        user_text = REVISION_PROMPTS[self.agent_id]["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text
