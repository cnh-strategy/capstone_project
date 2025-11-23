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

from config.agents import dir_info, agents_info
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
    # 메인 메서드
    # =========================================================================
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """MacroAgent 전용 searcher - 통합된 로직 사용"""
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError(f"{agent_id}: ticker가 지정되지 않았습니다.")

        self.ticker = ticker
        if ticker not in self.tickers:
            self.tickers = [ticker]

        # 모델 경로 업데이트
        self.model_path = os.path.join(model_dir, f"{ticker}_{agent_id}.pt")
        self.scaler_X_path = os.path.join(model_dir, "scalers", f"{ticker}_{agent_id}_xscaler.pkl")
        self.scaler_y_path = os.path.join(model_dir, "scalers", f"{ticker}_{agent_id}_yscaler.pkl")

        # 날짜 설정 (충분히 길게 잡아서 window 확보)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.window * 2 + 100)

        print("[INFO] MacroAgent 데이터 수집 및 처리 중 (Unified)...")
        
        # 1. 데이터 수집
        macro_df = self._fetch_macro_data(start_date, end_date)
        stock_df = self._fetch_stock_data(ticker, start_date, end_date)
        
        # 2. 파생변수 추가
        macro_df = self._add_derived_features(macro_df)
        
        # 3. 최종 데이터셋 구성
        X_input, merged = self._prepare_final_dataset(macro_df, stock_df, ticker)
        
        if len(X_input) < self.window:
            # 데이터가 너무 적으면 에러 대신 dummy 데이터로 시도 (테스트용)
            if len(X_input) == 0:
                 raise ValueError(f"데이터가 부족합니다. ({len(X_input)} < {self.window})")
            # pad with first row
            pad_len = self.window - len(X_input)
            first_row = X_input.iloc[[0]]
            X_pad = pd.concat([first_row] * pad_len + [X_input])
            X_input = X_pad
            
        # 4. Tensor 변환 (Raw Data)
        X_seq_np = np.expand_dims(X_input.tail(self.window).values, axis=0)
        X_seq = torch.FloatTensor(X_seq_np).to(self.device)
        
        print(f"[OK] 데이터 준비 완료: {X_seq.shape}")
        
        # 데이터 저장
        self.X_raw = X_input.copy()
        self.macro_df = merged.copy()
        
        # StockData 구성
        self.stockdata = StockData(ticker=ticker)
        
        # last_price
        try:
            if not stock_df.empty and f"Close_{ticker}" in stock_df.columns:
                self.stockdata.last_price = float(stock_df[f"Close_{ticker}"].iloc[-1])
                self.last_price = self.stockdata.last_price
            else:
                self.stockdata.last_price = None
        except Exception:
            self.stockdata.last_price = None
            
        # 통화코드
        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception:
            self.stockdata.currency = "USD"

        # feature_dict
        df_latest = pd.DataFrame(X_input.tail(self.window).values, columns=FINAL_FEATURES)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        
        # StockData 업데이트 (피처 및 메타데이터)
        setattr(self.stockdata, agent_id, feature_dict)
        self.stockdata.feature_cols = FINAL_FEATURES

        return X_seq

    def pretrain(self):
        """Agent별 사전학습 루틴 (통합 로직)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        # Config
        cfg = agents_info.get(self.agent_id, {})
        epochs = cfg.get("epochs", 60)
        lr = cfg.get("learning_rate", 0.0005)
        batch_size = cfg.get("batch_size", 16)
        
        # 1. 데이터 수집 (최대 5년치)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365*5)
        
        print("[INFO] 학습 데이터 수집 중...")
        macro_df = self._fetch_macro_data(start_date, end_date)
        stock_df = self._fetch_stock_data(self.ticker, start_date, end_date)
        macro_df = self._add_derived_features(macro_df)
        X_raw, merged = self._prepare_final_dataset(macro_df, stock_df, self.ticker)
        
        # 2. 타깃 생성 (다음날 수익률)
        close_col = f"Close_{self.ticker}"
        if close_col not in merged.columns:
             # 대체 컬럼 찾기
             cols = [c for c in merged.columns if "Close" in c and self.ticker in c]
             if cols:
                 close_col = cols[0]
             else:
                 raise ValueError(f"Target column {close_col} not found.")
             
        y_raw = merged[close_col].pct_change().shift(-1) # 다음날 수익률
        
        # 유효 데이터 필터링
        valid_idx = ~y_raw.isna()
        X_data = X_raw[valid_idx]
        y_data = y_raw[valid_idx]
        
        if len(X_data) < self.window:
             print("[WARN] 학습 데이터 부족으로 중단")
             return
             
        # 3. 스케일링
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        X_scaled = scaler_X.fit_transform(X_data)
        y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1, 1))
        
        # 스케일러 저장
        os.makedirs(os.path.dirname(self.scaler_X_path), exist_ok=True)
        scaler_X.feature_names_in_ = np.array(FINAL_FEATURES) # 명시적 지정
        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        # 4. 시퀀싱
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.window):
            X_seq.append(X_scaled[i : i + self.window])
            y_seq.append(y_scaled[i + self.window])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 5. 학습 준비
        # input_dim이 실제 데이터와 다를 경우 레이어 재생성
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
            
        # 모델 = self
        model = self
        model.to(self.device)
        model.train()
        
        # 데이터셋 구성
        dataset = TensorDataset(torch.FloatTensor(X_seq).to(self.device), 
                                torch.FloatTensor(y_seq).to(self.device))
        
        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.L1Loss() # or HuberLoss
        
        # 6. 학습 루프
        best_loss = float('inf')
        patience_cnt = 0
        patience = 10
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    pred = model(bx)
                    loss = loss_fn(pred, by)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
                
            if val_loss < best_loss:
                best_loss = val_loss
                patience_cnt = 0
                # 모델 저장
                torch.save({"model_state_dict": model.state_dict()}, self.model_path)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        print(f"[OK] 학습 완료. Best Val Loss: {best_loss:.4f}")
        # 학습 후 최고 모델 로드
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
            return True
        except Exception as e:
            print(f"[{self.agent_id}] load_model 실패: {e}")
            return False

    def predict(self, X, n_samples: int = 30, current_price: float = None, X_last: np.ndarray = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)
        """
        # 모델 준비 및 스케일러 로드
        if not hasattr(self, "scaler_X") or self.scaler_X is None:
             if os.path.exists(self.scaler_X_path):
                 self.scaler_X = joblib.load(self.scaler_X_path)
                 self.scaler_y = joblib.load(self.scaler_y_path)
             else:
                 raise RuntimeError("스케일러가 없습니다. pretrain()을 먼저 실행하세요.")

        # 입력 변환 및 스케일링
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
        # (X는 FINAL_FEATURES 순서로 들어왔다고 가정)
        X_df = pd.DataFrame(X_2d, columns=FINAL_FEATURES)
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
        sigma = max(sigma, 1e-6)
        confidence = 1 / (1 + np.log1p(sigma))

        # 가격 계산
        if current_price is None:
            current_price = getattr(self.stockdata, 'last_price', None) or self.last_price or 100.0

        predicted_return = float(pred_inv[-1, 0]) if pred_inv.ndim > 1 else float(pred_inv[-1])
        predicted_return = np.clip(predicted_return, -0.5, 0.5)
        
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

    def reviewer_revise(self, my_opinion, others, rebuttals, stock_data, fine_tune=True, lr=1e-4, epochs=5):
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

        # 시계열 데이터 포함 (최근 14일)
        # agent_data에 저장된 리스트들 중, feature_imp가 아닌 실제 시계열 데이터만 추출
        for col, values in agent_data.items():
            if col == 'feature_importance': continue
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-14:]  # 최근 14일치
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
