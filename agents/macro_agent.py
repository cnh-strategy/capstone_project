# ======================================================================
# PART 1 — MacroLSTM + MacroAgent Base + Data Fetch (Merged)
# ======================================================================
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from agents.base_agent import StockData
from core.macro_classes.macro_analysis_reviewer import MacroAnalysisReviewer
from config.agents import dir_info


# ======================================================================
# LSTM MODEL (원본 그대로 유지)
# ======================================================================
class MacroLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32],
                 output_dim=1, dropout_rates=[0.3, 0.3, 0.2]):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.drop1 = nn.Dropout(dropout_rates[0])

        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.drop2 = nn.Dropout(dropout_rates[1])

        self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
        self.drop3 = nn.Dropout(dropout_rates[2])

        self.fc1 = nn.Linear(hidden_dims[2], 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        h1, _ = self.lstm1(x)
        h1 = self.drop1(h1)
        h2, _ = self.lstm2(h1)
        h2 = self.drop2(h2)
        h3, _ = self.lstm3(h2)
        h3 = self.drop3(h3)

        h_last = h3[:, -1, :]
        out = torch.relu(self.fc1(h_last))
        return self.fc2(out)


# ======================================================================
# MacroAgent — FULL RESTORED unified version
# ======================================================================
class MacroAgent:
    """
    Full Restored MacroAgent =
    - pretrain()의 feature engineering 로직을 그대로 사용
    - searcher()도 동일한 피처 구조/스케일러 기반
    - MakeDatasetMacro 완전 제거
    """

    def __init__(self, agent_id="MacroAgent", ticker="NVDA", window=40):
        self.agent_id = agent_id
        self.ticker = ticker.upper()
        self.window = window

        # 저장 위치
        self.model_dir = dir_info["model_dir"]
        self.scaler_dir = f"{self.model_dir}/scalers"

        self.model_path = f"{self.model_dir}/{self.ticker}_{self.agent_id}.pt"
        self.scaler_X_path = f"{self.scaler_dir}/{self.ticker}_{self.agent_id}_xscaler.pkl"
        self.scaler_y_path = f"{self.scaler_dir}/{self.ticker}_{self.agent_id}_yscaler.pkl"
        self.feature_json_path = f"{self.scaler_dir}/{self.ticker}_{self.agent_id}_features.json"

        # pretrain() 에서 채워짐
        self.scaler_X = None
        self.scaler_y = None
        self.feature_order = None
        self.model = None

        # 5년치 데이터
        self.start_date = datetime.now() - timedelta(days=5 * 365)
        self.end_date = datetime.now()

        # full macro tickers
        self.macro_tickers = {
            "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
            "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
            "^VIX": "^VIX",
            "DX-Y.NYB": "DX-Y.NYB",
            "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
            "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F",
        }

        # DebateAgent에서 사용하는 데이터 구조
        self.stockdata = StockData(
            ticker=self.ticker,
            last_price=None,
            currency="USD"
        )

    # ==================================================================
    # 1) 매크로 데이터 수집
    # ==================================================================
    def fetch_macro_data(self):
        print(f"[FETCH] Macro data...")

        df = yf.download(
            tickers=list(self.macro_tickers.values()),
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=0)
            df.index.names = ["Date", "Ticker"]
            df = df.unstack(level="Ticker")
            df.columns = ["_".join(col).strip() for col in df.columns.values]

        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.ffill().bfill()

        return df

    # ==================================================================
    # 2) 개별 종목 데이터 수집 (Close 1-D 강제)
    # ==================================================================
    def fetch_price_data(self):
        print(f"[FETCH] Price data for {self.ticker}")

        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        # MultiIndex → 단순화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

        # Close 강제 변환
        close_col = None
        if "Close" in df.columns:
            close_col = "Close"
        else:
            candidates = [c for c in df.columns if "close" in c.lower()]
            if candidates:
                close_col = candidates[0]
            else:
                raise ValueError(f"Close column not found for {self.ticker}")

        close_series = df[close_col]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        close_series = close_series.astype(float)
        close_series.index = pd.to_datetime(close_series.index)

        return pd.DataFrame({
            "Date": close_series.index,
            self.ticker: close_series.values
        })

    # ==================================================================
    # 3) 매크로 + 가격 데이터 병합
    # ==================================================================
    def merge_data(self):
        df_macro = self.fetch_macro_data()
        df_price = self.fetch_price_data()

        merged = pd.merge(df_price, df_macro, on="Date", how="inner")
        merged = merged.sort_values("Date").reset_index(drop=True)
        merged = merged.ffill().bfill()

        print(f"[MERGE] merged: {merged.shape}")
        return merged


    # ==================================================================
    # 4) Feature Engineering (pretrain()와 searcher()가 동일 구조로 사용)
    # ==================================================================
    def feature_engineering(self, df):
        df = df.copy()

        # ----------------------------------------------
        # ① 매크로 ret1
        # ----------------------------------------------
        for macro in self.macro_tickers.values():
            close_col = f"Close_{macro}" if f"Close_{macro}" in df.columns else f"{macro}_Close"
            if close_col in df.columns:
                df[f"{macro}_ret1"] = df[close_col].pct_change()

        # ----------------------------------------------
        # ② 금리 스프레드
        # ----------------------------------------------
        if "^TNX_Close" in df.columns and "^IRX_Close" in df.columns:
            df["Yield_spread"] = df["^TNX_Close"] - df["^IRX_Close"]

        # ----------------------------------------------
        # ③ 위험심리
        # ----------------------------------------------
        if "SPY_ret1" in df.columns and "DX-Y.NYB_ret1" in df.columns and "^VIX_ret1" in df.columns:
            df["Risk_Sentiment"] = df["SPY_ret1"] - df["DX-Y.NYB_ret1"] - df["^VIX_ret1"]

        # ----------------------------------------------
        # ④ 개별 종목 ret1, ma5, ma10
        # ----------------------------------------------
        ticker = self.ticker
        if ticker in df.columns:
            df[f"{ticker}_ret1"] = df[ticker].pct_change()
            df[f"{ticker}_ma5"] = df[ticker].rolling(5).mean()
            df[f"{ticker}_ma10"] = df[ticker].rolling(10).mean()

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        # ----------------------------------------------
        # ⑤ constant 삭제
        # ----------------------------------------------
        constant_cols = [c for c in df.columns if df[c].std() == 0]
        if constant_cols:
            df = df.drop(columns=constant_cols)

        # ----------------------------------------------
        # ⑥ Volume 제거 규칙
        # ----------------------------------------------
        remove_patterns = [
            "Volume_^FVX", "Volume_^IRX", "Volume_^TNX", "Volume_^VIX",
            "Volume_DX-Y.NYB", "Volume_EURUSD=X", "Volume_USDJPY=X"
        ]
        df = df.drop(columns=[c for c in df.columns if any(p in c for p in remove_patterns)], errors="ignore")

        df = df.ffill().bfill()

        return df

    # ==================================================================
    # 5) Trainset 생성
    # ==================================================================
    def build_trainset(self, df):
        ticker = self.ticker

        # target 생성 — (1일 후 수익률)
        df["target"] = df[ticker].pct_change().shift(-1)
        df = df.dropna().reset_index(drop=True)

        # feature 컬럼 (Date, price, target 제외)
        feature_cols = [c for c in df.columns if c not in ["Date", ticker, "target"]]
        X = df[feature_cols]
        y = df[["target"]]

        # 스케일러 없으면 새로 생성
        if self.scaler_X is None:
            self.scaler_X = StandardScaler().fit(X)
        if self.scaler_y is None:
            self.scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y)

        # 스케일링
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)

        # 시퀀스 생성
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.window)

        # 저장
        os.makedirs(os.path.dirname(self.scaler_X_path), exist_ok=True)
        joblib.dump(self.scaler_X, self.scaler_X_path)
        joblib.dump(self.scaler_y, self.scaler_y_path)

        # feature order 저장 (예측 때 반드시 필요)
        with open(self.feature_json_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        self.feature_order = feature_cols

        print(f"[TRAINSET] X_seq={X_seq.shape}, y_seq={y_seq.shape}")
        return X_seq, y_seq, feature_cols

    # ==================================================================
    # 6) LSTM sequence builder
    # ==================================================================
    def create_sequences(self, X, y, window):
        Xs, ys = [], []
        for i in range(len(X) - window):
            Xs.append(X[i:i + window])
            ys.append(y[i + window])
        return np.array(Xs), np.array(ys)

    # ==================================================================
    # 7) PRETRAIN — 완전 복원
    # ==================================================================
    def pretrain(self):
        print(f"\n[PRETRAIN] MacroAgent Training Start — {self.ticker}")

        # ----------------------------------------------
        # 데이터 통합
        # ----------------------------------------------
        df = self.merge_data()
        df = self.feature_engineering(df)

        # ----------------------------------------------
        # trainset
        # ----------------------------------------------
        X_seq, y_seq, feature_cols = self.build_trainset(df)

        # ----------------------------------------------
        # train/test split
        # ----------------------------------------------
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        # ----------------------------------------------
        # 모델 준비
        # ----------------------------------------------
        input_dim = X_train.shape[2]
        model = MacroLSTM(input_dim=input_dim, output_dim=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.L1Loss()

        # tensor
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

        # ----------------------------------------------
        # Train loop (early stopping 포함)
        # ----------------------------------------------
        best_val = float("inf")
        patience = 10
        counter = 0

        for epoch in range(60):
            model.train()
            optimizer.zero_grad()

            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_test), y_test).item()

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"[EARLY STOP] epoch={epoch}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train={loss.item():.6f}, val={val_loss:.6f}")

        # ----------------------------------------------
        # 모델 저장
        # ----------------------------------------------
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, self.model_path)

        self.model = model
        print(f"[PRETRAIN COMPLETE] Model saved at {self.model_path}")


    # ==================================================================
    # 8) searcher() — 예측 입력 생성 (train과 동일 피처 구조)
    # ==================================================================
    def searcher(self, ticker=None, rebuild=False):
        print(f"[SEARCH] Preparing prediction input for {self.ticker}")

        if ticker is not None:
            self.ticker = ticker

        # feature_order 로드
        if not os.path.exists(self.feature_json_path):
            raise FileNotFoundError("Feature JSON not found — 먼저 pretrain()을 실행하세요.")

        with open(self.feature_json_path, "r", encoding="utf-8") as f:
            self.feature_order = json.load(f)

        # 최신 데이터 수집
        df = self.merge_data()
        df = self.feature_engineering(df)

        # 누락 피처 → 0 채움
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0

        df = df[self.feature_order]

        # window 만큼 tail
        X = df.tail(self.window).values

        # 스케일러 로드
        if self.scaler_X is None:
            self.scaler_X = joblib.load(self.scaler_X_path)

        X_scaled = self.scaler_X.transform(X)
        X_scaled = X_scaled.reshape(1, self.window, -1)

        return X_scaled

    # ==================================================================
    # 9) predict() — Monte Carlo Dropout 예측
    # ==================================================================
    def predict(self, X):
        print(f"[PREDICT] Inference for {self.ticker}")

        # 모델 로드
        if self.model is None:
            model = MacroLSTM(input_dim=X.shape[2], output_dim=1)
            state = torch.load(self.model_path, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            self.model = model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        X_tensor = torch.FloatTensor(X).to(device)

        # Monte-Carlo Dropout
        preds = []
        self.model.train()

        with torch.no_grad():
            for _ in range(30):
                out = self.model(X_tensor).cpu().numpy().flatten()
                preds.append(out)

        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        # y scaler 로드
        if self.scaler_y is None:
            self.scaler_y = joblib.load(self.scaler_y_path)

        # inverse scaling
        try:
            restored = self.scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
            mean_pred = restored
        except Exception:
            pass

        sigma = float(std_pred[-1])
        sigma = max(sigma, 1e-6)
        confidence = 1 / (1 + np.log1p(sigma))

        return {
            "pred_return": float(mean_pred[-1]),
            "uncertainty": sigma,
            "confidence": confidence,
        }

    # ==================================================================
    # 10) Reviewer 통합 (DebateAgent 100% 호환)
    # ==================================================================
    def __post_init_reviewer(self):
        """Reviewer 객체를 predictor와 연결"""
        self.reviewer = MacroAnalysisReviewer(predictor=self)

    # DebateAgent 가 reviewer_draft() 호출함
    def reviewer_draft(self, stock_data=None, target=None):
        if not hasattr(self, "reviewer"):
            self.__post_init_reviewer()
        return self.reviewer.reviewer_draft(stock_data=stock_data, target=target)

    def reviewer_rebut(self, my_opinion, other_opinion, round: int):
        if not hasattr(self, "reviewer"):
            self.__post_init_reviewer()
        return self.reviewer.reviewer_rebut(my_opinion, other_opinion, round)

    # DebateAgent 내부 네이밍 맞춰주기
    def reviewer_rebuttal(self, my_opinion, other_opinion, round_index: int):
        if not hasattr(self, "reviewer"):
            self.__post_init_reviewer()
        return self.reviewer.reviewer_rebut(my_opinion, other_opinion, round_index)

    def reviewer_revise(self, my_opinion, others, rebuttals, stock_data,
                        fine_tune=True, lr=1e-4, epochs=5):
        if not hasattr(self, "reviewer"):
            self.__post_init_reviewer()
        return self.reviewer.reviewer_revise(
            my_opinion=my_opinion,
            others=others,
            rebuttals=rebuttals,
            stock_data=stock_data,
            fine_tune=fine_tune,
            lr=lr,
            epochs=epochs
        )
