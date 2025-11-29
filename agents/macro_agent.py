import os
import json
from dataclasses import asdict
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from datetime import datetime

from torch.utils.data import TensorDataset, DataLoader

from config.agents import dir_info, agents_info
from core.macro_classes.macro_class_dataset import MacroAData
from core.macro_classes.macro_sub import MakeDatasetMacro
from core.macro_classes.macro_llm import GradientAnalyzer

from agents.base_agent import BaseAgent, Target, StockData, Opinion, Rebuttal
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS


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
        self.input_dim = cfg.get("input_dim", 13)  # 기본값, pretrain에서 실제 값으로 업데이트
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
        self.pred_df = None
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


    # -------------------------------------------------------------
    def load_assets(self):
        """모델 및 스케일러 로드 (TechnicalAgent 패턴)"""
        print("[INFO] 모델 및 스케일러 로드 중...")
        print(f"model_path: {self.model_path}")

        # 스케일러 로드
        self.scaler_X = joblib.load(self.scaler_X_path)
        self.scaler_y = joblib.load(self.scaler_y_path)

        # input_dim이 실제 데이터와 다를 경우 레이어 재생성
        actual_input_dim = len(self.scaler_X.feature_names_in_)
        if actual_input_dim != self.input_dim:
            print(f"[INFO] input_dim 불일치 감지: {self.input_dim} -> {actual_input_dim}, 레이어 재생성")
            self.input_dim = actual_input_dim
            self.output_dim = len(self.tickers)
            cfg = agents_info.get(self.agent_id, {})
            hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
            dropout_rates = cfg.get("dropout_rates", [0.3, 0.3, 0.2])

            self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
            self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
            self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
            self.drop1 = nn.Dropout(dropout_rates[0])
            self.drop2 = nn.Dropout(dropout_rates[1])
            self.drop3 = nn.Dropout(dropout_rates[2])
            self.fc1 = nn.Linear(hidden_dims[2], 32)
            self.fc2 = nn.Linear(32, self.output_dim)

        # 모델 가중치 로드
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.to(self.device)
            self.eval()
        else:
            print(f"[WARN] 모델 파일이 없습니다: {self.model_path}")

        print("[OK] 모델 및 스케일러 로드 완료")


    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """MacroAgent 전용 searcher - TechnicalAgent 패턴과 유사하게 파일 기반 독립성 확보"""
        import yfinance as yf

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

        # 모델 및 스케일러 로드 (파일 기반)
        if os.path.exists(self.scaler_X_path) and os.path.exists(self.scaler_y_path):
            self.load_assets()
        else:
            print(f"[{agent_id}] 스케일러가 없습니다 → pretrain() 자동 실행합니다.")
            self.pretrain()
            self.load_assets()

        # 매크로 데이터 수집
        print("[INFO] MacroAgent 데이터 수집 중...")
        macro_agent = MakeDatasetMacro(base_date=self.base_date,
                                       window=self.window, target_tickers=self.tickers)
        macro_agent.fetch_data()
        macro_agent.add_features()
        df = macro_agent.data.reset_index()
        self.macro_df = df.tail(self.window + 5)
        print(f"[OK] 매크로 데이터 수집 완료: {self.macro_df.shape}")

        # 피처 정리 및 스케일링
        print("[INFO] 피처 정리 및 스케일링 중...")
        macro_full = self.macro_df.copy()
        feature_cols = [c for c in macro_full.columns if c != "Date"]
        X_input = macro_full[feature_cols]

        expected_features = list(self.scaler_X.feature_names_in_)

        # 누락된 피처는 0으로 채움
        for col in expected_features:
            if col not in X_input.columns:
                X_input[col] = 0

        # 불필요한 피처 제거
        X_input = X_input[expected_features]

        print(f"[Check] 입력 피처 수: {X_input.shape[1]} / 스케일러 기준 피처 수: {len(expected_features)}")

        X_scaled = self.scaler_X.transform(X_input)
        X_scaled = pd.DataFrame(X_scaled, columns=expected_features)

        if len(X_scaled) < self.window:
            raise ValueError(f"데이터가 {self.window}일보다 적습니다.")

        X_seq_np = np.expand_dims(X_scaled.tail(self.window).values, axis=0)
        X_seq = torch.FloatTensor(X_seq_np).to(self.device)
        print("[OK] 스케일링 및 시퀀스 변환 완료")
        self.X_scaled = X_scaled

        # StockData 구성
        self.stockdata = StockData(ticker=ticker)

        # last_price 안전 변환
        try:
            data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
            if data is not None and not data.empty:
                last_val = data["Close"].iloc[-1]
                self.stockdata.last_price = float(last_val.item() if hasattr(last_val, "item") else last_val)
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

        # feature_dict 구성
        df_latest = pd.DataFrame(X_scaled.tail(self.window).values, columns=expected_features)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)

        # feature_cols도 expected_features로 업데이트
        self.stockdata.feature_cols = expected_features

        # StockData 생성 완료 (로그는 DebateAgent에서 처리)

        return X_seq


    def prepare_features(self):
        """하위 호환성 유지용"""
        print("[INFO] 피처 정리 및 스케일링 중...")

        macro_full = self.macro_df.copy()
        feature_cols = [c for c in macro_full.columns if c != "Date"]
        X_input = macro_full[feature_cols]

        expected_features = list(self.scaler_X.feature_names_in_)

        # 누락된 피처는 0으로 채움
        for col in expected_features:
            if col not in X_input.columns:
                X_input[col] = 0

        # 불필요한 피처 제거
        X_input = X_input[expected_features]

        print(f"[Check] 입력 피처 수: {X_input.shape[1]} / 스케일러 기준 피처 수: {len(expected_features)}")

        X_scaled = self.scaler_X.transform(X_input)
        X_scaled = pd.DataFrame(X_scaled, columns=expected_features)

        if len(X_scaled) < self.window:
            raise ValueError(f"데이터가 {self.window}일보다 적습니다.")

        X_seq_np = np.expand_dims(X_scaled.tail(self.window).values, axis=0)
        X_seq = torch.FloatTensor(X_seq_np).to(self.device)
        print("[OK] 스케일링 및 시퀀스 변환 완료")
        self.X_scaled = X_scaled
        return X_seq, X_scaled

    def pretrain(self):
        """Agent별 사전학습 루틴 (모델 생성, 학습, 저장, self.model 연결까지 포함)"""

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        # Config에서 하이퍼파라미터 가져오기
        cfg = agents_info.get(self.agent_id, {})
        epochs = cfg.get("epochs", 60)
        lr = cfg.get("learning_rate", 0.0005)
        batch_size = cfg.get("batch_size", 16)
        patience = cfg.get("patience", 10)
        loss_fn_name = cfg.get("loss_fn", "L1Loss")

        # Loss 함수 선택
        if loss_fn_name == "L1Loss":
            loss_fn = nn.L1Loss()
        elif loss_fn_name == "HuberLoss":
            loss_fn = nn.HuberLoss(delta=1.0)
        else:
            loss_fn = nn.L1Loss()  # 기본값

        # MacroAData를 사용하여 데이터 준비
        macro_data_agent = MacroAData(ticker=self.ticker)
        macro_data_agent.fetch_data()
        macro_data_agent.add_features()
        macro_data_agent.save_csv()
        macro_data_agent.make_close_price()
        macro_data_agent.model_maker()  # 학습 전체 파이프라인 실행

        # 스케일러 및 데이터 가져오기
        scaler_X = macro_data_agent.scaler_X if hasattr(macro_data_agent, 'scaler_X') else None
        scaler_y = macro_data_agent.scaler_y if hasattr(macro_data_agent, 'scaler_y') else None

        if scaler_X is None or scaler_y is None:
            raise RuntimeError(f"[{self.agent_id}] model_maker() 실행 후에도 스케일러가 생성되지 않았습니다.")

        # model_maker()에서 생성된 데이터 사용
        X_train = macro_data_agent.X_train
        y_train = macro_data_agent.y_train
        X_test = macro_data_agent.X_test
        y_test = macro_data_agent.y_test



        # input_dim이 실제 데이터와 다를 경우 레이어 재생성
        actual_input_dim = X_train.shape[-1]
        if actual_input_dim != self.input_dim:
            print(f"[INFO] input_dim 불일치 감지: {self.input_dim} -> {actual_input_dim}, 레이어 재생성")
            self.input_dim = actual_input_dim
            self.output_dim = len(self.tickers)
            hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
            dropout_rates = cfg.get("dropout_rates", [0.3, 0.3, 0.2])

            self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
            self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
            self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
            self.drop1 = nn.Dropout(dropout_rates[0])
            self.drop2 = nn.Dropout(dropout_rates[1])
            self.drop3 = nn.Dropout(dropout_rates[2])
            self.fc1 = nn.Linear(hidden_dims[2], 32)
            self.fc2 = nn.Linear(32, self.output_dim)

        # 스케일러 저장
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        os.makedirs(os.path.dirname(self.scaler_X_path), exist_ok=True)
        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)

        # 학습
        model = self

        model.train()
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        # Validation split
        val_size = int(len(X_train_tensor) * 0.1)
        X_val_tensor = X_train_tensor[-val_size:]
        y_val_tensor = y_train_tensor[-val_size:]
        X_train_tensor = X_train_tensor[:-val_size]
        y_train_tensor = y_train_tensor[:-val_size]

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # 모델 저장 및 연결
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        # 모델 경로 업데이트
        self.model_path = model_path

        # nn.Module 자기 자신이면 self.model에 등록하지 않음
        if model is not self:
            self.model = model

        print(f" {self.agent_id} 모델 학습 및 저장 완료: {model_path}")

    def predict(self, X, n_samples: int = 30, current_price: float = None, X_last: np.ndarray = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)
        TechnicalAgent 패턴
        """
        # 모델 준비 및 스케일러 로드
        # 과거 자기참조(child) 정리 - RecursionError 방지
        if isinstance(self, nn.Module):
            for name, child in list(getattr(self, "_modules", {}).items()):
                if child is self:
                    del self._modules[name]
            if getattr(self, "model", None) is self:
                self.model = None

        # 이 에이전트가 nn.Module이면 그 자체 사용
        if isinstance(self, nn.Module) and hasattr(self, "forward"):
            model = self
            # 모델 가중치가 로드되지 않았다면 로드 시도
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    self.to(self.device)
                except Exception as e:
                    print(f"[WARN] 모델 로드 실패: {e}, pretrain 수행...")
                    self.pretrain()
            else:
                self.pretrain()
        else:
            if self.model is None or not hasattr(self.model, "parameters"):
                model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
                if os.path.exists(model_path):
                    self.load_model(model_path)
                else:
                    self.pretrain()
            if self.model is None:
                raise RuntimeError(f"{self.agent_id} 모델이 초기화되지 않음")
            model = self.model

        # 스케일러 로드
        if self.scaler_X is None or self.scaler_y is None:
            if os.path.exists(self.scaler_X_path) and os.path.exists(self.scaler_y_path):
                self.scaler_X = joblib.load(self.scaler_X_path)
                self.scaler_y = joblib.load(self.scaler_y_path)
            else:
                raise RuntimeError("스케일러가 없습니다. pretrain()을 먼저 실행하세요.")

        # 입력 변환
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        # Monte Carlo Dropout 추론
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor).cpu().numpy()
                preds.append(y_pred)

        preds = np.stack(preds)  # (samples, batch, output_dim)
        mean_pred = preds.mean(axis=0)
        std_pred = np.abs(preds.std(axis=0))  # 항상 양수

        # 역변환
        pred_inv = self.scaler_y.inverse_transform(mean_pred)
        std_inv = self.scaler_y.inverse_transform(std_pred)

        # σ 기반 confidence 계산
        sigma = float(std_inv[-1, 0]) if std_inv.ndim > 1 else float(std_inv[-1])
        sigma = max(sigma, 1e-6)

        # 신뢰도: 불확실성 작을수록 1에 가까움
        confidence = 1 / (1 + np.log1p(sigma))

        # 가격 계산
        if current_price is None:
            current_price = getattr(self.stockdata, 'last_price', None) or self.last_price or 100.0

        # 예측된 수익률로 종가 계산
        predicted_return = float(pred_inv[-1, 0]) if pred_inv.ndim > 1 else float(pred_inv[-1])
        predicted_price = current_price * (1 + predicted_return)

        # Target 생성 및 반환
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

        # 2) 예측값 생성
        if target is None:
            X_input = self.searcher(self.ticker)  # (1,T,F)
            target = self.predict(X_input)

        # 3) GradientAnalyzer를 사용한 해석 (기존 macro_reviewer_draft 로직 활용)
        X_scaled = self.X_scaled
        if X_scaled is None:
            # searcher에서 X_scaled를 설정하지 않았다면 다시 준비
            _, X_scaled = self.prepare_features()

        # GradientAnalyzer 실행
        feature_names = list(self.scaler_X.feature_names_in_)
        X_scaled_np = X_scaled.tail(self.window).values if isinstance(X_scaled, pd.DataFrame) else X_scaled
        if X_scaled_np.ndim == 2:
            X_scaled_np = np.expand_dims(X_scaled_np, axis=0)
        X_scaled_np = X_scaled_np.astype(np.float32)
        X_scaled_np = X_scaled_np[:, :, :300]  # 최대 300개 피처만 사용
        feature_names = feature_names[:300]

        # GradientAnalyzer 실행 중...
        model = self if isinstance(self, nn.Module) else self.model
        gradient_analyzer = GradientAnalyzer(model, feature_names)
        importance_dict, temporal_df, consistency_df, sensitivity_df, grad_results = gradient_analyzer.run_all_gradients(X_scaled_np)

        # 요약 데이터 추출
        temporal_summary = temporal_df.head().to_dict(orient="records") if temporal_df is not None else []
        consistency_summary = consistency_df.to_dict(orient="records") if consistency_df is not None else []
        sensitivity_summary = sensitivity_df.to_dict(orient="records") if sensitivity_df is not None else []
        stability_summary = grad_results["stability_summary"]
        feature_summary = grad_results["feature_summary"]

        # StockData에 feature_importance 저장
        if stock_data is None:
            stock_data = self.stockdata
        if stock_data is None:
            stock_data = StockData(ticker=self.ticker)

        setattr(stock_data, self.agent_id, {
            'feature_importance': {
                'feature_summary': feature_summary,
                'importance_dict': importance_dict,
                'temporal_summary': temporal_summary,
                'consistency_summary': consistency_summary,
                'sensitivity_summary': sensitivity_summary,
                'stability_summary': stability_summary
            },
            'our_prediction': target.next_close,
            'uncertainty': round(target.uncertainty or 0.0, 8),
            'confidence': round(target.confidence or 0.0, 8)
        })

        # 4) LLM 호출(reason 생성)
        sys_text, user_text = self._build_messages_opinion(stock_data, target)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")

        # 5) Opinion 기록/반환
        self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason))

        return self.opinions[-1]

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""

        # 메시지 생성
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

    # DebateAgent.get_rebuttal() 호환용 래퍼
    def reviewer_rebuttal(
            self,
            my_opinion: Opinion,
            other_opinion: Opinion,
            round_index: int,
    ) -> Rebuttal:
        return self.reviewer_rebut(
            my_opinion=my_opinion,
            other_opinion=other_opinion,
            round=round_index,
        )




    # LLM Reasoning 메시지
    def _build_messages_opinion(self, stock_data, target):
        """ LLM 프롬프트 메시지 구성 """
        # ✅ 해당 agent_id의 데이터 가져오기
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 데이터가 필요함")

        # ✅ dataclass를 dict로 변환
        stock_data_dict = asdict(stock_data)

        # ✅ feature_importance는 agent_data 내부에서 가져오기
        feature_imp = agent_data.get("feature_importance", {})

        # ✅ context 구성
        ctx = {
            "agent_id": self.agent_id,
            "ticker": stock_data_dict.get("ticker", "Unknown"),
            "currency": stock_data_dict.get("currency", "USD"),
            "last_price": stock_data_dict.get("last_price", None),
            "our_prediction": float(target.next_close),     #our_prediction = next_close
            "uncertainty": float(target.uncertainty),
            "confidence": float(target.confidence),

            "feature_importance": {
                "feature_summary": feature_imp.get("feature_summary", []),
                "importance_dict": feature_imp.get("importance_dict", []),
                "temporal_summary": feature_imp.get("temporal_summary", []),
                'consistency_summary': feature_imp.get('consistency_summary', []),
                'sensitivity_summary': feature_imp.get('sensitivity_summary', []),
                'stability_summary': feature_imp.get('stability_summary', [])
            },
        }

        # feature_importance Top 5 요약 출력
        if 'importance_dict' in feature_imp and isinstance(feature_imp['importance_dict'], dict):
            importance_dict = feature_imp['importance_dict']
            try:
                # 숫자 값만 필터링하여 정렬
                numeric_items = [(k, v) for k, v in importance_dict.items()
                                 if isinstance(v, (int, float))]
                if numeric_items:
                    top5 = sorted(numeric_items, key=lambda x: abs(x[1]), reverse=True)[:5]
                    top5_str = ", ".join([f"{str(k)}={v:.2e}" for k, v in top5])
                    print(f"  |  [INFO] Top 5 features: {top5_str}")
            except Exception:
                pass

        # 각 컬럼별 최근 시계열 그대로 포함
        # (최근 7~14일 정도면 LLM이 이해 가능한 범위)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[self.window_size:]  # 최근 14일치 전체 시계열
            else:
                ctx[col] = [values]

        # 프롬프트 구성
        system_text = OPINION_PROMPTS[self.agent_id]["system"]
        user_text = OPINION_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        # user_text = OPINION_PROMPTS[self.agent_id]["user"].format(**ctx)

        return system_text, user_text



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

    def _build_messages_revision(
            self,
            my_opinion: Opinion,
            others: List[Opinion],
            rebuttals: Optional[List[Rebuttal]] = None,
            stock_data: StockData = None,
    ) -> tuple[str, str]:
        """
        Revision용 LLM 메시지 생성기
        - 내 의견(my_opinion), 타 에이전트 의견(others), 주가데이터(stock_data) 기반
        - rebuttals 중 나(self.agent_id)를 대상으로 한 내용만 포함
        """
        # 기본 메타데이터
        t = getattr(stock_data, "ticker", "UNKNOWN")
        ccy = getattr(stock_data, "currency", "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        # 타 에이전트 의견 및 rebuttal 통합 요약
        others_summary = []
        for o in others:
            entry = {
                "agent_id": o.agent_id,
                "predicted_price": float(o.target.next_close),
                "confidence": float(o.target.confidence),
                "uncertainty": float(o.target.uncertainty),
                "reason": str(o.reason)[:500],
            }

            # 나에게 온 rebuttal만 stance/message 추출
            if rebuttals:
                related_rebuts = [
                    {"stance": r.stance, "message": r.message}
                    for r in rebuttals
                    if r.from_agent_id == o.agent_id and r.to_agent_id == self.agent_id
                ]
                if related_rebuts:
                    entry["rebuttals_to_me"] = related_rebuts

            others_summary.append(entry)

        # Context 구성
        ctx = {
            "ticker": t,
            "currency": ccy,
            "agent_type": self.agent_id,
            "my_opinion": {
                "predicted_price": float(my_opinion.target.next_close),
                "confidence": float(my_opinion.target.confidence),
                "uncertainty": float(my_opinion.target.uncertainty),
                "reason": str(my_opinion.reason)[:1000],
            },
            "others_summary": others_summary,
            "data_summary": getattr(stock_data, self.agent_id, {}).get("feature_cols", []),
        }

        # 최근 시계열 데이터 포함 (기술/심리적 패턴)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-14:]  # 최근 14일치
            else:
                ctx[col] = [values]

        # Prompt 구성
        prompt_set = REVISION_PROMPTS.get(self.agent_id)
        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

        return system_text, user_text




    def reviewer_revise(
            self,
            my_opinion: Opinion,
            others: List[Opinion],
            rebuttals: List[Rebuttal],
            stock_data: StockData,
            fine_tune: bool = True,
            lr: float = 1e-4,
            epochs: int = 5,
    ):
        """
        MacroPredictor 전용 Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - (옵션) Keras fine-tuning
        - LLM reasoning 생성
        """
        gamma = getattr(self, "gamma", 0.3)
        delta_limit = getattr(self, "delta_limit", 0.05)

        try:
            # ① 불확실성 기반 β 계산
            my_price = my_opinion.target.next_close
            my_sigma = abs(my_opinion.target.uncertainty or 1e-6)
            other_prices = np.array([o.target.next_close for o in others])
            other_sigmas = np.array([abs(o.target.uncertainty or 1e-6) for o in others])

            inv_sigmas = 1 / (np.concatenate([[my_sigma], other_sigmas]) + 1e-6)
            betas = inv_sigmas / inv_sigmas.sum()

            delta = np.sum(betas[1:] * (other_prices - my_price))
            revised_price = my_price + gamma * delta

        except Exception as e:
            print(f"[{self.agent_id}] β/γ 계산 실패: {e}")
            revised_price = my_opinion.target.next_close

        # ② Fine-tuning (PyTorch 모델)
        loss_value = 0.0
        if fine_tune and isinstance(self, nn.Module):
            try:
                current_price = getattr(stock_data, 'last_price', None) or self.last_price or 100.0
                revised_return = (revised_price / current_price) - 1.0
                # 스케일러 적용 (MinMaxScaler 범위에 맞춤)
                revised_return_scaled = np.clip(revised_return, -1.0, 1.0)  # MinMaxScaler 범위
                revised_return_scaled = np.array([[revised_return_scaled]])  # (1, 1)
                revised_return_scaled = self.scaler_y.transform(revised_return_scaled)[0, 0]

                # 입력 데이터 준비
                X_seq = self.searcher(self.ticker)
                if isinstance(X_seq, torch.Tensor):
                    X_seq = X_seq.to(self.device)
                else:
                    X_seq = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

                y_true = torch.FloatTensor([[revised_return_scaled]]).to(self.device)

                # 모델을 training 모드로 설정
                self.train()
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)
                criterion = nn.MSELoss()

                # Fine-tuning 루프
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    pred = self(X_seq)
                    loss = criterion(pred, y_true)
                    loss.backward()
                    optimizer.step()
                    loss_value = loss.item()

                self.eval()
                print(f"[{self.agent_id}] fine-tuning 완료: loss={loss_value:.6f}")
            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")

        # ③ 새로운 Target 생성
        new_target = Target(
            next_close=float(revised_price),
            uncertainty=my_opinion.target.uncertainty,
            confidence=my_opinion.target.confidence,
        )

        # ④ LLM reasoning
        try:
            sys_text, user_text = self._build_messages_revision(
                my_opinion=my_opinion,
                others=others,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )
        except Exception as e:
            print(f"[{self.agent_id}] revision 메시지 생성 실패: {e}")
            sys_text, user_text = (
                "너는 거시경제 분석가다. 거시 지표를 기반으로 reason을 간단히 생성하라.",
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
