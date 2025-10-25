import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
from agents.base_agent import BaseAgent, StockData, Target

class TechnicalAgent(BaseAgent, nn.Module):
    """Technical Agent: BaseAgent + TCN 기반 DL 모델"""
    def __init__(self, agent_id="TechnicalAgent", input_dim=10, hidden_dim=64, dropout=0.1):
        BaseAgent.__init__(self, agent_id)
        nn.Module.__init__(self)
        # 통일된 모델 구조 (nn.Module 상속)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Temporal Convolutional Network
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim//2, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.last_pred = None
    
    def forward(self, x):
        """Forward pass for the model"""
        # x shape: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def searcher(self, ticker: str, window_size=7, data_dir="data/processed", agent_type="technical", rebuild=False):
        """
        기술적 지표 기반 searcher
        - 기본값:
            window_size = 7
            data_dir = "data/processed"
            agent_type = "technical"
        """
        dataset_path = os.path.join(data_dir, f"{ticker}_{agent_type}_dataset.csv")
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {agent_type} dataset 생성 중...")
            build_dataset(ticker, save_dir=data_dir, window_size=window_size)

        X, y, _, _, _ = load_csv_dataset(ticker, agent_type=agent_type, save_dir=data_dir)
        X_latest = X[-1:]
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)
        print(f"✅ {ticker} {agent_type} searcher loaded shape: {X_tensor.shape}")
        return X_tensor
    
    def _compute_rsi(self, series, window=14):
        """RSI 계산"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        return 100 - (100 / (1 + rs))

    # 2️⃣ 예측
    def predicter(self, stock_data: StockData) -> Target:
        df = pd.DataFrame(stock_data.technical)
        
        # 실제 컬럼명 확인 및 정규화
        print(f"Debug: Available columns: {list(df.columns)}")
        
        # 컬럼명이 튜플인 경우 첫 번째 요소만 사용
        if len(df.columns) > 0 and isinstance(df.columns[0], tuple):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            print(f"Debug: Normalized columns: {list(df.columns)}")
        
        # 10개 피처 사용 (전처리와 일치)
        feature_cols = ["Open", "High", "Low", "Close", "Volume", "returns", "sma_5", "sma_20", "rsi", "volume_z"]
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 5:  # 최소 5개 피처 필요
            # 기본 피처만 사용
            available_cols = ["Open", "High", "Low", "Close", "Volume"]
            # 실제로 존재하는 컬럼만 사용
            available_cols = [col for col in available_cols if col in df.columns]
        
        print(f"Debug: Using columns: {available_cols}")
        
        if len(available_cols) == 0:
            # 컬럼이 없으면 기본값 반환
            return Target(next_close=100.0)
        
        feats = torch.tensor(df[available_cols].values[-7:], dtype=torch.float32)
        x = feats.unsqueeze(0)  # (1, 7, features)
        
        # 입력 차원을 모델이 기대하는 차원으로 패딩 (10개 피처)
        if x.shape[-1] < 10:
            padding = torch.zeros(x.shape[0], x.shape[1], 10 - x.shape[-1])
            x = torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > 10:
            x = x[:, :, :10]  # 10개 피처만 사용
        
        with torch.no_grad():
            pred = self.forward(x).item()
        self.last_pred = pred
        return Target(next_close=pred)

    # 3️⃣ 학습 / 상호학습
    def train_model(self, X, y, epochs=10):
        self.train()
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_parameters(self, X, y_mutual):
        """Selective Mutual Learning"""
        self.train_model(X, y_mutual, epochs=3)

    # 4️⃣ LLM Reasoning 메시지
    def _build_messages_opinion(self, stock_data: StockData, target: Target):
        sys = "너는 기술적 지표를 기반으로 단기 주가를 예측하는 애널리스트다."
        
        # last_price가 Series인 경우 처리
        last_price = stock_data.last_price
        if hasattr(last_price, 'iloc'):
            last_price = last_price.iloc[-1] if len(last_price) > 0 else 100.0
        elif hasattr(last_price, 'item'):
            last_price = last_price.item()
        
        user = f"최근 종가: {float(last_price):.2f}, 예측 종가: {target.next_close:.2f}.\n주요 지표:\n{stock_data.technical[-1]}"
        return sys, user

    def _build_messages_rebuttal(self, my_latest, other_id, other_opinion, stock_data):
        sys = "너는 다른 기술적 애널리스트의 의견을 평가하는 투자 전문가다."
        user = (
            f"내 예측: {my_latest.target.next_close:.2f} / "
            f"상대({other_id}) 예측: {other_opinion.target.next_close:.2f}\n"
            f"내 근거: {my_latest.reason}\n상대 근거: {other_opinion.reason}"
        )
        return sys, user

    # 5️⃣ LLM 수정 결과 반영
    def apply_revision(self, revised_opinion):
        self.last_pred = revised_opinion.target.next_close
        return self.last_pred
