import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
from agents.base_agent import BaseAgent, StockData, Target

class SentimentalAgent(BaseAgent, nn.Module):
    """Sentimental Agent: BaseAgent + Transformer 기반 감성 분석"""
    def __init__(self, agent_id="SentimentalAgent", input_dim=8, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        BaseAgent.__init__(self, agent_id)
        nn.Module.__init__(self)
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.last_pred = None
    
    def forward(self, x):
        """Forward pass for the model"""
        # x shape: (batch, time, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Use the last time step output
        last_output = x[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

    def searcher(self, ticker: str, window_size=7, data_dir="data/processed", agent_type="sentimental", rebuild=False):
        dataset_path = os.path.join(data_dir, f"{ticker}_{agent_type}_dataset.csv")
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {agent_type} dataset 생성 중...")
            build_dataset(ticker, save_dir=data_dir, window_size=window_size)

        X, y, _, _, _ = load_csv_dataset(ticker, agent_type=agent_type, save_dir=data_dir)
        X_latest = X[-1:]
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)
        print(f"✅ {ticker} {agent_type} searcher loaded shape: {X_tensor.shape}")
        return X_tensor
        
    # 2️⃣ 예측
    def predicter(self, stock_data: StockData) -> Target:
        df = pd.DataFrame(stock_data.sentimental)
        
        # 컬럼명이 튜플인 경우 첫 번째 요소만 사용
        if len(df.columns) > 0 and isinstance(df.columns[0], tuple):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # 8개 피처 사용 (전처리와 일치)
        feature_cols = ["returns", "sentiment_mean", "sentiment_vol", "Close", "Volume", "Open", "High", "Low"]
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 3:  # 최소 3개 피처 필요
            # 기본 피처만 사용
            available_cols = ["returns", "sentiment_mean", "sentiment_vol"]
            available_cols = [col for col in available_cols if col in df.columns]
        
        if len(available_cols) == 0:
            return Target(next_close=100.0)
        
        feats = torch.tensor(df[available_cols].values[-7:], dtype=torch.float32)
        x = feats.unsqueeze(0)  # (1, 7, features)
        
        # 입력 차원을 모델이 기대하는 차원으로 패딩 (8개 피처)
        if x.shape[-1] < 8:
            padding = torch.zeros(x.shape[0], x.shape[1], 8 - x.shape[-1])
            x = torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > 8:
            x = x[:, :, :8]  # 8개 피처만 사용
        
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
        self.train_model(X, y_mutual, epochs=3)

    # 4️⃣ LLM Reasoning 메시지
    def _build_messages_opinion(self, stock_data: StockData, target: Target):
        sys = "너는 투자자 심리와 뉴스 흐름을 분석하는 감성 기반 애널리스트다."
        
        # last_price가 Series인 경우 처리
        last_price = stock_data.last_price
        if hasattr(last_price, 'iloc'):
            last_price = last_price.iloc[-1] if len(last_price) > 0 else 100.0
        elif hasattr(last_price, 'item'):
            last_price = last_price.item()
        
        user = f"최근 종가: {float(last_price):.2f}, 예측 종가: {target.next_close:.2f}"
        return sys, user

    def _build_messages_rebuttal(self, my_latest, other_id, other_opinion, stock_data):
        sys = "너는 다른 감성 애널리스트의 의견을 평가하는 투자 전문가다."
        user = (f"내 예측: {my_latest.target.next_close:.2f} / "
                f"상대({other_id}) 예측: {other_opinion.target.next_close:.2f}\n"
                f"내 근거: {my_latest.reason}\n상대 근거: {other_opinion.reason}")
        return sys, user

    # 5️⃣ 수정 반영
    def apply_revision(self, revised_opinion):
        self.last_pred = revised_opinion.target.next_close
        return self.last_pred
