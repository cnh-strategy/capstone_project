import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import os
from agents.base_agent import BaseAgent, StockData, Target
from config.agents import agents_info, dir_info

class TechnicalAgent(BaseAgent, nn.Module):
    """Technical Agent: BaseAgent + GRU 기반 DL 모델"""
    def __init__(self, 
    agent_id="TechnicalAgent", 
    input_dim=agents_info["TechnicalAgent"]["input_dim"],
    hidden_dim=agents_info["TechnicalAgent"]["hidden_dim"],
    dropout=agents_info["TechnicalAgent"]["dropout"],
    data_dir=dir_info["data_dir"],
    window_size=agents_info["TechnicalAgent"]["window_size"],
    epochs=agents_info["TechnicalAgent"]["epochs"],
    learning_rate=agents_info["TechnicalAgent"]["learning_rate"],
    batch_size=agents_info["TechnicalAgent"]["batch_size"],
    **kwargs):

        BaseAgent.__init__(self, agent_id, **kwargs)
        nn.Module.__init__(self)
        # 통일된 모델 구조 (nn.Module 상속)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU 모델
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.last_pred = None
    
    def forward(self, x) -> torch.Tensor:
        """Forward pass for the model"""
        # x shape: (batch, time, features)
        gru_out, _ = self.gru(x)
        # 마지막 시점의 출력 사용
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

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
    
