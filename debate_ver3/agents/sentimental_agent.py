import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import os
from agents.base_agent import BaseAgent, StockData, Target
from config.agents import agents_info, dir_info

class SentimentalAgent(BaseAgent, nn.Module):
    """Sentimental Agent: BaseAgent + Transformer 기반 감성 분석"""
    def __init__(self, 
    agent_id="SentimentalAgent", 
    input_dim=agents_info["SentimentalAgent"]["input_dim"],
    d_model=agents_info["SentimentalAgent"]["d_model"],
    nhead=agents_info["SentimentalAgent"]["nhead"],
    num_layers=agents_info["SentimentalAgent"]["num_layers"],
    dropout=agents_info["SentimentalAgent"]["dropout"],
    data_dir=dir_info["data_dir"],
    window_size=agents_info["SentimentalAgent"]["window_size"],
    epochs=agents_info["SentimentalAgent"]["epochs"],
    learning_rate=agents_info["SentimentalAgent"]["learning_rate"],
    batch_size=agents_info["SentimentalAgent"]["batch_size"],
    **kwargs):

        BaseAgent.__init__(self, agent_id, **kwargs)
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
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
    

