import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import os
from agents.base_agent import BaseAgent, StockData, Target
from config.agents import agents_info, dir_info

class FundamentalAgent(BaseAgent, nn.Module):
    def __init__(self, 
    agent_id="FundamentalAgent", 
    input_dim=agents_info["FundamentalAgent"]["input_dim"],
    hidden_dim=agents_info["FundamentalAgent"]["hidden_dim"],
    num_layers=agents_info["FundamentalAgent"]["num_layers"],
    dropout=agents_info["FundamentalAgent"]["dropout"],
    data_dir=dir_info["data_dir"],
    window_size=agents_info["FundamentalAgent"]["window_size"],
    epochs=agents_info["FundamentalAgent"]["epochs"],
    learning_rate=agents_info["FundamentalAgent"]["learning_rate"],
    batch_size=agents_info["FundamentalAgent"]["batch_size"],
    **kwargs):

        # BaseAgent 초기화, nn.Module 초기화
        BaseAgent.__init__(self, agent_id, **kwargs)
        nn.Module.__init__(self)

        # 모델 하이퍼파라미터 설정
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 모델 설정
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout 설정
        self.dropout = nn.Dropout(dropout)
        
        # 출력 레이어 설정
        self.fc = nn.Linear(hidden_dim, 1)
        
        # 최적화 설정, 손실함수 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x) -> torch.Tensor:
        # LSTM 모델 출력 계산
        # x shape: (batch, time, features)
        lstm_out, _ = self.lstm(x)
        # 마지막 시점의 출력 사용
        last_output = lstm_out[:, -1, :]
        # Dropout 적용
        last_output = self.dropout(last_output)
        # 출력 레이어 적용
        output = self.fc(last_output)
        return output
    

    # 4️⃣ LLM Reasoning 메시지
    def _build_messages_opinion(self, stock_data: StockData, target: Target):
        sys = "너는 거시경제 지표와 시장 환경을 기반으로 주가를 분석하는 애널리스트다."
        # last_price가 Series인 경우 처리
        last_price = stock_data.last_price
        if hasattr(last_price, 'iloc'):
            last_price = last_price.iloc[-1] if len(last_price) > 0 else 100.0
        elif hasattr(last_price, 'item'):
            last_price = last_price.item()
        
        # 상승/하락율 계산
        return_rate = (target.next_close - float(last_price)) / float(last_price) * 100
        user = f"최근 종가: ${float(last_price):.2f}, 예측 상승률: {return_rate:+.2f}%"
        return sys, user

    def _build_messages_rebuttal(self, my_latest, other_id, other_opinion, stock_data):
        sys = "너는 다른 거시경제 분석가의 의견을 평가하는 투자 전문가다."
        # 상승/하락율로 표시
        my_return = (my_latest.target.next_close - float(stock_data.last_price)) / float(stock_data.last_price) * 100
        other_return = (other_opinion.target.next_close - float(stock_data.last_price)) / float(stock_data.last_price) * 100
        
        user = (f"내 예측: {my_return:+.2f}% / "
                f"상대({other_id}) 예측: {other_return:+.2f}%\n"
                f"내 근거: {my_latest.reason}\n상대 근거: {other_opinion.reason}")
        return sys, user
    
