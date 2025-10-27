import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import os
from debate_ver4.agents_tmp.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal
from debate_ver4.config.agents import agents_info, dir_info
import json
from debate_ver4.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from typing import List, Optional


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
        **kwargs
    ):
        # 기본 초기화
        BaseAgent.__init__(self, agent_id, **kwargs)
        nn.Module.__init__(self)

        self.dropout_rate = float(dropout)  # 🔹 float형 dropout 값 저장
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # 입력 프로젝션
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer 인코더 정의 (float형 dropout 사용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=self.dropout_rate,  # float값 전달
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 레이어 및 학습 세팅
        self.dropout = nn.Dropout(self.dropout_rate)  # nn.Dropout 객체는 따로
        self.fc = nn.Linear(d_model, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.last_pred = None
 

    def _build_model(self):
        """SentimentalAgent 기본 Transformer 모델 자동 생성"""
        import torch.nn as nn

        input_dim = getattr(self, "input_dim", 8)
        d_model = getattr(self, "d_model", 64)
        nhead = getattr(self, "nhead", 4)
        num_layers = getattr(self, "num_layers", 2)
        dropout_rate = getattr(self, "dropout_rate", 0.1)

        class TransformerNet(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, dropout_rate):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 2,
                    dropout=dropout_rate,
                    activation='gelu',
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc = nn.Sequential(
                    nn.Linear(d_model, 1),
                    nn.Tanh()   # 🔹출력값을 -1~1로 제한
                )

            def forward(self, x):
                x = self.input_projection(x)
                x = self.transformer(x)     # TransformerEncoder는 Tensor 반환
                x = x[:, -1, :]             # 마지막 시점 hidden 사용
                x = self.dropout(x)
                return self.fc(x)

        model = TransformerNet(input_dim, d_model, nhead, num_layers, dropout_rate)
        print(f" SentimentalAgent Transformer 생성 완료 "
            f"(d_model={d_model}, nhead={nhead}, layers={num_layers})")
        return model

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
        

   # LLM Reasoning 메시지
    def _build_messages_opinion(self, stock_data, target):
        """FundamentalAgent용 LLM 프롬프트 메시지 구성 (시계열 포함 버전)"""

        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        # 기본 컨텍스트
        ctx = {
            "ticker": getattr(stock_data, "ticker", "Unknown"),
            "currency": getattr(stock_data, "currency", "USD"),
            "last_price": getattr(stock_data, "last_price", None),
            "our_prediction": float(target.next_close),
            "uncertainty": float(target.uncertainty),
            "confidence": float(target.confidence),
            "recent_days": len(next(iter(agent_data.values()))) if agent_data else 0,
        }

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