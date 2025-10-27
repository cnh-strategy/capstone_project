import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import os
from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal
from config.agents import agents_info, dir_info
import json
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from typing import List, Optional

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
        **kwargs
    ):
        BaseAgent.__init__(self, agent_id, **kwargs)
        nn.Module.__init__(self)

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # LSTM 모델 정의
        self.lstm = nn.LSTM(
            self.input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

        # 학습 세팅
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        """FundamentalAgent 기본 LSTM 모델 자동 생성 (학습 수렴 개선 버전)"""
        import torch.nn as nn

        input_dim = getattr(self, "input_dim", 16)
        hidden_dim = getattr(self, "hidden_dim", 64)
        num_layers = getattr(self, "num_layers", 2)
        dropout_rate = getattr(self, "dropout_rate", 0.1)  # 🔹조정: 0.2 → 0.1

        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    dropout=dropout_rate if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout_rate)
                # 🔹출력층에 Tanh 추가 (출력 제한)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Tanh()    # 🔹출력값 -1~1로 제한
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]       # 마지막 시점 hidden
                out = self.dropout(out)
                return self.fc(out)

        model = LSTMNet(input_dim, hidden_dim, num_layers, dropout_rate)
        print(f"🧠 FundamentalAgent LSTM 생성 완료 "
            f"(input_dim={input_dim}, hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout_rate})")
        return model


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
        # -----------------------------
        # 기본 메타데이터
        # -----------------------------
        t = getattr(stock_data, "ticker", "UNKNOWN")
        ccy = getattr(stock_data, "currency", "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        # -----------------------------
        # 타 에이전트 의견 및 rebuttal 통합 요약
        # -----------------------------
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

        # -----------------------------
        # Context 구성
        # -----------------------------
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

        # -----------------------------
        # Prompt 구성
        # -----------------------------
        prompt_set = REVISION_PROMPTS.get(self.agent_id)
        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

        return system_text, user_text

