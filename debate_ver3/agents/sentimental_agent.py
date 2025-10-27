# debate_ver3/agents/sentimental_agent.py
# ===============================================================
# SentimentalAgent: 감성(뉴스/텍스트) + LSTM 기반 예측 에이전트
#  - BaseAgent에 완전 호환 (reviewer_* 로직은 BaseAgent 구현 사용)
#  - 데이터는 core/data_set.py에서 만든 CSV를 BaseAgent.searcher로 로드
#  - 타깃은 "다음날 수익률"이며 BaseAgent.predict가 "가격"으로 변환
# ===============================================================

from __future__ import annotations
import json
from typing import Tuple

import torch
import torch.nn as nn

from debate_ver3.agents.base_agent import BaseAgent, StockData, Target, Opinion
from debate_ver3.config.agents import agents_info
from debate_ver3.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS


# ---------------------------------------------------------------
# 모델 정의: LSTM + Dropout (MC Dropout을 위해 train() 상태에서 사용)
# ---------------------------------------------------------------
class SentimentalNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)         # (B, T, H)
        h_last = out[:, -1, :]        # (B, H)
        h_last = self.dropout(h_last) # MC Dropout: 학습/추론 공통 적용
        y = self.fc(h_last)           # (B, 1)  → "다음날 수익률(정규화 공간)"
        return y


# ---------------------------------------------------------------
# 에이전트 구현
# ---------------------------------------------------------------
class SentimentalAgent(BaseAgent, nn.Module):
    def __init__(self, agent_id: str = "SentimentalAgent", verbose: bool = False, ticker: str = "TSLA"):
        BaseAgent.__init__(self, agent_id=agent_id, verbose=verbose, ticker=ticker)
        nn.Module.__init__(self)

        cfg = agents_info.get("SentimentalAgent", {})
        input_dim  = int(cfg.get("input_dim", 8))
        hidden_dim = int(cfg.get("d_model", 128))
        num_layers = int(cfg.get("num_layers", 2))
        dropout    = float(cfg.get("dropout", 0.2))

        self.net = SentimentalNet(input_dim, hidden_dim, num_layers, dropout)
        self.window_size = int(cfg.get("window_size", 40))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # -----------------------------------------------------------
    # LLM: Opinion 메시지 구성
    # -----------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """
        OPINION_PROMPTS['SentimentalAgent']가 없으면 기본 프롬프트로 대체.
        context에는 모델 예측/불확실성, 최근 윈도 길이, 마지막 가격, 사용 피처 일부를 담습니다.
        """
        prompt_set = OPINION_PROMPTS.get("SentimentalAgent") or {
            "system": (
                "너는 감성/뉴스 기반 단기 주가 분석 전문가다. "
                "제공된 컨텍스트(예측값, 불확실성, 특징요약)를 근거로 "
                "다음 종가에 대한 간결하고 논리적인 의견을 한국어로 3~4문장으로 작성해."
            ),
            "user": "다음 컨텍스트를 바탕으로 의견을 작성:\n{context}",
        }

        # ✅ 뉴스 감성 점수(news_sentiment) 포함된 컨텍스트
        ctx = {
            "agent_id": self.agent_id,
            "ticker": stock_data.ticker,
            "last_price": getattr(stock_data, "last_price", None),
            "window_size": self.window_size,
            "our_prediction": float(getattr(target, "next_close", 0.0)),
            "uncertainty": float(getattr(target, "uncertainty", 0.0) or 0.0),
            "confidence": float(getattr(target, "confidence", 0.0) or 0.0),
            "feature_cols": (stock_data.feature_cols[:12] if stock_data.feature_cols else None),
            "news_sentiment": getattr(stock_data, "news_sentiment", None),
        }

        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    # -----------------------------------------------------------
    # Rebuttal / Revision 메시지 기본 구현
    # -----------------------------------------------------------
    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        prompt_set = REBUTTAL_PROMPTS.get("SentimentalAgent") or {
            "system": "너는 토론에서 상대 의견의 강점/약점을 짚는 분석가다.",
            "user": "컨텍스트를 바탕으로 REBUT 또는 SUPPORT와 메시지를 JSON으로 답하라:\n{context}",
        }
        context = kwargs.get("context", "")
        return prompt_set["system"], prompt_set["user"].format(context=context)

    def _build_messages_revision(self, *args, **kwargs) -> Tuple[str, str]:
        prompt_set = REVISION_PROMPTS.get("SentimentalAgent") or REVISION_PROMPTS.get("default") or {
            "system": "수정된 예측값과 반박 내용을 반영해 최종 의견을 간결히 정리하라.",
            "user": "컨텍스트:\n{context}",
        }
        context = kwargs.get("context", "")
        return prompt_set["system"], prompt_set["user"].format(context=context)
