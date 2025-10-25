#!/usr/bin/env python3
"""
Stage 3: Inference-Time Debate with Uncertainty
-----------------------------------------------
각 Agent의 예측(next_close, σ, β)을 Monte Carlo Dropout으로 계산하고,
LLM을 통해 reasoning을 생성한 뒤,
가중 평균 및 요약 reasoning을 산출한다.
"""

import torch
import numpy as np
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent
from core.debate_engine import aggregate_consensus

# -----------------------------------
# 1️⃣ Agent 로드
# -----------------------------------
agents = {
    "technical_agent": TechnicalAgent(),
    "fundamental_agent": FundamentalAgent(),
    "sentimental_agent": SentimentalAgent(),
}

ticker = "TSLA"
opinions = {}

print(f"\n🔮 Stage 3: Inference-Time Debate on {ticker}")
print("=" * 70)

# -----------------------------------
# 2️⃣ 각 Agent 예측 및 reasoning 생성
# -----------------------------------
for name, agent in agents.items():
    print(f"\n🤖 {name.upper()} ────────────────────────────────")

    # (1) 데이터 검색
    X = agent.searcher(ticker)

    # (2) 예측 (Monte Carlo Dropout)
    target = agent.predicter(X)
    print(f"  • Predicted next_close = {target.next_close:.3f}")
    print(f"  • σ(uncertainty) = {target.uncertainty:.5f}, β(confidence) = {target.confidence:.5f}")

    # (3) reasoning 생성
    opinion = agent.reviewer_draft(X, target)
    print(f"  • Reasoning: {opinion.reason[:200]}...")

    opinions[name] = opinion

# -----------------------------------
# 3️⃣ 최종 합의 (Consensus)
# -----------------------------------
print("\n🧩 Aggregating consensus from all agents...")
result = aggregate_consensus(agents, opinions, llm_client=None)  # 내부에 LLM 모듈 있을 시 교체

print("\n🏁 Final Consensus Result")
print("=" * 70)
print(f"📈 Weighted mean next_close : {result['consensus_next_close']:.3f}")
print(f"📉 Weighted std             : {result['std']:.5f}")
print(f"🧠 Summary reasoning        : {result['reason']}")
print(f"🔢 Weights by agent         : {result['weights']}")
