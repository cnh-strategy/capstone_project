#!/usr/bin/env python3
"""
Run Test: Evaluate Pretrained MCP Agents
사전학습된 Agent 모델들을 불러와 테스트 데이터를 통해 예측 및 비교 수행
"""
import os
import torch
import pandas as pd
from datetime import datetime
from core.preprocessing import build_dataset
from core.evaluation import evaluate_agents
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent

# ---------------------------------------------------------
# 1️⃣ 사전학습된 모델 로드
# ---------------------------------------------------------
def load_pretrained_agents(model_dir="models"):
    agents = {
        "technical": TechnicalAgent("TechnicalAgent"),
        "fundamental": FundamentalAgent("FundamentalAgent"),
        "sentimental": SentimentalAgent("SentimentalAgent"),
    }

    for name, agent in agents.items():
        model_path = os.path.join(model_dir, f"{name}_agent.pt")
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            agent.model.load_state_dict(state["model_state_dict"])
            print(f"✅ Loaded pretrained model for {name}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
    return agents

# ---------------------------------------------------------
# 2️⃣ 테스트 실행
# ---------------------------------------------------------
def test_agents(ticker="TSLA"):
    print("\n🧪 [Test MCP Agents]")
    print("=" * 60)
    X, y, _, _ = build_dataset(ticker)
    X_t, y_t = torch.tensor(X), torch.tensor(y)

    # 모델 로드
    agents = load_pretrained_agents()

    # 예측 및 평가
    results = evaluate_agents(agents, X_t, y_t)

    # 결과 요약
    print("\n📊 Agent Test Summary")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name.capitalize():15} RMSE={metrics['RMSE']:.4f} | "
              f"MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.3f}")
    print("=" * 60)
    print(f"✅ Test Completed for {ticker}\n")

# ---------------------------------------------------------
# 3️⃣ CLI Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MCP Agent Model Tests")
    parser.add_argument("--ticker", type=str, default="TSLA", help="테스트할 종목 코드")
    args = parser.parse_args()

    test_agents(args.ticker)
