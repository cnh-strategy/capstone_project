#!/usr/bin/env python3
"""
MCP Hybrid System Launcher
전체 Stage(1~3) 자동 실행 엔트리포인트
"""
import argparse
from datetime import datetime
from core.training import pretrain_all_agents
from core.debate_engine import mutual_learning_with_individual_data
from core.orchestrator import run_debate_rounds
from core.evaluation import evaluate_consensus
from core.preprocessing import build_dataset, load_csv_dataset
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent
import torch

# ---------------------------------------------------------
# 0️⃣ 헬퍼 함수들
# ---------------------------------------------------------
def evaluate_agents_with_individual_data(agents, datasets):
    """각 에이전트별로 적절한 데이터를 사용하여 평가"""
    print("🤖 개별 에이전트 평가:")
    for name, agent in agents.items():
        X_agent, y_agent = datasets[name]
        with torch.no_grad():
            predictions = agent.forward(X_agent)
            mse = torch.mean((predictions - y_agent) ** 2).item()
            mae = torch.mean(torch.abs(predictions - y_agent)).item()
            print(f"   - {name}: MSE={mse:.6f}, MAE={mae:.6f}")

# ---------------------------------------------------------
# 1️⃣ 전체 MCP 실행 함수
# ---------------------------------------------------------
def run_mcp_pipeline(ticker="TSLA", pre_epochs=20, mutual_rounds=10, debate_rounds=2):
    print("\n🚀 [MCP Hybrid System Orchestration Start]")
    print(f"Ticker: {ticker}")
    print("=" * 70)

    # 0️⃣ 에이전트 정의
    agents = {
        "technical": TechnicalAgent("TechnicalAgent"),
        "fundamental": FundamentalAgent("FundamentalAgent"),
        "sentimental": SentimentalAgent("SentimentalAgent"),
    }

    # 1️⃣ 에이전트별 데이터셋 로드
    datasets = {}

    for name, agent in agents.items():
        try:
            X, y, cols = load_csv_dataset(ticker, name)
            print(f"✅ [{name}] CSV 로드 완료: X={X.shape}, y={y.shape}")
            
        except FileNotFoundError:
            print(f"⚠️  [{name}] CSV 없음 → build_dataset() 실행")
            # build_dataset은 모든 에이전트 데이터를 한번에 생성
            if not datasets:  # 첫 번째 에이전트에서만 실행
                build_dataset(ticker)
            # 생성된 데이터를 다시 로드
            X, y, col = load_csv_dataset(ticker, name)
            print(f"✅ [{name}] 데이터 생성 완료: X={X.shape}, y={y.shape}")

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        datasets[name] = (X_t, y_t)

    for name, agent in agents.items():
        agent.pretrain()
        
    # 2️⃣ Stage 1: 사전학습
    print("\n # Stage 1: Pretraining Agents")
    pretrain_all_agents(agents, datasets, epochs=pre_epochs)

    # 3️⃣ Stage 2: Selective Mutual Learning
    print("\n # Stage 2: Selective Mutual Learning")
    # 각 에이전트별로 적절한 데이터를 사용하여 상호 학습
    mutual_learning_with_individual_data(agents, datasets, rounds=mutual_rounds)

    # 4️⃣ Stage 3: Debate & Revision
    print("\n # Stage 3: LLM Debate & Consensus")
    run_debate_rounds(agents, ticker, max_rounds=debate_rounds)

    # 5️⃣ 평가
    print("\n # 평가 지표 출력")
    evaluate_agents_with_individual_data(agents, datasets)
    evaluate_consensus(agents)

    print("\ MCP Pipeline Completed Successfully.")
    print("=" * 70)



# ---------------------------------------------------------
# 2️⃣ CLI Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MCP Hybrid pipeline (Stage 1~3)")
    parser.add_argument("--ticker", type=str, default="TSLA", help="예측할 종목 코드")
    parser.add_argument("--epochs", type=int, default=20, help="사전학습 epoch 수")
    parser.add_argument("--mutual", type=int, default=10, help="상호학습 라운드 수")
    parser.add_argument("--debate", type=int, default=2, help="토론 라운드 수")
    args = parser.parse_args()

    run_mcp_pipeline(
        ticker=args.ticker,
        pre_epochs=args.epochs,
        mutual_rounds=args.mutual,
        debate_rounds=args.debate,
    )
