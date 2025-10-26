#!/usr/bin/env python3
"""
MCP Hybrid System Launcher
전체 Stage(1~3) 자동 실행 엔트리포인트
"""
import argparse
from datetime import datetime

from agents.macro_agent import MacroPredictor
from core.evaluation import evaluate_consensus
from core.data_set import build_dataset, load_dataset
import torch
import statistics
from debate_ver3_tmp.agents.base_agent import Opinion, Target
from config.agents import agents_info
from debate_ver3_tmp.agents.technical_agent import MacroAgent
from make_macro_model.model2_lstm_all_4 import make_lstm_macro_model


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
def run_debate(agents:dict = None, ticker: str = "NVDA", rounds: int = 1, load_agent: bool = False):
    print(f"🚀 Start Monte Carlo Debate on {ticker}, rounds={rounds}")
    # 0️⃣ 에이전트 정의
    if load_agent:
        agents = {
            # "TechnicalAgent": TechnicalAgent("TechnicalAgent", ticker=ticker),
            "MacroAgent": MacroAgent("MacroAgent", ticker=ticker),
            # "SentimentalAgent": SentimentalAgent("SentimentalAgent", ticker=ticker),
        }
    else:
        agents = {
            # "TechnicalAgent": TechnicalAgent("TechnicalAgent", ticker=ticker),
            "MacroAgent": MacroAgent("MacroAgent", ticker=ticker),
            # "SentimentalAgent": SentimentalAgent("SentimentalAgent", ticker=ticker),
        }

        datasets = {}

        # 데이터셋이 없으면 먼저 생성
        if not datasets:
            print("📊 데이터셋 생성 중...")
            build_dataset(ticker=ticker)
            
        for name, agent in agents.items():
            try:
                X, y, cols = load_dataset(ticker=ticker, agent_id=name)
                print(f"✅ [{name}] CSV 로드 완료: X={X.shape}, y={y.shape}")
                
            except FileNotFoundError:
                print(f"⚠️  [{name}] CSV 없음 → build_dataset() 실행")
                build_dataset(ticker=ticker)
                # 생성된 데이터를 다시 로드
                X, y, cols = load_dataset(ticker=ticker, agent_id=name)
                print(f"✅ [{name}] 데이터 생성 완료: X={X.shape}, y={y.shape}")

            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            datasets[name] = (X_t, y_t)

        # 사전학습 실행 - 첫번째 모델 생성하기
        for agent_id, agent in agents.items():
            print(f"🤖 [{agent_id}] 사전학습 시작...")

            # MacroAgent 경우
            if agent_id == 'MacroAgent':
                macro_predictor = MacroPredictor(
                    base_date=datetime.today(),
                    window=40,
                    ticker = ticker  # ✅ 리스트 형태로 단일 티커 지정
                )
                pred_prices, target, _ = macro_predictor.run_prediction()
            else:
                agent.pretrain()

    for r in range(1, rounds + 1):
        print(f"=== Round {r} ===")
        # -----------------------------
        # 1️⃣ Monte Carlo 예측 수행
        # -----------------------------
        results = {}
        for agent_id, agent in agents.items():
            # MacroAgent 경우
            if agent_id == 'MacroAgent':
                macro_predictor = MacroPredictor(
                    base_date=datetime.today(),
                    window=40,
                    ticker = ticker  # ✅ 리스트 형태로 단일 티커 지정
                )
                pred_prices, target, _ = macro_predictor.run_prediction()

                results[agent_id] = target
                agent.opinions.append(
                    Opinion(agent_id=agent_id,
                            target=target,
                            reason="(1차 Monte Carlo 예측)")
                )
                print(f"1️⃣ MacroAgent > Monte Carlo 예측 수행 완료")

            else:
                X_input = agent.searcher(ticker)
                target = agent.predict(X_input)  # Target(next_close, uncertainty, confidence)
                results[agent_id] = target
                agent.opinions.append(
                    Opinion(agent_id=agent_id, target=target, reason="(1차 Monte Carlo 예측)")
                )
            print(f"{agent_id:<18} | mean={target.next_close:.3f}, σ={target.uncertainty:.4f}")

        # -----------------------------
        # 2️⃣ σ 기반 신뢰도(β) 계산
        # -----------------------------
        sigmas = {agent_id: t.uncertainty for agent_id, t in results.items()}
        inv = {k: 1 / (v + 1e-8) for k, v in sigmas.items()}
        total_inv = sum(inv.values())
        betas = {agent_id: inv[agent_id] / total_inv for agent_id in inv}
        print(f"β weights: {betas}")

        # -----------------------------
        # 3️⃣ revised_target 계산 및 fine-tune + reasoning
        # -----------------------------
        means = {agent_id: t.next_close for agent_id, t in results.items()}

        for agent_id, agent in agents.items():
            others_means = {k: v for k, v in means.items() if k != agent_id}
            weighted_others = sum(betas[k] * others_means[k] for k in others_means)
            revised_val = betas[agent_id] * means[agent_id] + (1 - betas[agent_id]) * weighted_others

            revised_target = Target(
                next_close=float(revised_val),
                uncertainty=results[agent_id].uncertainty,
                confidence=betas[agent_id],
            )

            old_opinion = agent.opinions[-1]
            others_ops = [agents[k].opinions[-1] for k in agents if k != agent_id]

            # ✅ Fine-tuning + Reasoning + Opinion 업데이트
            agent.reviewer_revise(
                revised_target=revised_target,
                old_opinion=old_opinion,
                rebuttals=[],
                others=others_ops,
                X_input=agent.stockdata.X[-1:] if hasattr(agent.stockdata, 'X') else None,
            )

            print(f"revise[{agent_id:<18}] → {revised_target.next_close:.3f}")

        # -----------------------------
        # 4️⃣ Round summary 저장
        # -----------------------------
        round_summary = {
            "round": r,
            "agents": {
                agent_id: {
                    "mean": results[aid].next_close,
                    "σ": results[agent_id].uncertainty,
                    "β": betas[agent_id],
                    "revised": agent.opinions[-1].target.next_close,
                }
                for aid, agent in agents.items()
            },
        }
        if not hasattr(run_debate, 'logs'):
            run_debate.logs = []
        run_debate.logs.append(round_summary)

    # -----------------------------
    # 5️⃣ 최종 앙상블 계산
    # -----------------------------
    last_means = [agent.opinions[-1].target.next_close for agent in agents.values()]
    final_mean = statistics.fmean(last_means)
    final_median = statistics.median(last_means)
    print(f"🏁 Final Ensemble → mean={final_mean:.3f}, median={final_median:.3f}")

    return run_debate.logs, {
        "ticker": ticker,
        "ensemble_mean": final_mean,
        "ensemble_median": final_median,
        "betas": betas,
    }

# ---------------------------------------------------------
# 2️⃣ CLI Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MCP Hybrid pipeline (Stage 1~3)")
    parser.add_argument("--ticker", type=str, default="NVDA", help="예측할 종목 코드")
    parser.add_argument("--epochs", type=int, default=20, help="사전학습 epoch 수")
    parser.add_argument("--mutual", type=int, default=10, help="상호학습 라운드 수")
    parser.add_argument("--debate", type=int, default=2, help="토론 라운드 수")
    args = parser.parse_args()

    logs, results = run_debate(
        ticker=args.ticker,
        rounds=args.debate,
        load_agent=False
    )
    
    print(f"\n🎯 최종 결과:")
    print(f"   - 앙상블 평균: {results['ensemble_mean']:.3f}")
    print(f"   - 앙상블 중앙값: {results['ensemble_median']:.3f}")
    print(f"   - β 가중치: {results['betas']}")
