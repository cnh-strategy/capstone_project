#!/usr/bin/env python3
"""
Agent Test Runner (Config 기반)
사전학습된 에이전트를 불러와 테스트 데이터를 통해 예측 및 비교 수행
"""

import os
import pandas as pd
from core.agent_loader import AgentLoader
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent
from tabulate import tabulate


def test_agents(ticker: str = "TSLA", config_path: str = "config/agents.yaml"):
    """
    사전학습된 모든 에이전트를 로드하고, 테스트 데이터를 기반으로 예측 수행
    Args:
        ticker (str): 테스트할 주식 코드
        config_path (str): YAML 설정파일 경로
    Returns:
        results (dict): agent별 예측 결과
    """

    print("🧪 Testing Pretrained Agents (Config-based)")
    print("=" * 60)

    # 1️⃣ 에이전트 클래스 매핑
    class_map = {
        "TechnicalAgent": TechnicalAgent,
        "FundamentalAgent": FundamentalAgent,
        "SentimentalAgent": SentimentalAgent,
    }

    # 2️⃣ AgentLoader 로드
    loader = AgentLoader(config_path=config_path, class_map=class_map)
    agents = loader.load_all()

    # 3️⃣ 테스트 데이터 로드
    test_data = {}
    for name in agents.keys():
        path = os.path.join(loader.data_dir, f"{ticker}_{name}_test.csv")
        try:
            df = pd.read_csv(path)
            test_data[name] = df
            print(f"✅ {name.capitalize()} test data loaded ({len(df)} samples)")
        except Exception as e:
            print(f"❌ {name.capitalize()} data load failed: {e}")

    # 4️⃣ 예측 수행
    results = []
    for agent_name, df in test_data.items():
        try:
            pred = loader.predict(agent_name, df)
            actual = df["Close"].iloc[-1]
            error = abs(pred - actual)
            results.append({
                "Agent": agent_name.capitalize(),
                "Predicted": round(pred, 3),
                "Actual": round(actual, 3),
                "Error": round(error, 3),
            })
        except Exception as e:
            results.append({
                "Agent": agent_name.capitalize(),
                "Predicted": None,
                "Actual": None,
                "Error": f"Error: {str(e)}",
            })

    # 5️⃣ 결과 요약 출력
    print("\n🔮 Prediction Results")
    print("-" * 60)
    print(tabulate(results, headers="keys", tablefmt="github"))

    # 6️⃣ 에이전트 정보 요약
    info_summary = []
    for name in agents.keys():
        try:
            info = loader.get_info(name)
            info_summary.append({
                "Agent": name.capitalize(),
                "Model": info["model_class"],
                "Params": f"{info['params']:,}",
                "Features": info["features"],
                "Window": info["window"],
            })
        except Exception as e:
            info_summary.append({
                "Agent": name.capitalize(),
                "Model": "N/A",
                "Params": "N/A",
                "Features": "N/A",
                "Window": f"Error: {e}",
            })

    print("\n📊 Agent Information")
    print("-" * 60)
    print(tabulate(info_summary, headers="keys", tablefmt="github"))

    return {
        "predictions": results,
        "agent_info": info_summary
    }


# 단독 실행 가능
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pretrained Agent Tests")
    parser.add_argument("--ticker", type=str, default="TSLA", help="테스트할 티커 (기본값: TSLA)")
    parser.add_argument("--config", type=str, default="config/agents.yaml", help="에이전트 설정파일 경로")
    args = parser.parse_args()

    result = test_agents(ticker=args.ticker, config_path=args.config)
