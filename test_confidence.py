#!/usr/bin/env python3
"""
신뢰도 계산 테스트 스크립트
각 에이전트의 predict 메서드에서 신뢰도가 제대로 계산되는지 확인
"""

import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.technical_agent import TechnicalAgent
from agents.macro_agent import MacroAgent
from agents.sentimental_agent import SentimentalAgent
from config.agents import dir_info

def test_confidence_calculation(ticker="AAPL"):
    """각 에이전트의 신뢰도 계산 테스트"""
    print("=" * 80)
    print(f"신뢰도 계산 테스트 - Ticker: {ticker}")
    print("=" * 80)
    print()
    
    agents = {
        "TechnicalAgent": TechnicalAgent,
        "MacroAgent": MacroAgent,
        "SentimentalAgent": SentimentalAgent,
    }
    
    results = {}
    
    for agent_name, AgentClass in agents.items():
        print(f"\n[{agent_name}] 테스트 시작...")
        try:
            agent = AgentClass(
                agent_id=agent_name,
                ticker=ticker,
                data_dir=dir_info["data_dir"],
                model_dir=dir_info["model_dir"]
            )
            
            # 데이터 검색
            print(f"  - 데이터 검색 중...")
            agent.search(ticker=ticker)
            
            # 예측 수행
            print(f"  - 예측 수행 중...")
            target = agent.predict(agent.stockdata, n_samples=10)  # 빠른 테스트를 위해 샘플 수 줄임
            
            # 결과 출력
            print(f"  ✅ 예측 완료:")
            print(f"     - next_close: {target.next_close:.2f}")
            print(f"     - uncertainty: {target.uncertainty}")
            print(f"     - confidence: {target.confidence}")
            print(f"     - predicted_return: {target.predicted_return}")
            
            results[agent_name] = {
                "success": True,
                "target": target
            }
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            results[agent_name] = {
                "success": False,
                "error": str(e)
            }
    
    # 요약
    print("\n" + "=" * 80)
    print("테스트 요약")
    print("=" * 80)
    
    for agent_name, result in results.items():
        if result["success"]:
            target = result["target"]
            confidence_status = "✅ 계산됨" if target.confidence is not None else "⚠️ None"
            print(f"{agent_name}:")
            print(f"  - 신뢰도: {target.confidence} {confidence_status}")
            print(f"  - 불확실성: {target.uncertainty}")
        else:
            print(f"{agent_name}: ❌ 실패 - {result['error']}")
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="신뢰도 계산 테스트")
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="테스트할 티커 (기본값: AAPL)"
    )
    
    args = parser.parse_args()
    test_confidence_calculation(args.ticker)

