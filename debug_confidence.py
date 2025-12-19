#!/usr/bin/env python3
"""
신뢰도 계산 디버깅 스크립트
왜 TechnicalAgent와 MacroAgent에서 confidence가 None인지 확인
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.technical_agent import TechnicalAgent
from agents.macro_agent import MacroAgent
from agents.sentimental_agent import SentimentalAgent
from config.agents import dir_info, common_params

def debug_confidence_calculation(ticker="AAPL"):
    """각 에이전트의 신뢰도 계산 디버깅"""
    print("=" * 80)
    print(f"신뢰도 계산 디버깅 - Ticker: {ticker}")
    print("=" * 80)
    print()
    
    agents = {
        "TechnicalAgent": TechnicalAgent,
        "MacroAgent": MacroAgent,
        "SentimentalAgent": SentimentalAgent,
    }
    
    for agent_name, AgentClass in agents.items():
        print(f"\n[{agent_name}] 디버깅 시작...")
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
            
            # 모델 상태 확인
            print(f"  - 모델 상태 확인:")
            print(f"     model is None: {agent.model is None}")
            if agent.model is not None:
                print(f"     model type: {type(agent.model)}")
                print(f"     model_loaded: {getattr(agent, 'model_loaded', 'N/A')}")
            
            # 데이터셋 확인
            dataset_path = os.path.join(dir_info["data_dir"], f"{ticker}_{agent_name}_dataset.csv")
            print(f"  - 데이터셋 확인:")
            print(f"     dataset exists: {os.path.exists(dataset_path)}")
            
            if os.path.exists(dataset_path):
                import pandas as pd
                df = pd.read_csv(dataset_path)
                unique_samples = sorted(df['sample_id'].unique())
                lookback_days = common_params.get("confidence_lookback_days", 30)
                print(f"     unique_samples: {len(unique_samples)}")
                print(f"     lookback_days: {lookback_days}")
                print(f"     samples >= lookback_days: {len(unique_samples) >= lookback_days}")
            
            # _calculating_confidence 플래그 확인
            print(f"  - _calculating_confidence: {getattr(agent, '_calculating_confidence', False)}")
            
            # 신뢰도 계산 시도
            print(f"  - 신뢰도 계산 시도...")
            confidence = agent._calculate_confidence_from_direction_accuracy()
            print(f"     confidence 결과: {confidence}")
            print(f"     confidence type: {type(confidence)}")
            
            # 예측 수행
            print(f"  - 예측 수행 중...")
            target = agent.predict(agent.stockdata, n_samples=10)
            
            print(f"  ✅ 예측 완료:")
            print(f"     - confidence: {target.confidence}")
            print(f"     - confidence type: {type(target.confidence)}")
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("디버깅 완료")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="신뢰도 계산 디버깅")
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="테스트할 티커 (기본값: AAPL)"
    )
    
    args = parser.parse_args()
    debug_confidence_calculation(args.ticker)

