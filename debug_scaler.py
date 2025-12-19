#!/usr/bin/env python3
"""
스케일러 역변환 디버깅
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.macro_agent import MacroAgent
from config.agents import dir_info

def debug_scaler(ticker="AAPL"):
    """스케일러 역변환 테스트"""
    print("=" * 80)
    print(f"스케일러 역변환 디버깅 - Ticker: {ticker}")
    print("=" * 80)
    print()
    
    agent = MacroAgent(
        agent_id="MacroAgent",
        ticker=ticker,
        data_dir=dir_info["data_dir"],
        model_dir=dir_info["model_dir"]
    )
    
    # 스케일러 로드
    agent.scaler.load(ticker)
    
    print(f"스케일러 정보:")
    print(f"  - y_scaler 존재: {hasattr(agent.scaler, 'y_scaler') and agent.scaler.y_scaler is not None}")
    if hasattr(agent.scaler, 'y_scaler') and agent.scaler.y_scaler is not None:
        print(f"  - y_scaler 타입: {type(agent.scaler.y_scaler)}")
        print(f"  - y_scaler: {agent.scaler.y_scaler}")
    
    print()
    
    # 테스트 값들
    test_values = [-0.395430, -0.231095, -0.381342, -0.062056, 0.0, 0.5, -0.5]
    
    print("역변환 테스트:")
    print("-" * 80)
    for val in test_values:
        try:
            val_scaled = np.array([[val]])
            val_inverse = agent.scaler.inverse_y(val_scaled)[0, 0]
            print(f"  스케일링된 값: {val:8.6f} -> 역변환: {val_inverse:8.6f}")
        except Exception as e:
            print(f"  스케일링된 값: {val:8.6f} -> 역변환 실패: {e}")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="스케일러 역변환 디버깅")
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="테스트할 티커 (기본값: AAPL)"
    )
    
    args = parser.parse_args()
    debug_scaler(args.ticker)

