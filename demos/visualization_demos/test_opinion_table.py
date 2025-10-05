#!/usr/bin/env python3
"""
투자의견 표 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import DebateVisualizer
from dataclasses import dataclass
from typing import List

@dataclass
class MockOpinion:
    agent_id: str
    target: object
    reason: str

@dataclass
class MockTarget:
    next_close: float

@dataclass
class MockLog:
    round_no: int
    opinions: List[MockOpinion]
    rebuttals: List

def create_test_data():
    """테스트 데이터 생성"""
    logs = [
        MockLog(1, [
            MockOpinion("SentimentalAgent", MockTarget(100.0), "긍정적"),
            MockOpinion("TechnicalAgent", MockTarget(105.0), "상승 추세"),
            MockOpinion("FundamentalAgent", MockTarget(98.0), "가치 평가")
        ], []),
        MockLog(2, [
            MockOpinion("SentimentalAgent", MockTarget(102.0), "여론 개선"),
            MockOpinion("TechnicalAgent", MockTarget(108.0), "모멘텀 강화"),
            MockOpinion("FundamentalAgent", MockTarget(99.0), "안정적")
        ], []),
        MockLog(3, [
            MockOpinion("SentimentalAgent", MockTarget(101.0), "중립적"),
            MockOpinion("TechnicalAgent", MockTarget(110.0), "강한 신호"),
            MockOpinion("FundamentalAgent", MockTarget(100.0), "균형")
        ], [])
    ]
    
    final = {
        'agents': {
            'SentimentalAgent': 101.0,
            'TechnicalAgent': 110.0,
            'FundamentalAgent': 100.0
        },
        'mean_next_close': 103.67,
        'median_next_close': 101.0,
        'currency': 'USD',
        'current_price': 100.0
    }
    
    return logs, final

if __name__ == "__main__":
    print("🧪 투자의견 표 테스트 시작")
    
    # 테스트 데이터 생성
    logs, final = create_test_data()
    
    # 시각화 객체 생성
    visualizer = DebateVisualizer()
    
    print("📊 투자의견 표 생성 중...")
    try:
        visualizer.plot_opinion_table(logs, final)
        print("✅ 투자의견 표 생성 완료!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
