# demo_visualization.py
"""
시각화 기능 데모 스크립트
실제 토론 없이도 시각화 기능을 테스트할 수 있습니다.
"""

from visualization import DebateVisualizer
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class MockOpinion:
    agent_id: str
    target: object
    reason: str

@dataclass
class MockTarget:
    next_close: float

@dataclass
class MockRebuttal:
    from_agent_id: str
    to_agent_id: str
    stance: str
    message: str

@dataclass
class MockLog:
    round_no: int
    opinions: List[MockOpinion]
    rebuttals: List[MockRebuttal]

def create_mock_data(ticker: str = "AAPL", rounds: int = 3) -> tuple:
    """실제 토론 데이터와 유사한 모의 데이터 생성"""
    
    # 초기 가격 설정
    base_price = 150.0
    price_variations = {
        "SentimentalAgent": [0.02, -0.01, 0.03],  # 센티멘탈 변동
        "TechnicalAgent": [0.05, 0.02, -0.01],    # 기술적 변동
        "FundamentalAgent": [-0.01, 0.01, 0.02]   # 펀더멘털 변동
    }
    
    logs = []
    
    for round_num in range(1, rounds + 1):
        opinions = []
        rebuttals = []
        
        # 각 에이전트의 의견 생성
        for agent_id, variations in price_variations.items():
            # 가격에 노이즈 추가
            noise = np.random.normal(0, 0.01)
            price_change = variations[round_num - 1] + noise
            predicted_price = base_price * (1 + price_change)
            
            opinion = MockOpinion(
                agent_id=agent_id,
                target=MockTarget(next_close=predicted_price),
                reason=f"라운드 {round_num}에서 {agent_id}의 분석 결과"
            )
            opinions.append(opinion)
        
        # 반박/지지 생성 (모든 라운드에서)
        agents = list(price_variations.keys())
        for i, from_agent in enumerate(agents):
            for j, to_agent in enumerate(agents):
                if i != j:
                    stance = "SUPPORT" if np.random.random() > 0.5 else "REBUT"
                    rebuttal = MockRebuttal(
                        from_agent_id=from_agent,
                        to_agent_id=to_agent,
                        stance=stance,
                        message=f"{from_agent}가 {to_agent}에게 {stance} 메시지"
                    )
                    rebuttals.append(rebuttal)
        
        log = MockLog(
            round_no=round_num,
            opinions=opinions,
            rebuttals=rebuttals
        )
        logs.append(log)
    
    # 최종 결과 생성
    final_opinions = logs[-1].opinions
    final_prices = [op.target.next_close for op in final_opinions]
    
    final = {
        "ticker": ticker,
        "agents": {op.agent_id: op.target.next_close for op in final_opinions},
        "mean_next_close": np.mean(final_prices),
        "median_next_close": np.median(final_prices),
        "currency": "USD",
        "last_price": base_price,
        "current_price": base_price  # plot_opinion_table에서 사용
    }
    
    return logs, final

def demo_all_visualizations():
    """모든 시각화 기능 데모"""
    print("🎨 시각화 기능 데모를 시작합니다...")
    
    # 모의 데이터 생성
    logs, final = create_mock_data("AAPL", rounds=3)
    
    # 시각화 객체 생성 (백엔드 상태 자동 출력)
    visualizer = DebateVisualizer()
    
    # 파일 저장 모드로 실행
    print("\n📁 차트는 ./demo_reports/ 디렉토리에 저장됩니다.")
    
    print("\n📊 1. 라운드별 의견 변화")
    visualizer.plot_round_progression(logs, final)
    
    print("\n📈 2. 의견 일치도 분석")
    visualizer.plot_consensus_analysis(logs, final)
    
    print("\n🕸️ 3. 반박/지지 네트워크")
    visualizer.plot_rebuttal_network(logs)
    
    print("\n📊 4. 투자의견 표")
    visualizer.plot_opinion_table(logs, final)
    
    print("\n📱 5. 인터랙티브 대시보드")
    visualizer.create_interactive_dashboard(logs, final, "AAPL")
    
    print("\n📋 6. 전체 리포트 생성")
    visualizer.generate_report(logs, final, "AAPL", save_dir="./demo_reports")
    
    print("\n✅ 모든 시각화 데모가 완료되었습니다!")

def demo_individual_visualization():
    """개별 시각화 선택 데모"""
    print("🎨 개별 시각화 데모")
    print("1. 라운드별 의견 변화")
    print("2. 의견 일치도 분석")
    print("3. 반박/지지 네트워크")
    print("4. 투자의견 표")
    print("5. 주식 컨텍스트")
    print("6. 인터랙티브 대시보드")
    print("7. 전체 리포트")
    
    choice = input("선택하세요 (1-7): ").strip()
    
    logs, final = create_mock_data("AAPL", rounds=3)
    visualizer = DebateVisualizer()
    
    # 파일 저장 모드로 실행
    print("\n📁 차트는 파일로 저장됩니다.")
    
    if choice == "1":
        visualizer.plot_round_progression(logs, final)
    elif choice == "2":
        visualizer.plot_consensus_analysis(logs, final)
    elif choice == "3":
        visualizer.plot_rebuttal_network(logs)
    elif choice == "4":
        visualizer.plot_opinion_table(logs, final)
    elif choice == "5":
        visualizer.plot_stock_context("AAPL")
    elif choice == "6":
        visualizer.create_interactive_dashboard(logs, final, "AAPL")
    elif choice == "7":
        visualizer.generate_report(logs, final, "AAPL")
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    print("🚀 시각화 데모 스크립트")
    print("1. 모든 시각화 데모")
    print("2. 개별 시각화 선택")
    
    mode = input("모드를 선택하세요 (1-2): ").strip()
    
    if mode == "1":
        demo_all_visualizations()
    elif mode == "2":
        demo_individual_visualization()
    else:
        print("잘못된 선택입니다.")
