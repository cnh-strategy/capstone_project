
import os
import sys
from datetime import datetime

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈 임포트가 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from agents.debate_agent import DebateAgent
except ImportError as e:
    print(f"Error importing DebateAgent: {e}")
    sys.exit(1)

def main():
    ticker = "RZLV"
    rounds = 2 # 테스트 목적이므로 라운드 수는 적게 설정 (기본값은 3일 수 있음)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Debate for {ticker} with {rounds} rounds...")

    try:
        # DebateAgent 인스턴스 생성
        debate = DebateAgent(ticker=ticker, rounds=rounds)
        
        # 토론 실행
        result = debate.run()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Debate Finished Successfully.")
        print("Final Ensemble Result:")
        print(result)

    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error occurred during debate execution:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

