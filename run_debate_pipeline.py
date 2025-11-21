# run_debate_pipeline.py
"""
뉴스 + 가격 기반 LSTM 예측까지 끝난 SentimentalAgent / TechnicalAgent / MacroAgent들을
DebateAgent 구조로 묶어서

1) Opinion
2) Rebuttal
3) Revision

을 한 번에 실행하고 콘솔에 예쁘게 출력하는 구동 코드.
"""

import os
import sys
from pprint import pprint

# 1) 프로젝트 루트 기준으로 import 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.debate_agent import DebateAgent  # 이미 있는 DebateAgent 사용


def run_debate_for_ticker(ticker: str, round_id: int = 1) -> None:
    """
    하나의 티커에 대해 DebateAgent 전체 파이프라인 실행:
    - get_opinion(round_id)
    - get_rebuttal(round_id)
    - get_revise(round_id)

    각각의 결과를 콘솔에 보기 좋게 출력한다.
    """

    print("=" * 80)
    print(f"🎯 Debate 파이프라인 시작 – Ticker: {ticker}, Round: {round_id}")
    print("=" * 80)

    # 1) DebateAgent 생성
    debate = DebateAgent(ticker=ticker)

    # ----------------------------------------------------------------------
    # 2) Opinion 단계
    # ----------------------------------------------------------------------
    print("\n[1] 🧠 Opinion 생성 중...\n")
    # ✅ 여기서 round_id 넘겨주기 (에러 원인)
    opinions = debate.get_opinion(round_id)

    if isinstance(opinions, dict):
        for agent_id, op in opinions.items():
            print("-" * 80)
            print(f"[Opinion - Round {round_id}] Agent: {agent_id}")
            print("-" * 80)
            print(op)
            print()
    else:
        print(">>> get_opinion() 결과:")
        pprint(opinions)

    # ----------------------------------------------------------------------
    # 3) Rebuttal 단계
    # ----------------------------------------------------------------------
    print("\n[2] ⚔️ Rebuttal 생성 중...\n")
    rebuttals = debate.get_rebuttal(round_id)

    if isinstance(rebuttals, dict):
        for agent_id, rb in rebuttals.items():
            print("-" * 80)
            print(f"[Rebuttal - Round {round_id}] Agent: {agent_id}")
            print("-" * 80)
            print(rb)
            print()
    else:
        print(">>> get_rebuttal() 결과:")
        pprint(rebuttals)

    # ----------------------------------------------------------------------
    # 4) Revision 단계
    # ----------------------------------------------------------------------
    print("\n[3] ✅ Revision(최종 의견) 생성 중...\n")
    revisions = debate.get_revise(round_id)

    if isinstance(revisions, dict):
        for agent_id, rv in revisions.items():
            print("-" * 80)
            print(f"[Revision - Round {round_id}] Agent: {agent_id}")
            print("-" * 80)
            print(rv)
            print()
    else:
        print(">>> get_revise() 결과:")
        pprint(revisions)

    print("=" * 80)
    print(f"🎉 Debate 파이프라인 완료 – Ticker: {ticker}, Round: {round_id}")
    print("=" * 80)


if __name__ == "__main__":
    # 예) python run_debate_pipeline.py NVDA 2
    #     → NVDA에 대해 round_id = 2로 debate 실행
    if len(sys.argv) >= 2:
        ticker = sys.argv[1]
    else:
        ticker = "NVDA"

    if len(sys.argv) >= 3:
        try:
            round_id = int(sys.argv[2])
        except ValueError:
            round_id = 1
    else:
        round_id = 1

    run_debate_for_ticker(ticker, round_id=round_id)
