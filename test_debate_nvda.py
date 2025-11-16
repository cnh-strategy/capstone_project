#!/usr/bin/env python3
"""
NVDA 전체 디베이트 실행 및 문제점 분석 스크립트
"""
import sys
import os
from datetime import datetime
from pathlib import Path
import traceback

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.debate_agent import DebateAgent

def test_debate_nvda():
    """NVDA에 대해 전체 디베이트 실행 및 문제점 분석"""
    print("=" * 80)
    print("NVDA 전체 디베이트 실행 및 문제점 분석")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    ticker = "NVDA"
    rounds = 2  # 테스트용으로 2라운드만
    
    try:
        # 1. DebateAgent 초기화
        print("1️⃣ DebateAgent 초기화...")
        debate = DebateAgent(rounds=rounds, ticker=ticker)
        print("✅ 초기화 완료\n")
        
        # 2. 데이터셋 빌드 (선택적 - 이미 있으면 스킵)
        print("2️⃣ 데이터셋 확인...")
        from core.data_set import build_dataset
        try:
            build_dataset(ticker)
            print("✅ 데이터셋 준비 완료\n")
        except Exception as e:
            print(f"⚠️ 데이터셋 빌드 중 오류 (계속 진행): {e}\n")
        
        # 3. Round 0: 초기 의견 수집
        print("3️⃣ Round 0: 초기 의견 수집...")
        print("-" * 80)
        try:
            opinions_0 = debate.get_opinion(round=0, ticker=ticker, rebuild=False, force_pretrain=False)
            print("\n✅ Round 0 완료")
            print(f"   수집된 의견 수: {len(opinions_0)}")
            for agent_id, opinion in opinions_0.items():
                if opinion and opinion.target:
                    print(f"   - {agent_id}: ${opinion.target.next_close:.2f}")
            print()
        except Exception as e:
            print(f"\n❌ Round 0 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            return
        
        # 4. Round 1: 반박 및 수정
        print("4️⃣ Round 1: 반박 및 의견 수정...")
        print("-" * 80)
        try:
            # 4-1. 반박 생성
            print("\n4-1. 반박 생성...")
            rebuttals_1 = debate.get_rebuttal(round=1)
            print(f"✅ 반박 생성 완료: {len(rebuttals_1)}개")
            for rebut in rebuttals_1[:3]:  # 처음 3개만 출력
                print(f"   - {rebut.from_agent_id} → {rebut.to_agent_id}: {rebut.stance}")
            
            # 4-2. 의견 수정
            print("\n4-2. 의견 수정...")
            revised_1 = debate.get_revise(round=1)
            print(f"✅ 의견 수정 완료: {len(revised_1)}개")
            for agent_id, opinion in revised_1.items():
                if opinion and opinion.target:
                    print(f"   - {agent_id}: ${opinion.target.next_close:.2f}")
            print()
        except Exception as e:
            print(f"\n❌ Round 1 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            return
        
        # 5. Round 2: 반박 및 수정
        if rounds >= 2:
            print("5️⃣ Round 2: 반박 및 의견 수정...")
            print("-" * 80)
            try:
                # 5-1. 반박 생성
                print("\n5-1. 반박 생성...")
                rebuttals_2 = debate.get_rebuttal(round=2)
                print(f"✅ 반박 생성 완료: {len(rebuttals_2)}개")
                
                # 5-2. 의견 수정
                print("\n5-2. 의견 수정...")
                revised_2 = debate.get_revise(round=2)
                print(f"✅ 의견 수정 완료: {len(revised_2)}개")
                for agent_id, opinion in revised_2.items():
                    if opinion and opinion.target:
                        print(f"   - {agent_id}: ${opinion.target.next_close:.2f}")
                print()
            except Exception as e:
                print(f"\n❌ Round 2 실패: {type(e).__name__}: {e}")
                traceback.print_exc()
                return
        
        # 6. 최종 결과
        print("6️⃣ 최종 결과 집계...")
        print("-" * 80)
        try:
            ensemble = debate.get_ensemble()
            print("\n✅ 최종 결과:")
            print(f"   티커: {ensemble['ticker']}")
            print(f"   현재가: ${ensemble.get('last_price', 'N/A')}")
            print(f"   평균 예측: ${ensemble.get('mean_next_close', 'N/A'):.2f}")
            print(f"   중앙값 예측: ${ensemble.get('median_next_close', 'N/A'):.2f}")
            print("\n   에이전트별 예측:")
            for key, value in ensemble.get('agents', {}).items():
                print(f"     - {key}: ${value:.2f}")
        except Exception as e:
            print(f"\n❌ 최종 결과 집계 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("✅ 전체 디베이트 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 치명적 오류: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_debate_nvda()

