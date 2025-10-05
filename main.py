#!/usr/bin/env python3
"""
Multi-Agent Debate System for Stock Price Prediction
메인 진입점 - 에이전트 설정과 토론 시스템 실행
"""

import os
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from debate_agent import Debate
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent
from agents.technical_agent import TechnicalAgent


@dataclass
class AgentConfig:
    """에이전트 설정 클래스"""
    name: str
    agent_class: type
    prediction_range: tuple  # (min_ratio, max_ratio) - 현재가 대비 비율
    personality: str  # 에이전트 성격 설명
    analysis_focus: str  # 분석 초점


class DebateSystem:
    """토론 시스템 메인 클래스"""
    
    def __init__(self):
        self.agent_configs = self._setup_agent_configs()
        self.prompt_configs = self._setup_prompt_configs()
    
    def _setup_agent_configs(self) -> Dict[str, AgentConfig]:
        """에이전트 설정 초기화"""
        return {
            'fundamental': AgentConfig(
                name='FundamentalAgent',
                agent_class=FundamentalAgent,
                prediction_range=(0.95, 1.05),  # ±5% 범위
                personality='보수적인 펀더멘털 분석가',
                analysis_focus='장기 가치와 재무 건전성에 기반한 안정적이고 신중한 예측'
            ),
            'sentimental': AgentConfig(
                name='SentimentalAgent', 
                agent_class=SentimentalAgent,
                prediction_range=(0.90, 1.10),  # ±10% 범위
                personality='중립적인 센티멘탈 분석가',
                analysis_focus='시장 심리와 여론에 기반한 균형 잡힌 예측'
            ),
            'technical': AgentConfig(
                name='TechnicalAgent',
                agent_class=TechnicalAgent,
                prediction_range=(0.85, 1.15),  # ±15% 범위
                personality='공격적인 기술적 분석가',
                analysis_focus='차트 패턴과 모멘텀에 기반한 적극적이고 대담한 예측'
            )
        }
    
    def _setup_prompt_configs(self) -> Dict[str, Dict[str, str]]:
        """프롬프트 설정 초기화"""
        return {
            'fundamental': {
                'predicter_system': (
                    "너는 '보수적인 펀더멘털 분석 전문가'다. 장기 가치와 재무 건전성에 기반하여 "
                    "안정적이고 신중한 예측을 제공한다. 현재가 대비 ±5% 범위 내에서만 예측하며, "
                    "급격한 변동보다는 점진적 변화를 선호한다. "
                    "반환은 JSON {\"next_close\": number, \"reason\": string}만 허용한다."
                ),
                'rebuttal_system': (
                    "너는 '보수적인 펀더멘털(가치) 분석 전문가'다. "
                    "상대 에이전트의 주장을 신중하게 검토하고, 장기 가치와 재무 건전성 전문가의 관점에서 "
                    "내 의견(next_close, reason)과 상대 에이전트의 의견(next_close, reason)을 비교하라. "
                    "가치 가정(성장, 마진, 현금흐름, 레버리지, 밸류에이션)과 이벤트 해석이 안정적이고 일관적인지 평가하라. "
                    "특히 과도한 낙관이나 급격한 변동 예측에 대해서는 반드시 반박하고, 신중한 접근을 주장해야 한다. "
                    "'REBUT'(반박) 또는 'SUPPORT'(지지) 중 하나를 반드시 선택하고, "
                    "판단 근거(message)는 한국어 최소 4문장, 최대 5문장으로 작성하라. "
                    "숫자는 float 형태로, % 기호 없이 표현한다. "
                    "출력은 JSON 객체 {\"stance\":\"REBUT|SUPPORT\", \"message\": string}만 허용한다."
                ),
                'revision_system': (
                    "너는 '보수적인 펀더멘털(가치) 분석 전문가'다. "
                    "동료 에이전트의 주장을 신중하게 검토하고, 장기 가치와 재무 건전성 전문가의 관점에서 "
                    "내 의견(next_close, reason), 동료 의견, 받은 반박/지지를 종합해 "
                    "다음 거래일 종가(next_close)와 근거(reason)를 업데이트하라. "
                    "규칙:\n"
                    "- 밸류에이션 논리와 단기 이벤트 반영 가능성을 신중하게 고려\n"
                    "- 현재가 대비 ±5% 범위 내에서 수정 (보수적 접근)\n"
                    "- 급격한 변동보다는 점진적 변화를 선호\n"
                    "- 반드시 내 전문가적 관점에서 안정적인 논리 유지\n"
                    "출력은 JSON 객체 {\"next_close\": number, \"reason\": string}만 허용한다."
                )
            },
            'sentimental': {
                'predicter_system': (
                    "너는 '중립적인 센티멘탈 분석 전문가'다. 시장 심리와 투자자 여론을 바탕으로 "
                    "균형 잡힌 예측을 제공한다. 현재가 대비 ±10% 범위에서 예측하며, "
                    "과도한 낙관이나 비관보다는 현실적인 시장 반응을 반영한다. "
                    "반환은 JSON {\"next_close\": number, \"reason\": string}만 허용한다."
                ),
                'rebuttal_system': (
                    "너는 '중립적인 센티멘탈 분석 전문가'다. "
                    "상대 에이전트의 의견을 비판적으로 검토하고, 시장 심리와 여론 분석 전문가의 관점에서 "
                    "내 의견(next_close, reason)과 상대 에이전트의 의견(next_close, reason)을 비교하여 "
                    "여론 해석(긍/부정 비율, 이벤트 해석 등)이 현실적이고 일관적인지 판단하라. "
                    "특히 과도한 낙관이나 비관에 대해서는 반드시 반박해야 한다. "
                    "'REBUT'(반박) 또는 'SUPPORT'(지지) 중 하나를 반드시 선택하고, "
                    "판단 근거(message)는 한국어 최소 4문장, 최대 5문장으로 작성하라. "
                    "숫자는 float 형태로, % 기호 없이 표현한다. "
                    "출력은 JSON 객체 {\"stance\":\"REBUT|SUPPORT\", \"message\": string}만 허용한다."
                ),
                'revision_system': (
                    "너는 '중립적인 센티멘탈 분석 전문가'다. "
                    "동료 에이전트의 의견을 비판적으로 검토하고, 시장 심리와 여론 분석 전문가의 관점에서 "
                    "내 의견(next_close, reason), 동료 의견, 받은 반박/지지를 종합해 "
                    "다음 거래일 종가(next_close)와 근거(reason)를 업데이트하라. "
                    "규칙:\n"
                    "- 현재가 대비 ±10% 범위 내에서 수정 (중립적 접근)\n"
                    "- SUPPORT/REBUT 비중과 여론 신호(긍/부정 흐름)를 균형 있게 고려\n"
                    "- 과도한 낙관이나 비관을 경계하고 현실적인 시장 반응을 반영\n"
                    "- 반드시 내 전문가적 관점에서 일관된 논리 유지\n"
                    "출력은 JSON 객체 {\"next_close\": number, \"reason\": string}만 허용한다."
                )
            },
            'technical': {
                'predicter_system': (
                    "너는 '공격적인 기술적 분석 전문가'다. 차트 패턴과 모멘텀 지표에 기반하여 "
                    "적극적이고 대담한 예측을 제공한다. 현재가 대비 ±15% 범위에서 예측하며, "
                    "강한 신호가 있을 때는 과감한 변동을 예상한다. "
                    "반환은 JSON {\"next_close\": number, \"reason\": string}만 허용한다."
                ),
                'rebuttal_system': (
                    "너는 '공격적인 기술적 분석 전문가'다. "
                    "상대 에이전트의 의견을 비판적으로 검토하고, 차트 패턴과 모멘텀 분석 전문가의 시각에서 "
                    "내 의견(next_close, reason)과 상대 에이전트의 의견(next_close, reason)을 비교하라. "
                    "기술적 신호 해석(추세, RSI, 모멘텀, 거래량 등)이 대담하고 일관적인지 평가하라. "
                    "특히 보수적인 예측에 대해서는 반드시 반박하고, 강한 신호가 있을 때는 과감한 변동을 주장해야 한다. "
                    "'REBUT'(반박) 또는 'SUPPORT'(지지) 중 하나를 반드시 선택하고, "
                    "판단 근거(message)는 한국어 최소 4문장, 최대 5문장으로 작성하라. "
                    "숫자는 float 형태로, % 기호 없이 표현한다. "
                    "출력은 JSON 객체 {\"stance\":\"REBUT|SUPPORT\", \"message\": string}만 허용한다."
                ),
                'revision_system': (
                    "너는 '공격적인 기술적 분석 전문가'다. "
                    "동료 에이전트의 의견을 비판적으로 검토하고, 차트 패턴과 모멘텀 분석 전문가의 시각에서 "
                    "내 의견(next_close, reason), 동료 의견, 받은 반박/지지를 종합해 "
                    "다음 거래일 종가(next_close)와 근거(reason)를 업데이트하라. "
                    "규칙:\n"
                    "- 추세/강도/신호의 대담한 해석을 우선 고려\n"
                    "- 현재가 대비 ±15% 범위 내에서 수정 (공격적 접근)\n"
                    "- 강한 신호가 있을 때는 과감한 변동을 주장\n"
                    "- 반드시 내 전문가적 관점에서 적극적 수정\n"
                    "출력은 JSON 객체 {\"next_close\": number, \"reason\": string}만 허용한다."
                )
            }
        }
    
    def create_agents(self) -> Dict[str, Any]:
        """설정에 따라 에이전트 생성"""
        agents = {}
        
        for agent_type, config in self.agent_configs.items():
            agent = config.agent_class(agent_id=config.name)
            
            # 프롬프트 설정 적용
            if hasattr(agent, '_update_prompts'):
                agent._update_prompts(self.prompt_configs[agent_type])
            
            agents[config.name] = agent
        
        return agents
    
    def run_debate(self, ticker: str, rounds: int = 1) -> tuple:
        """토론 실행"""
        print(f"\n🚀 Multi-Agent Debate System 시작")
        print(f"📊 분석 대상: {ticker}")
        print(f"🔄 토론 라운드: {rounds}")
        
        # 에이전트 생성
        agents = self.create_agents()
        
        # 에이전트 정보 출력
        print(f"\n👥 참여 에이전트:")
        for agent_type, config in self.agent_configs.items():
            print(f"  - {config.personality}")
            print(f"    예측 범위: ±{int((1-config.prediction_range[0])*100)}%")
            print(f"    분석 초점: {config.analysis_focus}")
        
        # 토론 실행
        debate = Debate(agents, verbose=True)
        logs, final = debate.run(ticker=ticker, rounds=rounds)
        
        return logs, final
    
    def show_visualization_options(self, logs: List, final: Dict, ticker: str):
        """시각화 옵션 표시"""
        try:
            from visualization import DebateVisualizer
            
            visualize = input("\n시각화를 생성하시겠습니까? (y/n): ").strip().lower()
            if visualize in ['y', 'yes', '예', 'ㅇ']:
                visualizer = DebateVisualizer()
                
                print("\n=== 시각화 옵션 ===")
                print("1. 라운드별 의견 변화")
                print("2. 의견 일치도 분석") 
                print("3. 반박/지지 네트워크")
                print("4. 투자의견 표")
                print("5. 주식 컨텍스트")
                print("6. 인터랙티브 대시보드")
                print("7. 전체 리포트 생성")
                
                choice = input("선택하세요 (1-7, all=전체): ").strip().lower()
                
                if choice == '1':
                    visualizer.plot_round_progression(logs, final)
                elif choice == '2':
                    visualizer.plot_consensus_analysis(logs, final)
                elif choice == '3':
                    visualizer.plot_rebuttal_network(logs)
                elif choice == '4':
                    visualizer.plot_opinion_table(logs, final)
                elif choice == '5':
                    visualizer.plot_stock_context(ticker)
                elif choice == '6':
                    visualizer.create_interactive_dashboard(logs, final, ticker)
                elif choice == '7' or choice == 'all':
                    visualizer.generate_report(logs, final, ticker)
                else:
                    print("잘못된 선택입니다.")
                    
        except ImportError:
            print("\n시각화 모듈을 찾을 수 없습니다. visualization.py 파일을 확인하세요.")
        except Exception as e:
            print(f"\n시각화 생성 중 오류 발생: {e}")


def format_debate_summary(logs: List, final: Dict) -> str:
    """토론 결과 요약 포맷팅"""
    summary = []
    
    for log in logs:
        summary.append(f"□ round {log.round_no} :")
        for opinion in log.opinions:
            agent_name = opinion.agent_id.replace('Agent', '').lower()
            # reason이 없는 경우 기본 메시지 사용
            reason = getattr(opinion.target, 'reason', f"{agent_name} 분석 결과")
            summary.append(f"- {agent_name} : {opinion.target.next_close} / {reason}")
        summary.append("")
        summary.append("----")
        summary.append("")
    
    summary.append("□ 결론 ")
    summary.append(f" - 목표가 : {final['mean_next_close']}")
    summary.append(f" - 이유 : {final.get('reason', '토론 결과 종합')}")
    
    return "\n".join(summary)


def main():
    """메인 함수"""
    print("🎯 Multi-Agent Debate System for Stock Price Prediction")
    print("=" * 60)
    
    # 실행 모드 선택 (자동으로 Streamlit 모드 선택)
    print("\n🚀 자동으로 Streamlit 웹 대시보드를 시작합니다...")
    print("📱 브라우저에서 http://localhost:8501 에 접속하세요")
    print("⏹️  중지하려면 Ctrl+C를 누르세요")
    
    try:
        import subprocess
        import sys
        
        # Streamlit 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit 대시보드가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ Streamlit 실행 중 오류 발생: {e}")
        print("💡 streamlit이 설치되어 있는지 확인하세요: pip install streamlit")


if __name__ == "__main__":
    main()
