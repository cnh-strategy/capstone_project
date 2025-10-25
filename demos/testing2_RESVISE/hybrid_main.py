#!/usr/bin/env python3
"""
디베이팅 통합 코드 
"""

import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import streamlit as st

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 기존 LLM 시스템 import
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent
from agents.technical_agent import TechnicalAgent

# ML 시스템 import
try:
    from ml_models.agent_utils import AgentLoader
    from ml_models.debate_system import DebateSystem as MLDebateSystem
    from ml_models.stage2_trainer import Stage2Trainer
    from ml_models.train_agents import MLModelTrainer
    ML_AVAILABLE = True
    print("✅ ML 모듈 로드 성공")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML 모듈을 사용할 수 없습니다: {e}")
    print("LLM 모드만 사용됩니다.")


@dataclass
# 에이전트 설정 클래스
class AgentConfig:
    name: str
    agent_class: type
    prediction_range: tuple  # (min_ratio, max_ratio) - 현재가 대비 비율
    personality: str  # 에이전트 성격 설명
    analysis_focus: str  # 분석 초점

# 하이브리드 토론 시스템 메인 클래스
class HybridDebateSystem:
    # 초기화
    def __init__(self):

        # 에이전트 설정
        self.agent_configs = self._setup_agent_configs()
           
        # ML 시스템 초기화
        try:
            # ML 모델 로드
            self.ml_loader = AgentLoader('models')
            # ML 디베이팅 시스템 로드 
            self.ml_debate_system = MLDebateSystem('models')
            # ML 스테이지 2 트레이너 로드 
            self.stage2_trainer = Stage2Trainer('models')
            print("✅ ML 시스템 초기화 완료")

        except Exception as e:
            print(f"❌ ML 시스템 초기화 실패: {e}")
    
    # 에이전트 설정 초기화
    def _setup_agent_configs(self) -> Dict[str, AgentConfig]:
        """LLM 에이전트 설정 초기화"""
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
                personality='적극적인 센티멘탈 분석가',
                analysis_focus='시장 심리와 뉴스 감성에 기반한 동적이고 반응적인 예측'
            ),
            'technical': AgentConfig(
                name='TechnicalAgent',
                agent_class=TechnicalAgent,
                prediction_range=(0.92, 1.08),  # ±8% 범위
                personality='체계적인 기술적 분석가',
                analysis_focus='차트 패턴과 기술적 지표에 기반한 정확하고 논리적인 예측'
            )
        }

    # ML 기반 예측 실행 (Stage)
    def run_ml_prediction(self, ticker: str) -> Dict[str, Any]:
        try:
            print(f"# ML 모델로 {ticker} 예측 시작...")
            
            # ML 모델 로드
            print("## ML 에이전트 로딩 중...")
            self.ml_loader.load_all_agents()
            print(f"✅ {len(self.ml_loader.agents)} 개 ML 에이전트 로드 완료")
            
            # 데이터 로드 (최근 1주일 데이터)
            print("## 최근 데이터 로딩 중...")
            data_dict = self._load_recent_data_for_debate(ticker)
            if not data_dict:
                return {'success': False, 'error': '최근 데이터 로드 실패'}
            
            # 예측 실행
            print("## ML 예측 실행 중...")
            ml_results = self.ml_debate_system.online_debate_prediction(data_dict)
            
            return {
                'success': True,
                'predictions': ml_results,
                'message': f"ML 모델 예측 완료",
                'agents_loaded': len(self.ml_loader.agents)
            }

        except Exception as e:
            print(f"# ML 예측 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'message': f"ML 예측 실패: {e}"
            }
    
    # 최근 1주일 데이터를 로드하여 디베이팅용 데이터 준비
    def load_data_for_debate(self, ticker, days=7):
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            
            # 최근 1주일 데이터 로드
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data_dict = {}
            
            # 각 에이전트별 테스트 데이터 로드
            for agent_type in ['technical', 'fundamental', 'sentimental']:

                # data 폴더에서 테스트 데이터 로드
                test_file = f"data/{ticker}_{agent_type}_test.csv"

                # 테스트 데이터 로드
                try:
                    df = pd.read_csv(test_file)
                    if not df.empty:
                        df['Date'] = pd.to_datetime(df['Date'])
                        # 최근 1주일 필터링
                        recent_df = df[df['Date'] >= start_date.strftime('%Y-%m-%d')]
                        if not recent_df.empty:
                            data_dict[agent_type] = recent_df
                            print(f"✅ {agent_type}: {len(recent_df)}개 최근 샘플 로드")
                        else:
                            print(f"⚠️ {agent_type}: 최근 데이터 없음, 전체 테스트 데이터 사용")
                            data_dict[agent_type] = df
                    else:
                        print(f"⚠️ {agent_type}: 빈 테스트 파일")

                except FileNotFoundError:
                    print(f"⚠️ {agent_type}: 테스트 파일 없음 - {test_file}")

                except Exception as e:
                    print(f"⚠️ {agent_type}: 데이터 로드 오류 - {e}")
            
            if not data_dict:
                print("# 디베이팅용 데이터 없음")
                return None
            
            print(f"# 디베이팅용 데이터 로드 완료: {list(data_dict.keys())}")
            return data_dict
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None
    
    # ML 결과에 대한 LLM 해석 제공
    def interpret_ml_results(self, ml_results: Dict, ticker: str) -> Dict:
        """ML 결과에 대한 LLM 해석 제공"""
        try:
            print(f"🧠 ML 결과 해석 시작: {ticker}")
            
            # ML 결과에서 예측값 추출
            ml_prediction = None
            if ml_results.get('success') and ml_results.get('predictions'):
                predictions = ml_results['predictions']
                if isinstance(predictions, tuple) and len(predictions) >= 2:
                    ml_prediction = predictions[1].get('consensus', 0.0)
            
            if ml_prediction is None:
                return {
                    'success': False,
                    'error': 'ML 예측 결과를 찾을 수 없습니다.',
                    'ticker': ticker
                }
            
            # 간단한 해석 생성
            interpretation = self._generate_interpretation(ml_prediction, ticker)
            
            return {
                'success': True,
                'interpretation': interpretation,
                'ml_prediction': ml_prediction,
                'ticker': ticker
            }
            
        except Exception as e:
            print(f"❌ ML 결과 해석 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
    
    # ML 예측에 대한 해석 생성
    def generate_interpretation(self, ml_prediction: float, ticker: str) -> str:
        if ml_prediction > 10:
            trend = "상승"
            confidence = "높음"
        elif ml_prediction > 5:
            trend = "약간 상승"
            confidence = "보통"
        elif ml_prediction > 0:
            trend = "소폭 상승"
            confidence = "낮음"
        else:
            trend = "하락"
            confidence = "높음"
        
        return f"""
                📊 {ticker} ML 예측 해석:

                🎯 예측값: ${ml_prediction:.2f}
                📈 추세: {trend}
                🎲 신뢰도: {confidence}""".strip()
    
    # 분석 실행
    def run_analysis(self, ticker: str, rounds: int = 3) -> Dict[str, Any]:
        print(f"# 분석 시작: {ticker}")
        print("=" * 60)
        
        results = {
            'ticker': ticker,
            'ml_results': {},
            'llm_results': {},
            'consensus': {},
            'timestamp': None
        }
        
        # 1. ML 기반 예측
        if self.use_ml_modules:
            print("📊 1단계: ML 모델 예측")
            ml_results = self.run_ml_prediction(ticker)
            results['ml_results'] = ml_results
            
            if ml_results.get('success'):
                print("✅ ML 예측 성공")
            else:
                print("⚠️ ML 예측 실패, LLM만 사용")
        
        # 2. LLM 기반 ML 결과 해석
        if self.use_llm_debate and results['ml_results'].get('success'):
            print("\n🧠 2단계: ML 결과 해석")
            llm_results = self.interpret_ml_results(results['ml_results'], ticker)
            results['llm_results'] = llm_results
            
            if llm_results.get('success'):
                print("✅ ML 결과 해석 성공")
            else:
                print("⚠️ ML 결과 해석 실패")
        
        # 3. 최종 합의 도출
        print("\n🎯 3단계: 최종 합의 도출")
        consensus = self._generate_consensus(results)
        results['consensus'] = consensus
        
        print("=" * 60)
        print("🎉 하이브리드 분석 완료!")
        
        return results
    
    # 최종 합의 도출
    def _generate_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """최종 합의 도출"""
        consensus = {
            'method': 'hybrid',
            'ml_weight': 0.4 if results['ml_results'].get('success') else 0.0,
            'llm_weight': 0.6 if results['llm_results'].get('success') else 1.0,
            'final_prediction': None,
            'confidence': 'medium',
            'reasoning': []
        }
        
        # ML 예측 결과 처리
        if results['ml_results'].get('success'):
            ml_predictions = results['ml_results'].get('predictions', {})
            if ml_predictions:
                consensus['reasoning'].append("ML 모델의 정량적 예측을 기반으로 함")
        
        # LLM 해석 결과 처리
        if results['llm_results'].get('success'):
            llm_interpretation = results['llm_results'].get('interpretation', '')
            if llm_interpretation:
                consensus['reasoning'].append("LLM 해석을 통한 정성적 분석 추가")
        
        # 최종 예측 가격 계산 (ML과 LLM 결과 종합)
        ml_prediction = None
        llm_prediction = None
        
        # ML 예측값 추출
        if results['ml_results'].get('success'):
            ml_predictions = results['ml_results'].get('predictions', {})
            if isinstance(ml_predictions, tuple) and len(ml_predictions) >= 2:
                ml_final = ml_predictions[1]
                ml_prediction = ml_final.get('consensus')
        
        # 최종 예측 계산 (ML 결과만 사용)
        if ml_prediction is not None:
            # ML 예측값을 최종 결과로 사용
            final_prediction = ml_prediction
            consensus['reasoning'].append(f"ML 예측값 사용: {ml_prediction:.2f}")
            consensus['reasoning'].append("LLM은 해석 제공용으로만 활용")
        else:
            # ML 예측 실패한 경우
            final_prediction = None
            consensus['reasoning'].append("ML 예측 실패")
        
        consensus['final_prediction'] = final_prediction
        
        return consensus
    
    # 시각화 옵션 표시
    def show_visualization_options(self, results: Dict[str, Any]):
        """시각화 옵션 표시"""
        print("\n📊 시각화 옵션:")
        print("1. ML 모델 성능 분석")
        print("2. LLM 토론 과정 시각화")
        print("3. 하이브리드 결과 비교")
        print("4. 통합 대시보드")

# 메인 실행 함수
def main():
    """메인 실행 함수"""
    print("🚀 Hybrid Multi-Agent Debate System")
    print("=" * 60)
    
    # 시스템 초기화
    system = HybridDebateSystem()
    
    # 사용자 티커 입력 
    ticker = input("분석할 주식 티커를 입력하세요 (예: AAPL, TSLA): ").upper().strip()
    if not ticker:
        ticker = "AAPL"
        print(f"기본값 사용: {ticker}")
    
    # 사용자 라운드 수 입력 
    rounds = input("토론 라운드 수를 입력하세요 (기본=3): ").strip()
    try:
        rounds = int(rounds) if rounds else 3
    except ValueError:
        rounds = 3
        print(f"기본값 사용: {rounds}라운드")
    
    # 분석 실행
    results = system.run_hybrid_analysis(ticker, rounds)
    
    # 결과 출력
    print("\n📋 분석 결과 요약:")
    print(f"종목: {results['ticker']}")
    print(f"ML 예측: {'성공' if results['ml_results'].get('success') else '실패'}")
    print(f"최종 예측: {results['consensus'].get('final_prediction', 'N/A')}")
    
    # 시각화 옵션
    system.show_visualization_options(results)
    
    return results

# 메인 실행 함수
if __name__ == "__main__":
    main()
