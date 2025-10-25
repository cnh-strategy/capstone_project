#!/usr/bin/env python3
"""
MVP 하이브리드 주식 예측 시스템
- ML 예측만 (LLM 토론 제거)
- LLM은 ML 결과 해석만
- 핵심 기능만 유지
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Agent import
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MVPHybridSystem:
    """MVP 하이브리드 시스템"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        
        # 에이전트 생성 (ML만 사용)
        self.agents = self._create_agents()
        
    def _create_agents(self) -> List:
        """에이전트 생성"""
        agents = []
        
        try:
            # Technical Agent
            tech_agent = TechnicalAgent(
                agent_id="TechnicalAgent",
                use_ml_modules=True,
                verbose=True
            )
            agents.append(tech_agent)
            
            # Fundamental Agent
            fund_agent = FundamentalAgent(
                agent_id="FundamentalAgent", 
                use_ml_modules=True,
                verbose=True
            )
            agents.append(fund_agent)
            
            # Sentimental Agent
            sent_agent = SentimentalAgent(
                agent_id="SentimentalAgent",
                use_ml_modules=True,
                verbose=True
            )
            agents.append(sent_agent)
            
            logger.info("✅ 모든 에이전트 생성 완료")
            
        except Exception as e:
            logger.error(f"❌ 에이전트 생성 실패: {str(e)}")
        
        return agents
    
    def step1_data_search(self) -> Dict[str, str]:
        """1단계: 데이터 수집"""
        logger.info(f"🔍 1단계: {self.ticker} 데이터 수집 시작...")
        
        results = {}
        
        for agent in self.agents:
            try:
                if hasattr(agent, 'search_data'):
                    filepath = agent.search_data(self.ticker)
                    results[agent.agent_id] = filepath
                    logger.info(f"✅ {agent.agent_id}: {filepath}")
                else:
                    logger.warning(f"⚠️ {agent.agent_id}: search_data 메서드 없음")
                    results[agent.agent_id] = None
            except Exception as e:
                logger.error(f"❌ {agent.agent_id} 데이터 수집 실패: {str(e)}")
                results[agent.agent_id] = None
        
        success_count = sum(1 for path in results.values() if path is not None)
        logger.info(f"📊 데이터 수집 완료: {success_count}/{len(self.agents)} 성공")
        
        return results
    
    def step2_model_training(self, force_retrain: bool = False) -> Dict[str, bool]:
        """2단계: 모델 학습 (선택사항)"""
        logger.info(f"🎯 2단계: {self.ticker} 모델 학습 시작...")
        
        results = {}
        
        for agent in self.agents:
            try:
                if hasattr(agent, 'train_model'):
                    success = agent.train_model(self.ticker)
                    results[agent.agent_id] = success
                    if success:
                        logger.info(f"✅ {agent.agent_id}: 학습 완료")
                    else:
                        logger.warning(f"⚠️ {agent.agent_id}: 학습 실패")
                else:
                    logger.warning(f"⚠️ {agent.agent_id}: train_model 메서드 없음")
                    results[agent.agent_id] = False
            except Exception as e:
                logger.error(f"❌ {agent.agent_id} 모델 학습 실패: {str(e)}")
                results[agent.agent_id] = False
        
        success_count = sum(results.values())
        logger.info(f"📊 모델 학습 완료: {success_count}/{len(self.agents)} 성공")
        
        return results
    
    def step3_prediction(self) -> Dict[str, Any]:
        """3단계: ML 예측"""
        logger.info(f"🎯 3단계: {self.ticker} ML 예측 시작...")
        
        predictions = {}
        
        for agent in self.agents:
            try:
                if hasattr(agent, 'predict_price'):
                    pred, uncertainty = agent.predict_price(self.ticker)
                    predictions[agent.agent_id] = {
                        'prediction': pred,
                        'uncertainty': uncertainty,
                        'beta': getattr(agent, 'beta_value', 0.5)
                    }
                    logger.info(f"📊 {agent.agent_id}: {pred:.2f} (불확실성: {uncertainty:.4f})")
                else:
                    logger.warning(f"⚠️ {agent.agent_id}: predict_price 메서드 없음")
                    predictions[agent.agent_id] = {
                        'prediction': 0.0,
                        'uncertainty': 1.0,
                        'beta': 0.5
                    }
            except Exception as e:
                logger.error(f"❌ {agent.agent_id} 예측 실패: {str(e)}")
                predictions[agent.agent_id] = {
                    'prediction': 0.0,
                    'uncertainty': 1.0,
                    'beta': 0.5
                }
        
        # 최종 합의 계산
        consensus = self._calculate_consensus(predictions)
        
        result = {
            'success': True,
            'predictions': predictions,
            'consensus': consensus,
            'ticker': self.ticker
        }
        
        logger.info(f"✅ {self.ticker} ML 예측 완료: 합의={consensus:.4f}")
        
        return result
    
    def step4_llm_interpretation(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """4단계: LLM 해석 (ML 결과에 대한 의견만)"""
        logger.info(f"💭 4단계: {self.ticker} LLM 해석 시작...")
        
        try:
            # 간단한 LLM 해석 생성
            interpretation = self._generate_simple_interpretation(ml_results)
            
            return {
                'success': True,
                'interpretation': interpretation,
                'ticker': self.ticker
            }
            
        except Exception as e:
            logger.error(f"❌ LLM 해석 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'interpretation': "해석 생성 실패"
            }
    
    def _calculate_consensus(self, predictions: Dict[str, Dict]) -> float:
        """최종 합의 계산"""
        weights = {
            'TechnicalAgent': 0.4,
            'FundamentalAgent': 0.35,
            'SentimentalAgent': 0.25
        }
        
        consensus = 0.0
        total_weight = 0.0
        
        for agent_id, pred_data in predictions.items():
            weight = weights.get(agent_id, 0.0)
            prediction = pred_data['prediction']
            consensus += prediction * weight
            total_weight += weight
        
        if total_weight > 0:
            consensus /= total_weight
        
        return consensus
    
    def _generate_simple_interpretation(self, ml_results: Dict[str, Any]) -> str:
        """간단한 LLM 해석 생성"""
        consensus = ml_results.get('consensus', 0.0)
        predictions = ml_results.get('predictions', {})
        
        # 각 에이전트별 예측 요약
        agent_summaries = []
        for agent_id, pred_data in predictions.items():
            pred = pred_data['prediction']
            uncertainty = pred_data['uncertainty']
            beta = pred_data['beta']
            
            agent_name = agent_id.replace('Agent', '')
            agent_summaries.append(f"{agent_name}: ${pred:.2f} (신뢰도: {beta:.3f})")
        
        interpretation = f"""
📊 {self.ticker} 주식 예측 분석 결과

🎯 최종 예측: ${consensus:.2f}

📈 에이전트별 예측:
{chr(10).join(agent_summaries)}

💡 분석 의견:
- 기술적 분석: 차트 패턴과 거래량을 기반으로 한 예측
- 펀더멘털 분석: 재무 지표와 기업 가치를 기반으로 한 예측  
- 감정 분석: 시장 심리와 뉴스 감정을 기반으로 한 예측

⚠️ 주의사항: 이 예측은 과거 데이터를 기반으로 한 것으로, 실제 투자 결정 시 추가적인 분석이 필요합니다.
"""
        
        return interpretation
    
    def run_full_analysis(self, force_retrain: bool = False) -> Dict[str, Any]:
        """전체 분석 실행"""
        logger.info(f"🚀 {self.ticker} MVP 분석 시작...")
        logger.info("=" * 60)
        
        results = {
            'ticker': self.ticker,
            'data_search': {},
            'model_training': {},
            'ml_prediction': {},
            'llm_interpretation': {},
            'final_result': {},
            'timestamp': None
        }
        
        # 1단계: 데이터 수집
        results['data_search'] = self.step1_data_search()
        
        # 2단계: 모델 학습 (선택사항)
        results['model_training'] = self.step2_model_training(force_retrain)
        
        # 3단계: ML 예측
        results['ml_prediction'] = self.step3_prediction()
        
        # 4단계: LLM 해석
        if results['ml_prediction']['success']:
            results['llm_interpretation'] = self.step4_llm_interpretation(results['ml_prediction'])
        
        # 최종 결과
        results['final_result'] = {
            'prediction': results['ml_prediction'].get('consensus', 0.0),
            'interpretation': results['llm_interpretation'].get('interpretation', '해석 없음'),
            'confidence': 'medium'
        }
        
        logger.info("=" * 60)
        logger.info("🎉 MVP 분석 완료!")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """결과 출력"""
        print(f"\n📊 {self.ticker} MVP 분석 결과")
        print("=" * 60)
        
        # 데이터 수집 결과
        print("\n🔍 1단계: 데이터 수집")
        for agent_id, filepath in results['data_search'].items():
            if filepath:
                print(f"✅ {agent_id}: {filepath}")
            else:
                print(f"❌ {agent_id}: 실패")
        
        # 모델 학습 결과
        print("\n🎯 2단계: 모델 학습")
        for agent_id, success in results['model_training'].items():
            if success:
                print(f"✅ {agent_id}: 완료")
            else:
                print(f"❌ {agent_id}: 실패")
        
        # ML 예측 결과
        print("\n📈 3단계: ML 예측")
        if results['ml_prediction']['success']:
            print(f"✅ 최종 예측: ${results['ml_prediction']['consensus']:.2f}")
            for agent_id, pred_data in results['ml_prediction']['predictions'].items():
                print(f"  • {agent_id}: ${pred_data['prediction']:.2f} (β: {pred_data['beta']:.3f})")
        else:
            print(f"❌ ML 예측 실패")
        
        # LLM 해석 결과
        print("\n💭 4단계: LLM 해석")
        if results['llm_interpretation']['success']:
            print("✅ 해석 완료")
            print(results['llm_interpretation']['interpretation'])
        else:
            print(f"❌ 해석 실패: {results['llm_interpretation']['error']}")
        
        # 최종 결과
        print("\n🎯 최종 결과")
        final_result = results['final_result']
        print(f"✅ 예측: ${final_result['prediction']:.2f}")
        print(f"📝 해석: {final_result['interpretation'][:100]}...")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='MVP 하이브리드 주식 예측 시스템')
    parser.add_argument('--ticker', type=str, default='RZLV', help='주식 티커 (기본값: RZLV)')
    parser.add_argument('--force-retrain', action='store_true', help='모델 강제 재학습')
    parser.add_argument('--step', type=str, choices=['search', 'train', 'predict', 'interpret', 'all'], 
                       default='all', help='실행할 단계')
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = MVPHybridSystem(args.ticker)
    
    if args.step == 'search':
        # 1단계만 실행
        results = system.step1_data_search()
        print(f"\n📊 {args.ticker} 데이터 수집 결과:")
        for agent_id, filepath in results.items():
            if filepath:
                print(f"✅ {agent_id}: {filepath}")
            else:
                print(f"❌ {agent_id}: 실패")
    
    elif args.step == 'train':
        # 2단계만 실행
        results = system.step2_model_training(args.force_retrain)
        print(f"\n📊 {args.ticker} 모델 학습 결과:")
        for agent_id, success in results.items():
            if success:
                print(f"✅ {agent_id}: 완료")
            else:
                print(f"❌ {agent_id}: 실패")
    
    elif args.step == 'predict':
        # 3단계만 실행
        result = system.step3_prediction()
        if result['success']:
            print(f"\n📊 {args.ticker} ML 예측 결과:")
            print(f"✅ 최종 예측: ${result['consensus']:.2f}")
            for agent_id, pred_data in result['predictions'].items():
                print(f"  • {agent_id}: ${pred_data['prediction']:.2f}")
        else:
            print(f"❌ 예측 실패")
    
    elif args.step == 'interpret':
        # 4단계만 실행
        ml_result = system.step3_prediction()
        result = system.step4_llm_interpretation(ml_result)
        if result['success']:
            print(f"✅ 해석 완료")
            print(result['interpretation'])
        else:
            print(f"❌ 해석 실패: {result['error']}")
    
    else:  # all
        # 전체 분석 실행
        results = system.run_full_analysis(args.force_retrain)
        system.print_results(results)


if __name__ == "__main__":
    main()
