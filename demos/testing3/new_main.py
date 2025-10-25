#!/usr/bin/env python3
"""
새로운 하이브리드 주식 예측 시스템 - 간단한 메인
각 Agent에 ML 기능이 통합된 구조
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

# LLM 모듈 import (원래 capstone 구조)
from debate_agent import Debate

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleHybridSystem:
    """간단한 하이브리드 시스템"""
    
    def __init__(self, ticker: str, use_ml: bool = True):
        self.ticker = ticker.upper()
        self.use_ml = use_ml
        
        # 에이전트 생성
        self.agents = self._create_agents()
        
    def _create_agents(self) -> List:
        """에이전트 생성"""
        agents = []
        
        try:
            # Technical Agent
            tech_agent = TechnicalAgent(
                agent_id="TechnicalAgent",
                use_ml_modules=self.use_ml,
                verbose=True
            )
            agents.append(tech_agent)
            
            # Fundamental Agent
            fund_agent = FundamentalAgent(
                agent_id="FundamentalAgent", 
                use_ml_modules=self.use_ml,
                verbose=True
            )
            agents.append(fund_agent)
            
            # Sentimental Agent
            sent_agent = SentimentalAgent(
                agent_id="SentimentalAgent",
                use_ml_modules=self.use_ml,
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
        """3단계: 예측"""
        logger.info(f"🎯 3단계: {self.ticker} 예측 시작...")
        
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
        
        logger.info(f"✅ {self.ticker} 예측 완료: 합의={consensus:.4f}")
        
        return result
    
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
    
    def step4_debate(self, rounds: int = 3) -> Dict[str, Any]:
        """4단계: 토론"""
        logger.info(f"💬 4단계: {self.ticker} 토론 시작...")
        
        try:
            # 토론 실행
            debate = Debate(self.agents, verbose=True)
            logs, final = debate.run(self.ticker, rounds)
            
            return {
                'success': True,
                'logs': logs,
                'final': final,
                'ticker': self.ticker,
                'rounds': rounds
            }
            
        except Exception as e:
            logger.error(f"❌ 토론 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'logs': [],
                'final': {}
            }
    
    def run_full_analysis(self, force_retrain: bool = False, debate_rounds: int = 3) -> Dict[str, Any]:
        """전체 분석 실행"""
        logger.info(f"🚀 {self.ticker} 전체 분석 시작...")
        logger.info("=" * 60)
        
        results = {
            'ticker': self.ticker,
            'data_search': {},
            'model_training': {},
            'ml_prediction': {},
            'llm_debate': {},
            'final_consensus': {},
            'timestamp': None
        }
        
        # 1단계: 데이터 수집
        results['data_search'] = self.step1_data_search()
        
        # 2단계: 모델 학습 (선택사항)
        results['model_training'] = self.step2_model_training(force_retrain)
        
        # 3단계: ML 예측
        results['ml_prediction'] = self.step3_prediction()
        
        # 4단계: LLM 토론
        if results['ml_prediction']['success']:
            results['llm_debate'] = self.step4_debate(debate_rounds)
        
        # 최종 합의
        results['final_consensus'] = {
            'ml_prediction': results['ml_prediction'].get('consensus', 0.0),
            'llm_prediction': results['llm_debate'].get('final', {}).get('mean_next_close', 0.0),
            'final_prediction': results['ml_prediction'].get('consensus', 0.0),  # ML 우선
            'reasoning': ["ML 예측값을 최종 결과로 사용"]
        }
        
        logger.info("=" * 60)
        logger.info("🎉 전체 분석 완료!")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """결과 출력"""
        print(f"\n📊 {self.ticker} 분석 결과")
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
            print(f"✅ 최종 합의: ${results['ml_prediction']['consensus']:.2f}")
            for agent_id, pred_data in results['ml_prediction']['predictions'].items():
                print(f"  • {agent_id}: ${pred_data['prediction']:.2f} (β: {pred_data['beta']:.3f})")
        else:
            print(f"❌ ML 예측 실패")
        
        # LLM 토론 결과
        print("\n💬 4단계: LLM 토론")
        if results['llm_debate']['success']:
            print("✅ 토론 완료")
            if results['llm_debate']['final']:
                final = results['llm_debate']['final']
                print(f"  • 최종 의견: ${final.get('mean_next_close', 0.0):.2f}")
        else:
            print(f"❌ 토론 실패: {results['llm_debate']['error']}")
        
        # 최종 합의
        print("\n🎯 최종 합의")
        final_consensus = results['final_consensus']
        print(f"✅ 최종 예측: ${final_consensus['final_prediction']:.2f}")
        for reason in final_consensus['reasoning']:
            print(f"  • {reason}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='간단한 하이브리드 주식 예측 시스템')
    parser.add_argument('--ticker', type=str, default='RZLV', help='주식 티커 (기본값: RZLV)')
    parser.add_argument('--force-retrain', action='store_true', help='모델 강제 재학습')
    parser.add_argument('--rounds', type=int, default=3, help='토론 라운드 수 (기본값: 3)')
    parser.add_argument('--step', type=str, choices=['search', 'train', 'predict', 'debate', 'all'], 
                       default='all', help='실행할 단계')
    parser.add_argument('--no-ml', action='store_true', help='ML 기능 비활성화')
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = SimpleHybridSystem(args.ticker, use_ml=not args.no_ml)
    
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
            print(f"\n📊 {args.ticker} 예측 결과:")
            print(f"✅ 최종 합의: ${result['consensus']:.2f}")
            for agent_id, pred_data in result['predictions'].items():
                print(f"  • {agent_id}: ${pred_data['prediction']:.2f}")
        else:
            print(f"❌ 예측 실패")
    
    elif args.step == 'debate':
        # 4단계만 실행
        result = system.step4_debate(args.rounds)
        if result['success']:
            print(f"✅ 토론 완료")
        else:
            print(f"❌ 토론 실패: {result['error']}")
    
    else:  # all
        # 전체 분석 실행
        results = system.run_full_analysis(args.force_retrain, args.rounds)
        system.print_results(results)


if __name__ == "__main__":
    main()
