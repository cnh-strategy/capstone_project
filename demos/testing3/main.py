#!/usr/bin/env python3
"""
새로운 하이브리드 주식 예측 시스템 - 메인 진입점
원래 capstone 구조를 기반으로 ML과 LLM을 통합한 시스템
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ML 모듈 import
from ml_modules.searcher import DataSearcher
from ml_modules.trainer import ModelTrainer
from ml_modules.predicter import StockPredictor

# LLM 모듈 import (원래 capstone 구조)
from debate_agent import Debate
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent
from agents.technical_agent import TechnicalAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridStockPredictionSystem:
    """하이브리드 주식 예측 시스템"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.data_searcher = DataSearcher(ticker)
        self.model_trainer = ModelTrainer(ticker)
        self.stock_predictor = StockPredictor(ticker)
        
        # LLM 에이전트들
        self.llm_agents = self._create_llm_agents()
        
    def _create_llm_agents(self) -> List:
        """LLM 에이전트 생성"""
        try:
            agents = [
                FundamentalAgent(agent_id="FundamentalAgent"),
                TechnicalAgent(agent_id="TechnicalAgent"),
                SentimentalAgent(agent_id="SentimentalAgent")
            ]
            logger.info("✅ LLM 에이전트 생성 완료")
            return agents
        except Exception as e:
            logger.error(f"❌ LLM 에이전트 생성 실패: {str(e)}")
            return []
    
    def step1_data_search(self) -> Dict[str, str]:
        """1단계: 데이터 수집 (Searcher)"""
        logger.info(f"🔍 1단계: {self.ticker} 데이터 수집 시작...")
        
        results = self.data_searcher.search_all_data()
        
        success_count = sum(1 for path in results.values() if path is not None)
        logger.info(f"📊 데이터 수집 완료: {success_count}/3 성공")
        
        return results
    
    def step2_model_training(self, force_retrain: bool = False) -> Dict[str, bool]:
        """2단계: 모델 학습 (Trainer) - 선택사항"""
        logger.info(f"🎯 2단계: {self.ticker} 모델 학습 시작...")
        
        if force_retrain:
            # 강제 재학습
            results = self.model_trainer.train_all_models()
        else:
            # 기존 모델 확인 후 필요시에만 학습
            results = {}
            for agent_type in ['technical', 'fundamental', 'sentimental']:
                if self.model_trainer.load_existing_model(agent_type):
                    results[agent_type] = True
                    logger.info(f"✅ {agent_type}: 기존 모델 사용")
                else:
                    logger.info(f"🔄 {agent_type}: 새로 학습 시작...")
                    # 개별 모델 학습
                    df = self.model_trainer.load_data(agent_type)
                    if df is not None:
                        if agent_type == 'technical':
                            X, y = self.model_trainer.prepare_technical_data(df)
                        elif agent_type == 'fundamental':
                            X, y = self.model_trainer.prepare_fundamental_data(df)
                        else:  # sentimental
                            X, y = self.model_trainer.prepare_sentimental_data(df)
                        
                        success = self.model_trainer.train_model(agent_type, X, y)
                        results[agent_type] = success
                    else:
                        results[agent_type] = False
        
        success_count = sum(results.values())
        logger.info(f"📊 모델 학습 완료: {success_count}/3 성공")
        
        return results
    
    def step3_prediction(self) -> Dict[str, Any]:
        """3단계: 예측 (Predicter)"""
        logger.info(f"🎯 3단계: {self.ticker} 예측 시작...")
        
        result = self.stock_predictor.predict_next_day_close()
        
        if result['success']:
            logger.info(f"✅ 예측 완료: 합의={result['consensus']:.4f}")
        else:
            logger.error(f"❌ 예측 실패: {result['error']}")
        
        return result
    
    def step4_debate_rounds(self, ml_predictions: Dict[str, float], rounds: int = 3) -> Dict[str, Any]:
        """4단계: 토론 라운드 (Debate)"""
        logger.info(f"💬 4단계: {self.ticker} 토론 라운드 시작...")
        
        if not self.llm_agents:
            return {
                'success': False,
                'error': 'LLM 에이전트가 없습니다',
                'logs': [],
                'final': {}
            }
        
        try:
            # 토론 실행
            debate = Debate(self.llm_agents, verbose=True)
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
            ml_predictions = results['ml_prediction']['predictions']
            results['llm_debate'] = self.step4_debate_rounds(ml_predictions, debate_rounds)
        
        # 최종 합의
        results['final_consensus'] = self._generate_final_consensus(results)
        
        logger.info("=" * 60)
        logger.info("🎉 전체 분석 완료!")
        
        return results
    
    def _generate_final_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """최종 합의 생성"""
        consensus = {
            'ml_prediction': None,
            'llm_prediction': None,
            'final_prediction': None,
            'confidence': 'medium',
            'reasoning': []
        }
        
        # ML 예측 결과
        if results['ml_prediction']['success']:
            consensus['ml_prediction'] = results['ml_prediction']['consensus']
            consensus['reasoning'].append("ML 모델의 정량적 예측을 기반으로 함")
        
        # LLM 토론 결과
        if results['llm_debate']['success']:
            llm_final = results['llm_debate']['final']
            if llm_final:
                consensus['llm_prediction'] = llm_final.get('mean_next_close', 0.0)
                consensus['reasoning'].append("LLM 토론의 정성적 분석을 기반으로 함")
        
        # 최종 예측 (ML 우선, LLM 보조)
        if consensus['ml_prediction'] is not None:
            consensus['final_prediction'] = consensus['ml_prediction']
            consensus['reasoning'].append("ML 예측값을 최종 결과로 사용")
        elif consensus['llm_prediction'] is not None:
            consensus['final_prediction'] = consensus['llm_prediction']
            consensus['reasoning'].append("LLM 예측값을 최종 결과로 사용")
        
        return consensus
    
    def print_results(self, results: Dict[str, Any]):
        """결과 출력"""
        print(f"\n📊 {self.ticker} 분석 결과")
        print("=" * 60)
        
        # 데이터 수집 결과
        print("\n🔍 1단계: 데이터 수집")
        for agent_type, filepath in results['data_search'].items():
            if filepath:
                print(f"✅ {agent_type}: {filepath}")
            else:
                print(f"❌ {agent_type}: 실패")
        
        # 모델 학습 결과
        print("\n🎯 2단계: 모델 학습")
        for agent_type, success in results['model_training'].items():
            if success:
                print(f"✅ {agent_type}: 완료")
            else:
                print(f"❌ {agent_type}: 실패")
        
        # ML 예측 결과
        print("\n📈 3단계: ML 예측")
        if results['ml_prediction']['success']:
            print(f"✅ 최종 합의: ${results['ml_prediction']['consensus']:.2f}")
            for agent_type, prediction in results['ml_prediction']['predictions'].items():
                print(f"  • {agent_type}: ${prediction:.2f}")
        else:
            print(f"❌ ML 예측 실패: {results['ml_prediction']['error']}")
        
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
        if final_consensus['final_prediction'] is not None:
            print(f"✅ 최종 예측: ${final_consensus['final_prediction']:.2f}")
            for reason in final_consensus['reasoning']:
                print(f"  • {reason}")
        else:
            print("❌ 최종 예측 실패")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='하이브리드 주식 예측 시스템')
    parser.add_argument('--ticker', type=str, default='RZLV', help='주식 티커 (기본값: RZLV)')
    parser.add_argument('--force-retrain', action='store_true', help='모델 강제 재학습')
    parser.add_argument('--rounds', type=int, default=3, help='토론 라운드 수 (기본값: 3)')
    parser.add_argument('--step', type=str, choices=['search', 'train', 'predict', 'debate', 'all'], 
                       default='all', help='실행할 단계')
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = HybridStockPredictionSystem(args.ticker)
    
    if args.step == 'search':
        # 1단계만 실행
        results = system.step1_data_search()
        print(f"\n📊 {args.ticker} 데이터 수집 결과:")
        for agent_type, filepath in results.items():
            if filepath:
                print(f"✅ {agent_type}: {filepath}")
            else:
                print(f"❌ {agent_type}: 실패")
    
    elif args.step == 'train':
        # 2단계만 실행
        results = system.step2_model_training(args.force_retrain)
        print(f"\n📊 {args.ticker} 모델 학습 결과:")
        for agent_type, success in results.items():
            if success:
                print(f"✅ {agent_type}: 완료")
            else:
                print(f"❌ {agent_type}: 실패")
    
    elif args.step == 'predict':
        # 3단계만 실행
        result = system.step3_prediction()
        if result['success']:
            print(f"\n{predictor.get_prediction_summary()}")
        else:
            print(f"❌ 예측 실패: {result['error']}")
    
    elif args.step == 'debate':
        # 4단계만 실행
        result = system.step4_debate_rounds({}, args.rounds)
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
