#!/usr/bin/env python3
"""
Predicter Module - 각 Agent별 예측 및 상호학습
최근 1년 데이터로 상호학습 진행 후, 최근 7일 데이터로 다음날 종가 예측
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MutualLearningTrainer:
    """상호학습 트레이너"""
    
    def __init__(self, ticker: str, models_dir: str = "ml_modules/models"):
        self.ticker = ticker.upper()
        self.models_dir = models_dir
        
        # 에이전트별 신뢰도 (β)
        self.beta_values = {
            'technical': 0.5,
            'fundamental': 0.5,
            'sentimental': 0.5
        }
        
        # 상호학습 파라미터
        self.alpha = 0.1  # 학습률
        self.lambda_ema = 0.9  # EMA 가중치
        
    def load_model_and_scaler(self, agent_type: str) -> Tuple[Optional[nn.Module], Optional[dict]]:
        """모델과 스케일러 로드"""
        model_path = os.path.join(self.models_dir, f"{self.ticker}_{agent_type}_model.pt")
        scaler_path = os.path.join(self.models_dir, f"{self.ticker}_{agent_type}_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning(f"⚠️ {agent_type} 모델 또는 스케일러가 없습니다")
            return None, None
        
        try:
            # 모델 로드
            if agent_type == 'technical':
                from .trainer import TechnicalAgent
                model = TechnicalAgent()
            elif agent_type == 'fundamental':
                from .trainer import FundamentalAgent
                model = FundamentalAgent()
            else:  # sentimental
                from .trainer import SentimentalAgent
                model = SentimentalAgent()
            
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # 스케일러 로드
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            return model, scaler
            
        except Exception as e:
            logger.error(f"❌ {agent_type} 모델 로드 실패: {str(e)}")
            return None, None
    
    def predict_with_uncertainty(self, model: nn.Module, X: torch.Tensor, num_samples: int = 10) -> Tuple[float, float]:
        """Monte Carlo Dropout으로 불확실성과 함께 예측"""
        model.train()  # Dropout 활성화
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred.item())
        
        model.eval()  # Dropout 비활성화
        
        mean_pred = np.mean(predictions)
        uncertainty = np.var(predictions)
        
        return mean_pred, uncertainty
    
    def peer_correction(self, agent_type: str, prediction: float, peer_predictions: Dict[str, float]) -> float:
        """동료 에이전트들의 예측을 바탕으로 보정"""
        if len(peer_predictions) == 0:
            return prediction
        
        # 동료들의 평균 예측
        peer_mean = np.mean(list(peer_predictions.values()))
        
        # β 값에 따른 보정
        beta = self.beta_values[agent_type]
        corrected_prediction = prediction + self.alpha * beta * (peer_mean - prediction)
        
        return corrected_prediction
    
    def update_beta(self, agent_type: str, uncertainty: float, accuracy: float):
        """β 신뢰도 업데이트 (EMA 방식)"""
        # 불확실성 기반 신뢰도 계산
        confidence = 1.0 / (1.0 + uncertainty)
        
        # 정확도 기반 신뢰도 계산
        accuracy_confidence = accuracy
        
        # 종합 신뢰도
        new_beta = (confidence + accuracy_confidence) / 2.0
        
        # EMA 업데이트
        self.beta_values[agent_type] = (
            self.lambda_ema * self.beta_values[agent_type] + 
            (1 - self.lambda_ema) * new_beta
        )
    
    def mutual_learning_round(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """상호학습 라운드"""
        logger.info("🔄 상호학습 라운드 시작...")
        
        predictions = {}
        uncertainties = {}
        
        # 각 에이전트별 예측
        for agent_type in ['technical', 'fundamental', 'sentimental']:
            model, scaler = self.load_model_and_scaler(agent_type)
            if model is None or scaler is None:
                continue
            
            # 데이터 전처리
            X, _ = self._prepare_data_for_prediction(agent_type, data_dict[agent_type], scaler)
            if X is None:
                continue
            
            # 예측 (불확실성 포함)
            X_tensor = torch.FloatTensor(X).unsqueeze(0)  # 배치 차원 추가
            pred, uncertainty = self.predict_with_uncertainty(model, X_tensor)
            
            predictions[agent_type] = pred
            uncertainties[agent_type] = uncertainty
            
            logger.info(f"📊 {agent_type}: 예측={pred:.4f}, 불확실성={uncertainty:.4f}")
        
        # 상호학습 보정
        corrected_predictions = {}
        for agent_type, prediction in predictions.items():
            peer_predictions = {k: v for k, v in predictions.items() if k != agent_type}
            corrected_pred = self.peer_correction(agent_type, prediction, peer_predictions)
            corrected_predictions[agent_type] = corrected_pred
            
            logger.info(f"🔄 {agent_type}: 보정 전={prediction:.4f}, 보정 후={corrected_pred:.4f}")
        
        return corrected_predictions
    
    def _prepare_data_for_prediction(self, agent_type: str, df: pd.DataFrame, scaler: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """예측을 위한 데이터 전처리"""
        try:
            # 최근 30일 데이터 사용
            df = df.tail(30)
            
            if agent_type == 'technical':
                feature_columns = [
                    'open', 'high', 'low', 'close', 'volume',
                    'sma_20', 'sma_50', 'rsi', 'macd',
                    'bollinger_upper', 'bollinger_lower', 'atr', 'volume_sma'
                ]
            elif agent_type == 'fundamental':
                feature_columns = [
                    'market_cap', 'pe_ratio', 'pb_ratio', 'debt_to_equity',
                    'revenue_growth', 'profit_margin', 'roe', 'current_ratio',
                    'dividend_yield'
                ]
            else:  # sentimental
                feature_columns = [
                    'news_sentiment', 'social_sentiment', 'analyst_rating',
                    'price_target', 'earnings_surprise', 'insider_trading',
                    'institutional_flow', 'options_sentiment', 'fear_greed_index'
                ]
            
            # 결측값 처리
            df = df.dropna()
            
            if len(df) < 30:
                logger.warning(f"⚠️ {agent_type} 데이터가 부족합니다: {len(df)}개")
                return None, None
            
            # 특성 추출
            X = df[feature_columns].values
            
            # 정규화
            X_scaled = scaler['X'].transform(X)
            
            return X_scaled, None
            
        except Exception as e:
            logger.error(f"❌ {agent_type} 데이터 전처리 실패: {str(e)}")
            return None, None

class StockPredictor:
    """주식 예측 클래스"""
    
    def __init__(self, ticker: str, data_dir: str = "data", models_dir: str = "ml_modules/models"):
        self.ticker = ticker.upper()
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.mutual_trainer = MutualLearningTrainer(ticker, models_dir)
        
    def load_recent_data(self) -> Dict[str, pd.DataFrame]:
        """최근 데이터 로드"""
        logger.info(f"📊 {self.ticker} 최근 데이터 로드...")
        
        data_dict = {}
        
        for agent_type in ['technical', 'fundamental', 'sentimental']:
            filename = f"{self.ticker}_{agent_type}_data.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    # 최근 1년 데이터만 사용
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.tail(365)  # 최근 1년
                    data_dict[agent_type] = df
                    logger.info(f"✅ {agent_type}: {len(df)}개 데이터 로드")
                except Exception as e:
                    logger.error(f"❌ {agent_type} 데이터 로드 실패: {str(e)}")
            else:
                logger.warning(f"⚠️ {agent_type} 데이터 파일이 없습니다: {filepath}")
        
        return data_dict
    
    def predict_next_day_close(self) -> Dict[str, any]:
        """다음날 종가 예측"""
        logger.info(f"🎯 {self.ticker} 다음날 종가 예측 시작...")
        
        # 최근 데이터 로드
        data_dict = self.load_recent_data()
        
        if not data_dict:
            return {
                'success': False,
                'error': '데이터를 로드할 수 없습니다',
                'predictions': {},
                'consensus': None
            }
        
        # 상호학습 진행
        mutual_predictions = self.mutual_trainer.mutual_learning_round(data_dict)
        
        if not mutual_predictions:
            return {
                'success': False,
                'error': '상호학습 예측에 실패했습니다',
                'predictions': {},
                'consensus': None
            }
        
        # 최종 합의 예측 (가중평균)
        weights = {
            'technical': 0.4,
            'fundamental': 0.35,
            'sentimental': 0.25
        }
        
        consensus = 0.0
        total_weight = 0.0
        
        for agent_type, prediction in mutual_predictions.items():
            weight = weights.get(agent_type, 0.0)
            consensus += prediction * weight
            total_weight += weight
        
        if total_weight > 0:
            consensus /= total_weight
        
        # 결과 정리
        result = {
            'success': True,
            'predictions': mutual_predictions,
            'consensus': consensus,
            'weights': weights,
            'beta_values': self.mutual_trainer.beta_values,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ {self.ticker} 예측 완료: 합의={consensus:.4f}")
        
        return result
    
    def get_prediction_summary(self) -> str:
        """예측 결과 요약"""
        result = self.predict_next_day_close()
        
        if not result['success']:
            return f"❌ {self.ticker} 예측 실패: {result['error']}"
        
        summary = f"""
📊 {self.ticker} 다음날 종가 예측 결과

🎯 최종 합의: ${result['consensus']:.2f}

📈 각 에이전트별 예측:
"""
        
        for agent_type, prediction in result['predictions'].items():
            weight = result['weights'][agent_type]
            beta = result['beta_values'][agent_type]
            summary += f"• {agent_type.title()}: ${prediction:.2f} (가중치: {weight:.1%}, 신뢰도: {beta:.3f})\n"
        
        summary += f"""
🔄 상호학습 완료
⏰ 예측 시간: {result['timestamp']}
"""
        
        return summary


def main():
    """테스트 실행"""
    ticker = "RZLV"
    predictor = StockPredictor(ticker)
    
    # 예측 실행
    result = predictor.predict_next_day_close()
    
    if result['success']:
        print(f"\n{predictor.get_prediction_summary()}")
    else:
        print(f"❌ 예측 실패: {result['error']}")


if __name__ == "__main__":
    main()
