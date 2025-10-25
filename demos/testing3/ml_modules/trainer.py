#!/usr/bin/env python3
"""
Trainer Module - 각 Agent별 모델 학습
2022~2024년 데이터로 개별 Agent 모델 학습
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple
import logging
import pickle

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAgent(nn.Module):
    """기술적 분석 에이전트 (TCN 기반)"""
    
    def __init__(self, input_size: int = 14, hidden_size: int = 64, output_size: int = 1):
        super(TechnicalAgent, self).__init__()
        
        # Temporal Convolutional Network
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size//2, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        
        # Global average pooling
        x = torch.mean(x, dim=2)  # (batch_size, hidden_size//2)
        
        x = self.fc(x)
        return x

class FundamentalAgent(nn.Module):
    """펀더멘털 분석 에이전트 (LSTM 기반)"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
        super(FundamentalAgent, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 마지막 시점의 출력 사용
        x = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SentimentalAgent(nn.Module):
    """감정 분석 에이전트 (Transformer 기반)"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
        super(SentimentalAgent, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=3
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = self.input_projection(x)
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, hidden_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ModelTrainer:
    """모델 학습 클래스"""
    
    def __init__(self, ticker: str, data_dir: str = "data", models_dir: str = "ml_modules/models"):
        self.ticker = ticker.upper()
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.ensure_models_dir()
        
        # 모델 정의
        self.models = {
            'technical': TechnicalAgent(),
            'fundamental': FundamentalAgent(),
            'sentimental': SentimentalAgent()
        }
        
        # 스케일러
        self.scalers = {}
        
    def ensure_models_dir(self):
        """모델 디렉토리 생성"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def load_data(self, agent_type: str) -> Optional[pd.DataFrame]:
        """데이터 로드"""
        filename = f"{self.ticker}_{agent_type}_data.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"❌ 데이터 파일이 없습니다: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"✅ {agent_type} 데이터 로드 완료: {len(df)}개 행")
            return df
        except Exception as e:
            logger.error(f"❌ 데이터 로드 실패: {str(e)}")
            return None
    
    def prepare_technical_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """기술적 데이터 전처리"""
        # 특성 선택
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'rsi', 'macd',
            'bollinger_upper', 'bollinger_lower', 'atr', 'volume_sma'
        ]
        
        # 결측값 처리
        df = df.dropna()
        
        # 특성과 타겟 분리
        X = df[feature_columns].values
        y = df['close'].values
        
        # 정규화
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 시퀀스 데이터 생성 (30일 윈도우)
        sequence_length = 30
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 타겟 정규화
        y_scaler = MinMaxScaler()
        y_seq_scaled = y_scaler.fit_transform(y_seq.reshape(-1, 1)).flatten()
        
        self.scalers['technical'] = {'X': scaler, 'y': y_scaler}
        
        return X_seq, y_seq_scaled
    
    def prepare_fundamental_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """펀더멘털 데이터 전처리"""
        # 특성 선택
        feature_columns = [
            'market_cap', 'pe_ratio', 'pb_ratio', 'debt_to_equity',
            'revenue_growth', 'profit_margin', 'roe', 'current_ratio',
            'dividend_yield'
        ]
        
        # 결측값 처리
        df = df.dropna()
        
        # 특성과 타겟 분리 (종가를 타겟으로 사용)
        X = df[feature_columns].values
        y = df['market_cap'].values  # 시가총액을 타겟으로 사용
        
        # 정규화
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 시퀀스 데이터 생성 (30일 윈도우)
        sequence_length = 30
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 타겟 정규화
        y_scaler = MinMaxScaler()
        y_seq_scaled = y_scaler.fit_transform(y_seq.reshape(-1, 1)).flatten()
        
        self.scalers['fundamental'] = {'X': scaler, 'y': y_scaler}
        
        return X_seq, y_seq_scaled
    
    def prepare_sentimental_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """감정 분석 데이터 전처리"""
        # 특성 선택
        feature_columns = [
            'news_sentiment', 'social_sentiment', 'analyst_rating',
            'price_target', 'earnings_surprise', 'insider_trading',
            'institutional_flow', 'options_sentiment', 'fear_greed_index'
        ]
        
        # 결측값 처리
        df = df.dropna()
        
        # 특성과 타겟 분리 (가격 타겟을 타겟으로 사용)
        X = df[feature_columns].values
        y = df['price_target'].values
        
        # 정규화
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 시퀀스 데이터 생성 (30일 윈도우)
        sequence_length = 30
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 타겟 정규화
        y_scaler = MinMaxScaler()
        y_seq_scaled = y_scaler.fit_transform(y_seq.reshape(-1, 1)).flatten()
        
        self.scalers['sentimental'] = {'X': scaler, 'y': y_scaler}
        
        return X_seq, y_seq_scaled
    
    def train_model(self, agent_type: str, X: np.ndarray, y: np.ndarray) -> bool:
        """모델 학습"""
        logger.info(f"🎯 {agent_type} 모델 학습 시작...")
        
        try:
            # 데이터 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # PyTorch 텐서로 변환
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            
            # 데이터로더 생성
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 모델, 손실함수, 옵티마이저 설정
            model = self.models[agent_type]
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 학습
            num_epochs = 50
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                # 훈련
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 검증
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # 모델 저장
                    model_path = os.path.join(self.models_dir, f"{self.ticker}_{agent_type}_model.pt")
                    torch.save(model.state_dict(), model_path)
                    
                    # 스케일러 저장
                    scaler_path = os.path.join(self.models_dir, f"{self.ticker}_{agent_type}_scaler.pkl")
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[agent_type], f)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            logger.info(f"✅ {agent_type} 모델 학습 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ {agent_type} 모델 학습 실패: {str(e)}")
            return False
    
    def train_all_models(self) -> Dict[str, bool]:
        """모든 모델 학습"""
        logger.info(f"🚀 {self.ticker} 모든 모델 학습 시작...")
        
        results = {}
        
        for agent_type in ['technical', 'fundamental', 'sentimental']:
            # 데이터 로드
            df = self.load_data(agent_type)
            if df is None:
                results[agent_type] = False
                continue
            
            # 데이터 전처리
            if agent_type == 'technical':
                X, y = self.prepare_technical_data(df)
            elif agent_type == 'fundamental':
                X, y = self.prepare_fundamental_data(df)
            else:  # sentimental
                X, y = self.prepare_sentimental_data(df)
            
            # 모델 학습
            success = self.train_model(agent_type, X, y)
            results[agent_type] = success
        
        # 결과 요약
        success_count = sum(results.values())
        logger.info(f"📊 모델 학습 완료: {success_count}/3 성공")
        
        return results
    
    def load_existing_model(self, agent_type: str) -> bool:
        """기존 모델 로드"""
        model_path = os.path.join(self.models_dir, f"{self.ticker}_{agent_type}_model.pt")
        scaler_path = os.path.join(self.models_dir, f"{self.ticker}_{agent_type}_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                # 모델 로드
                self.models[agent_type].load_state_dict(torch.load(model_path))
                
                # 스케일러 로드
                with open(scaler_path, 'rb') as f:
                    self.scalers[agent_type] = pickle.load(f)
                
                logger.info(f"✅ {agent_type} 기존 모델 로드 완료")
                return True
            except Exception as e:
                logger.error(f"❌ {agent_type} 모델 로드 실패: {str(e)}")
                return False
        
        return False


def main():
    """테스트 실행"""
    ticker = "RZLV"
    trainer = ModelTrainer(ticker)
    
    # 기존 모델 확인
    print(f"\n🔍 {ticker} 기존 모델 확인:")
    for agent_type in ['technical', 'fundamental', 'sentimental']:
        if trainer.load_existing_model(agent_type):
            print(f"✅ {agent_type}: 기존 모델 사용")
        else:
            print(f"❌ {agent_type}: 새로 학습 필요")
    
    # 모든 모델 학습
    results = trainer.train_all_models()
    
    print(f"\n📋 {ticker} 모델 학습 결과:")
    for agent_type, success in results.items():
        if success:
            print(f"✅ {agent_type}: 학습 완료")
        else:
            print(f"❌ {agent_type}: 학습 실패")


if __name__ == "__main__":
    main()
