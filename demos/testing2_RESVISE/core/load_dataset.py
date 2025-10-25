import pandas as pd
import os
import numpy as np
# 데이터셋 로드

def load_dataset(dataset_type, phase='pretrain', ticker='TSLA'):
    # 데이터셋 로드
    file_path = f'data/{ticker}_{dataset_type}_{phase}.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prepare features and target
    if dataset_type == 'technical':
        # Technical features: Open, High, Low, Close, Volume, returns, sma_5, sma_20, rsi, volume_z
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'sma_5', 'sma_20', 'rsi', 'volume_z']
        target_col = 'Close'
    elif dataset_type == 'fundamental':
        # Fundamental features: Close, USD_KRW, NASDAQ, VIX, priceEarningsRatio, forwardPE, 
        # debtEquityRatio, returnOnAssets, returnOnEquity, profitMargins, grossMargins
        feature_cols = ['Close', 'USD_KRW', 'NASDAQ', 'VIX', 'priceEarningsRatio', 'forwardPE', 
                       'priceToBook', 'debtEquityRatio', 'returnOnAssets', 'returnOnEquity', 
                       'profitMargins', 'grossMargins']
        target_col = 'Close'
    elif dataset_type == 'sentimental':
        # Sentimental features: Close, returns, sentiment_mean, sentiment_vol
        feature_cols = ['Close', 'returns', 'sentiment_mean', 'sentiment_vol']
        target_col = 'Close'
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Extract features and target
    features = df[feature_cols].values
    target = df[target_col].values
    
    return features, target, feature_cols


def create_sequences(features, target, days=7):
    """Create time series sequences"""
    window_size = days
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])
    
    return np.array(X), np.array(y)
