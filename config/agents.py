# ===============================================
# MCP Agent Configuration
# ===============================================

agents_info = {
    "TechnicalAgent": {
        "description": "기술적 분석 기반 단기 추세 예측 모델 (TCN)",
        "input_dim": 10,
        "hidden_dim": 64,
        "dropout": 0.1,
        "data_cols": ["Open", "High", "Low", "Close", "Volume", "returns", "sma_5", "sma_20", "rsi", "volume_z"],
        "window_size": 14,
        "epochs": 50,  # 기존: 200 → 수정: 50으로 단축 (빠른 테스트용)
        "learning_rate": 0.0001,  # 기존: 0.001 → 수정: 0.01로 더 증가 (더 빠른 학습)
        "batch_size": 16,        # 기존: 32 → 수정: 16으로 감소 (더 안정적인 학습)
        "period": "2y", # 2y, 5y, 10y
        "interval": "1d", # 1d, 1w, 1m
        "x_scaler": "StandardScaler", # StandardScaler, MinMaxScaler, RobustScaler, None
        "y_scaler": "None",  # 기존: "None" → 수정: StandardScaler (오류 해결)
        "gamma": 0.3,
        "delta_limit": 0.05,
    },

    "MacroSentiAgent"  : {
        "description": "거시경제 데이터 기반 시장 분석 모델 (LSTM)",
        "input_dim": 13,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "data_cols": ["Open", "High", "Low", "Close", "Volume", "returns", "sma_5", "sma_20", "rsi", "volume_z", "USD_KRW", "NASDAQ", "VIX"],
        "window_size": 14,
        "epochs": 50,  # 기존: 200 → 수정: 50으로 단축 (빠른 테스트용)
        "learning_rate": 0.0001,  # 기존: 0.001 → 수정: 0.01로 더 증가 (더 빠른 학습)
        "batch_size": 16,        # 기존: 32 → 수정: 16으로 감소 (더 안정적인 학습)
        "period": "2y", # 2y, 5y, 10y
        "interval": "1d", # 1d, 1w, 1m
        "x_scaler": "StandardScaler", # StandardScaler, MinMaxScaler, RobustScaler, None
        "y_scaler": "None",  # 기존: None → 수정: StandardScaler (일관성 유지)
        "gamma": 0.5,
        "delta_limit": 0.1,
    },

    "SentimentalAgent": {
        "description": "투자자 심리 및 뉴스 감성 기반 시장 예측 모델 (Transformer)",
        "input_dim": 8,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "data_cols": ["returns", "sentiment_mean", "sentiment_vol", "Close", "Volume", "Open", "High", "Low"],
        "window_size": 14,
        "epochs": 50,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "period": "2y", # 2y, 5y, 10y
        "interval": "1d", # 1d, 1w, 1m      
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler"
    }
}


dir_info = {
    "data_dir": "data/processed",
    "model_dir": "models",
    "scaler_dir": "models/scalers"
}