# debate_ver3\config\agents.py
# ===============================================================

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
        "epochs": 50,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "period": "2y",
        "interval": "1d",
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler"
    },

    "FundamentalAgent": {
        "description": "거시경제 데이터 기반 시장 분석 모델 (LSTM)",
        "input_dim": 13,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "data_cols": ["Open", "High", "Low", "Close", "Volume", "returns", "sma_5", "sma_20", "rsi", "volume_z", "USD_KRW", "NASDAQ", "VIX"],
        "window_size": 14,
        "epochs": 50,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "period": "2y",
        "interval": "1d",
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler"
    },

    "SentimentalAgent": {
        "description": "(현진) FinBERT 기반 감성+LSTM 예측 모델",
        "input_dim": 8,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "data_cols": ["returns", "sentiment_mean", "sentiment_vol", "Close", "Volume", "Open", "High", "Low"],
        "window_size": 40,
        "epochs": 50,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "period": "2y",
        "interval": "1d",
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "output_type": "price",        # 'price' | 'return' | 'log_return'
        "return_index": -1,            # 다차원 출력일 때 사용할 인덱스
        "has_prob_head": False,        # 확률 헤드가 있을 때 True
    }
}

dir_info = {
    "data_dir": "data/processed",
    "model_dir": "models",
    "scaler_dir": "models/scalers"
}

agents_info["SentimentalAgentV3"] = {
    **agents_info["SentimentalAgent"],
    "id": "SentimentalAgentV3",   # 선택: 로깅용 ID 분리
    # "model_path": "models/{ticker}_SentimentalAgent.pt",  # 필요시 그대로/변경
    # "x_scaler":   "...",
    # "y_scaler":   "...",
}