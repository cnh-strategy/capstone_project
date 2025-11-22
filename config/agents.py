# config/agents.py
# ===============================================================
# 에이전트별 하이퍼파라미터 & 경로 설정
#  - BaseAgent는 agents_info[agent_id]에서 아래 키들을 사용합니다:
#    window_size, epochs, learning_rate, batch_size,
#    x_scaler, y_scaler, gamma, delta_limit
#  - 추가 모델 전용 하이퍼파라미터(예: d_model, nhead 등)는
#    각 에이전트 구현에서 선택적으로 사용합니다.
# ===============================================================

agents_info = {
    # -----------------------------------------------------------
    # TechnicalAgent: 기술적 분석 기반 (예: TCN/LSTM 등)
    # -----------------------------------------------------------
    "TechnicalAgent": {
        "description": "TECH(13) → LSTM×2 + time-attention 모델 사용",
        "data_cols": [
            "weekofyear_sin","weekofyear_cos","log_ret_lag1",
            "ret_3d","mom_10","ma_200",
            "macd","bbp","adx_14",
            "obv","vol_ma_20","vol_chg","vol_20d"
            ],
        "feature_builder": "core.technical_classes.technical:build_features_technical", # 수정
        "input_dim": 13,
        "window_size": 55,              # lookback
        "rnn_units1": 64,               # 1층 hidden size
        "rnn_units2": 32,               # 2층 hidden size
        "dropout": 0.18778570103014075,
        "epochs": 45,
        "patience": 8,
        "learning_rate": 4.2471233429729313e-4,
        "batch_size": 64,
        "period": "5y",
        "interval": "1d",
        "x_scaler": "MinMaxScaler",
        "y_scaler": "StandardScaler",
        "gamma": 0.3,
        "delta_limit": 0.05,
        "seed": 1234
    },

    # -----------------------------------------------------------
    # MacroAgent: 거시지표 + 시장심리 조합 모델
    #  (매크로 모듈이 없으면 코드에서 자동으로 우회되도록 구성)
    # -----------------------------------------------------------
    "MacroAgent": {
        "description": "거시경제 데이터 기반 시장 분석 모델",
        # 모델/피처 관련
        "input_dim": 13,  # 기본값, 실제는 데이터 로드 시 결정
        "hidden_dims": [128, 64, 32],  # LSTM 3층 hidden dimensions
        "dropout_rates": [0.3, 0.3, 0.2],  # 각 LSTM 레이어별 dropout
        "data_cols": [
            "Open", "High", "Low", "Close", "Volume",
            "returns", "sma_5", "sma_20", "rsi", "volume_z",
            "USD_KRW", "NASDAQ", "VIX"
        ],
        # 시퀀스/학습 관련
        "window_size": 40,  # 실제 사용값
        "epochs": 60,
        "patience": 10,  # Early stopping patience
        "learning_rate": 0.0005,  # 5e-4
        "batch_size": 16,
        "loss_fn": "L1Loss",  # Loss function type
        "period": "2y",
        "interval": "1d",
        # 스케일러
        "x_scaler": "StandardScaler",
        "y_scaler": "MinMaxScaler",  # 실제 사용값 (feature_range=(-1, 1))
        # 합의/수렴 관련
        "gamma": 0.5,
        "delta_limit": 0.1,
    },

    # -----------------------------------------------------------
    # SentimentalAgent: 뉴스/커뮤니티 감성 + 가격 피처
    # -----------------------------------------------------------
    "SentimentalAgent": {
        "description": "투자자 심리 및 뉴스 감성 기반 시장 예측 모델",
        # 모델/피처 관련
        "input_dim": 8,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "data_cols": [
            "returns", "sentiment_mean", "sentiment_vol",
            "Close", "Volume", "Open", "High", "Low"
        ],
        # 시퀀스/학습 관련
        "window_size": 40,
        "epochs": 50,
        "learning_rate": 5e-4,      # 0.0005
        "batch_size": 32,
        "period": "5y",
        "interval": "1d",
        # 스케일러
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        # 합의/수렴 관련
        "gamma": 0.3,               # 수렴율
        "delta_limit": 0.05,
    },
}

dir_info = {
    "data_dir": "data/processed",
    "model_dir": "models",
    "scaler_dir": "models/scalers",
    "artifacts_dir": "artifacts" # 아연추가(필요없을시 삭제)
}
