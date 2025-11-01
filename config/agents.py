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
        "description": "기술적 분석 기반 단기 추세 예측 모델",
        # 모델/피처 관련
        "input_dim": 10,
        "hidden_dim": 64,
        "dropout": 0.1,
        "data_cols": [
            "Open", "High", "Low", "Close", "Volume",
            "returns", "sma_5", "sma_20", "rsi", "volume_z"
        ],
        # 시퀀스/학습 관련
        "window_size": 14,
        "epochs": 50,               # 빠른 테스트를 위한 값
        "learning_rate": 1e-4,      # 주석과 값 일치(0.0001)
        "batch_size": 16,
        "period": "2y",             # yfinance: 2y/5y/10y
        "interval": "1d",           # yfinance: 1d/1wk/1mo 등
        # 스케일러
        "x_scaler": "StandardScaler",   # StandardScaler | MinMaxScaler | RobustScaler | None
        "y_scaler": "StandardScaler",   # 회귀 안정화를 위해 Standard 권장
        # 합의/수렴 관련
        "gamma": 0.3,
        "delta_limit": 0.05,
    },

    # -----------------------------------------------------------
    # MacroSentiAgent: 거시지표 + 시장심리 조합 모델
    #  (매크로 모듈이 없으면 코드에서 자동으로 우회되도록 구성)
    # -----------------------------------------------------------
    "MacroSentiAgent": {
        "description": "거시경제 데이터 기반 시장 분석 모델",
        # 모델/피처 관련
        "input_dim": 13,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "data_cols": [
            "Open", "High", "Low", "Close", "Volume",
            "returns", "sma_5", "sma_20", "rsi", "volume_z",
            "USD_KRW", "NASDAQ", "VIX"
        ],
        # 시퀀스/학습 관련
        "window_size": 14,
        "epochs": 50,
        "learning_rate": 1e-4,      # 0.0001
        "batch_size": 16,
        "period": "2y",
        "interval": "1d",
        # 스케일러
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        # 합의/수렴 관련
        "gamma": 0.5,
        "delta_limit": 0.1,
    },

    # -----------------------------------------------------------
    # SentimentalAgent: 뉴스/커뮤니티 감성 + 가격 피처 (Transformer 예시)
    # -----------------------------------------------------------
    "SentimentalAgent": {
        "description": "투자자 심리 및 뉴스 감성 기반 시장 예측 모델 (Transformer)",
        # 모델/피처 관련 (Transformer 예시 하이퍼파라미터)
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
        "period": "2y",
        "interval": "1d",
        # 스케일러
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        # 합의/수렴 관련
        "gamma": 0.3,               # 수렴율
        "delta_limit": 0.05,        # (누락되어 있던 값 추가)
    },
}

dir_info = {
    "data_dir": "data/processed",
    "model_dir": "models",
    "scaler_dir": "models/scalers",
}
