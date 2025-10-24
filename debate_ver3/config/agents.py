# ===============================================
# MCP Agent Configuration
# ===============================================


# 아연 테크니컬 수정
agents_info = {
    "TechnicalAgent": {
        "description": "TECH(12)+FUND(7) → [GRU ⊕ MLP] with Attention + 2-layer Gating",
        # 데이터/대상
        "tickers": ["NVDA", "AAPL", "MSFT"], # 생략 가능?
        "start_date": "2020-01-01",
        "end_date_inclusive": "2024-12-31",
        "assimilation_month_start": "2025-01-01",
        "assimilation_month_end": "2025-01-31",

        # 윈도우 및 피처
        "lookback": 40,
        "features_tech": [
            "vol_20d","obv","ret_3d","r2",
            "weekofyear","vol_ma_20","vol_chg",
            "bbp","ma_200","macd","adx_14","mom_10"
            ],
        "features_fund": [
            "log_marketcap","earnings_yield","book_to_market",
            "dividend_yield","net_margin","short_signal","avg_vol_60d"
        ],

        # 모델 구조
        "gru_units1": 32,
        "gru_units2": 16,
        "mlp_hidden": 32,
        "gate_hidden": 8,
        "dropout": 0.12,              # MC Dropout은 예측 단계에서만 활성화
        "use_one_hot_ticker": True,   # 시퀀스/벡터 모두에 티커 one-hot 부착

        # 학습
        "loss": "L1",
        "epochs": 90,
        "batch_size": 64,
        "learning_rate": 3.556365139054937e-4,
        "patience": 8,
        "test_ratio": 0.20,           # 뒤 20% 테스트
        "val_tail_ratio": 0.05,       # 학습 구간의 마지막 5%를 검증
        "scaler_tech": "MinMax",
        "scaler_fund": "MinMax",

        # 게이트 규제
        "gate_target": 0.45,
        "gate_lambda": 0.01,
        "gate_clip": [0.05, 0.95],

        # 결과 저장
        "outdir": "model_artifacts_filtered_v4",
        "save_explain_json": True,    # attention/GxI/fund zero-out/gate/pred 요약 저장

        # 예측(옵션)
        "mc_dropout_samples": 50,      # 예측 시 샘플 수(학습 파일엔 미사용)

        # BaseAgent 레거시 필드(읽히기만 함. 값은 무해한 기본값으로 설정)
        "input_dim": 10,
        "hidden_dim": 64,
        "window_size": 14,
        "period": "5y",
        "interval": "1d",
        "data_cols": ["Close"],
        "x_scaler": "MinMaxScaler",
        "y_scaler": "MinMaxScaler"
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
        "period": "2y", # 2y, 5y, 10y
        "interval": "1d", # 1d, 1w, 1m
        "x_scaler": "StandardScaler", # StandardScaler, MinMaxScaler, RobustScaler, None
        "y_scaler": "StandardScaler" # StandardScaler, MinMaxScaler, RobustScaler, None
    },

    "SentimentalAgent": {
        "description": "투자자 심리 및 뉴스 감성 기반 시장 예측 모델 (Transformer)",
        "input_dim": 8,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "data_cols": ["returns", "sentiment_mean", "sentiment_vol", "Close", "Volume", "Open", "High", "Low"],
        "window_size": 14,
        "epochs": 50,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "period": "2y", # 2y, 5y, 10y
        "interval": "1d", # 1d, 1w, 1m      
        "x_scaler": "StandardScaler", # StandardScaler, MinMaxScaler, RobustScaler, None
        "y_scaler": "StandardScaler" # StandardScaler, MinMaxScaler, RobustScaler, None
    }
}


dir_info = {
    "data_dir": "data/processed",
    "model_dir": "models",
    "scaler_dir": "models/scalers"
}