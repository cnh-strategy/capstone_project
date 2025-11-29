# config/agents.py
# ===============================================================
# 에이전트별 하이퍼파라미터 & 경로 설정
#  - BaseAgent는 agents_info[agent_id]에서 아래 키들을 사용합니다:
#    window_size, epochs, learning_rate, batch_size,
#    x_scaler, y_scaler, gamma, delta_limit
#  - 추가 모델 전용 하이퍼파라미터(예: d_model, nhead 등)는
#    각 에이전트 구현에서 선택적으로 사용합니다.
# ===============================================================

    # 공통 파라미터 (모든 Agent에서 사용)
common_params = {
    # Monte Carlo Dropout
    "n_samples": 30,                    # Monte Carlo Dropout 샘플 수
    # LLM 설정
    "temperature": 0.2,                 # LLM temperature
    "preferred_models": ["gpt-5-mini", "gpt-4.1-mini"],  # 모델 폴백 우선순위
    # Loss 함수
    "huber_loss_delta": 1.0,            # HuberLoss delta 파라미터
    # 데이터 스케일링
    "y_scale_factor": 100.0,            # 타겟 스케일링 배수 (수익률 * 100)
    # 평가 설정
    "eval_split_ratio": 0.8,            # 평가 데이터 분할 비율 (80% 학습, 20% 검증)
    # 기본값
    "default_current_price": 100.0,     # 기본 현재가 (데이터 없을 때)
    "sigma_min": 1e-6,                  # 최소 불확실성 (0으로 나누기 방지)
    # Fine-tuning
    "fine_tune_lr": 1e-4,               # Fine-tuning learning rate
    "fine_tune_epochs": 10,             # Fine-tuning epochs (BaseAgent 기본값)
    # Confidence 계산
    "confidence_formula": "1.0 / (1.0 + sigma)",  # Confidence 계산 공식
    # 데이터 수집 기간 (모든 에이전트 공통)
    "period": "2y",              # 일반 모드: searcher + pretrain 모두 동일한 기간 사용
    "period_test": "2y",         # 백테스팅 모드: searcher + pretrain 모두 동일한 기간 사용
    # Pretrain 출력 설정
    "pretrain_log_interval": 5,        # 에포크 출력 주기 (몇 에포크마다 출력할지)
    "pretrain_save_dataset": True,      # 전처리된 데이터셋 CSV 저장 여부
}

agents_info = {
    # -----------------------------------------------------------
    # TechnicalAgent: 기술적 분석 기반 (예: TCN/LSTM 등)
    # -----------------------------------------------------------
    "TechnicalAgent": {
        "description": "TECH(13) → LSTM×2 + time-attention 모델 사용",
        # 모델 구조
        "model_architecture": {
            "type": "LSTM_with_TimeAttention",
            "layers": [
                {"type": "LSTM", "input_dim": "input_dim", "hidden_dim": "rnn_units1", "name": "lstm1"},
                {"type": "Dropout", "rate": "dropout"},
                {"type": "LSTM", "input_dim": "rnn_units1", "hidden_dim": "rnn_units2", "name": "lstm2"},
                {"type": "Dropout", "rate": "dropout"},
                {"type": "TimeAttention", "hidden_dim": "rnn_units2", "name": "attn_vec"},
                {"type": "Linear", "input_dim": "rnn_units2", "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return"
        },
        "data_cols": [
            "weekofyear_sin","weekofyear_cos","log_ret_lag1",
            "ret_3d","mom_10","ma_200",
            "macd","bbp","adx_14",
            "obv","vol_ma_20","vol_chg","vol_20d"
            ],
        "feature_builder": "core.technical_classes.technical:build_features_technical", # 수정
        "input_dim": 13,
        "window_size": 40,              # lookback
        "rnn_units1": 64,               # 1층 hidden size
        "rnn_units2": 32,               # 2층 hidden size
        "dropout": 0.18778570103014075,
        "epochs": 45,
        "patience": 8,
        "learning_rate": 4.2471233429729313e-4,
        "batch_size": 64,
        # period 제거 → common_params["period"] 사용
        "interval": "1d",
        "x_scaler": "MinMaxScaler",
        "y_scaler": "StandardScaler",
        "gamma": 0.3,
        "delta_limit": 0.05,
        "seed": 1234,
        # Loss 함수
        "loss_fn": "HuberLoss",         # Loss function type (HuberLoss, L1Loss, MSELoss)
        # TechnicalAgent 전용 파라미터
        "fine_tune_epochs": 20,         # Fine-tuning epochs (TechnicalAgent)
        # 수익률 클리핑
        "return_clip_min": -0.5,        # 수익률 클리핑 최소값 (-50%)
        "return_clip_max": 0.5,         # 수익률 클리핑 최대값 (+50%)
        # period_searcher, period_pretrain 제거 → common_params["period"] 사용
        # Explainability 파라미터
        "occlusion_batch_size": 32,     # Occlusion 계산 시 배치 크기
        "top_k_features": 5,            # 상위 k개 피처 추출
        "shap_weight_time": 0.20,       # 시간 중요도에서 SHAP 가중치
        "shap_weight_feat": 0.30,       # 피처 중요도에서 SHAP 가중치
        "attention_weights": [0.4, 0.25, 0.15],  # 시간 중요도 융합 가중치 [attn, GI, occ]
        "feature_weights": [0.5, 0.2],  # 피처 중요도 융합 가중치 [GI, occ]
        "pack_idea_top_time": 8,        # _pack_idea에서 상위 시간 개수
        "pack_idea_top_feat": 6,        # _pack_idea에서 상위 피처 개수
        "pack_idea_coverage": 0.8,       # _pack_idea에서 커버리지 비율
    },

    # -----------------------------------------------------------
    # MacroAgent: 거시지표 + 시장심리 조합 모델
    #  (매크로 모듈이 없으면 코드에서 자동으로 우회되도록 구성)
    # -----------------------------------------------------------
    "MacroAgent": {
        "description": "거시경제 데이터 기반 시장 분석 모델",
        # 모델 구조
        "model_architecture": {
            "type": "LSTM_Stacked_Dense",
            "layers": [
                {"type": "LSTM", "input_dim": "input_dim", "hidden_dim": "hidden_dims[0]", "name": "lstm1"},
                {"type": "Dropout", "rate": "dropout_rates[0]"},
                {"type": "LSTM", "input_dim": "hidden_dims[0]", "hidden_dim": "hidden_dims[1]", "name": "lstm2"},
                {"type": "Dropout", "rate": "dropout_rates[1]"},
                {"type": "LSTM", "input_dim": "hidden_dims[1]", "hidden_dim": "hidden_dims[2]", "name": "lstm3"},
                {"type": "Dropout", "rate": "dropout_rates[2]"},
                {"type": "Linear", "input_dim": "hidden_dims[2]", "output_dim": 32, "name": "fc1", "activation": "ReLU"},
                {"type": "Linear", "input_dim": 32, "output_dim": "output_dim", "name": "fc2"}
            ],
            "output": "next_day_return",
            "note": "마지막 시점만 사용 (h3[:, -1, :])"
        },
        # 모델/피처 관련
        # "input_dim": 169,  # 제거: 코드에서 자동 계산
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
        # period 제거 → common_params["period"] 사용
        "interval": "1d",
        # 스케일러
        "x_scaler": "StandardScaler",
        "y_scaler": "MinMaxScaler",  # 실제 사용값 (feature_range=(-1, 1))
        # 합의/수렴 관련
        "gamma": 0.5,
        "delta_limit": 0.1,
        # MacroAgent 전용 파라미터
        "fine_tune_epochs": 5,           # Fine-tuning epochs (MacroAgent)
        "searcher_buffer_days": 50,     # searcher에서 window + buffer_days (파생변수 계산용 여유분, 최적화 유지)
        # backtest_years, normal_years 제거 → common_params["period"], common_params["period_test"] 사용
        "recent_days": 14,               # 최근 며칠치 데이터 사용
        # 수익률 클리핑
        "return_clip_min": -0.5,         # 수익률 클리핑 최소값 (-50%)
        "return_clip_max": 0.5,          # 수익률 클리핑 최대값 (+50%)
        "minmax_scaler_range": (-1, 1), # MinMaxScaler feature_range
    },

    # -----------------------------------------------------------
    # SentimentalAgent: 뉴스/커뮤니티 감성 + 가격 피처
    # -----------------------------------------------------------
    "SentimentalAgent": {
        "description": "투자자 심리 및 뉴스 감성 기반 시장 예측 모델",
        # 모델 구조
        "model_architecture": {
            "type": "SentimentalLSTM",
            "layers": [
                {"type": "LSTM", "input_dim": "input_dim", "hidden_dim": "d_model", "num_layers": "num_layers", "name": "lstm"},
                {"type": "Dropout", "rate": "dropout"},
                {"type": "Linear", "input_dim": "d_model", "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return",
            "note": "SentimentalLSTM 클래스 사용, BaseAgent의 _build_model()에서 생성"
        },
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
        # period 제거 → common_params["period"] 사용
        "interval": "1d",
        # 스케일러
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        # 합의/수렴 관련
        "gamma": 0.3,               # 수렴율
        "delta_limit": 0.05,
        # Loss 함수
        "loss_fn": "HuberLoss",     # Loss function type (HuberLoss, L1Loss, MSELoss)
        # SentimentalAgent 전용 파라미터
        # run_dataset_days 제거 → common_params["period"] 사용
        # 수익률 클리핑
        "return_clip_min": -0.5,        # 수익률 클리핑 최소값 (-50%)
        "return_clip_max": 0.5,         # 수익률 클리핑 최대값 (+50%)
    },
}

dir_info = {
    "data_dir": "data/processed",
    "model_dir": "models",
    "scaler_dir": "models/scalers",
    "artifacts_dir": "artifacts" # 아연추가(필요없을시 삭제)
}
