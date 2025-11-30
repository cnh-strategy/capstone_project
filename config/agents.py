# config/agents.py
# ===============================================================
# 에이전트 설정 및 하이퍼파라미터 관리
# ===============================================================
# 이 파일은 모든 에이전트(Technical, Macro, Sentimental)의 
# 학습 및 실행에 필요한 설정값을 정의합니다.
#
# 주요 구성:
# 1. common_params: 모든 에이전트가 공유하는 전역 설정
# 2. agents_info: 각 에이전트별 고유 하이퍼파라미터 및 모델 구조
# 3. dir_info: 데이터 및 모델 저장 경로
# ===============================================================

# 공통 파라미터 (모든 Agent에서 공유)
common_params = {
    # --- Monte Carlo Dropout 설정 ---
    "n_samples": 30,                    # 예측 시 샘플링 횟수 (불확실성 계산용)
    
    # --- LLM 설정 ---
    "temperature": 0.2,                 # LLM 생성 다양성 조절 (낮을수록 결정적)
    "preferred_models": ["gpt-5-mini", "gpt-4.1-mini"],  # 사용할 모델 우선순위 목록
    
    # --- 학습 및 Loss 설정 ---
    "huber_loss_delta": 1.0,            # Huber Loss의 delta 값 (이상치 민감도 조절)
    "fine_tune_lr": 1e-4,               # Fine-tuning 학습률
    "fine_tune_epochs": 10,             # Fine-tuning 에포크 수
    
    # --- 데이터 스케일링 및 처리 ---
    "y_scale_factor": 100.0,            # 수익률 타겟 스케일링 배수 (예: 0.01 -> 1.0)
    "eval_split_ratio": 0.8,            # 학습/검증 데이터 분할 비율 (0.8 = 80% 학습)
    "period": "2y",                     # 일반 모드 데이터 수집 기간
    "period_test": "2y",                # 백테스팅 모드 데이터 수집 기간
    "pretrain_save_dataset": True,      # 전처리된 데이터셋 저장 여부
    "pretrain_log_interval": 5,         # 학습 로그 출력 주기 (에포크 단위)
    
    # --- 예측 및 불확실성 ---
    "default_current_price": 100.0,     # 현재가가 없을 경우 사용할 기본값
    "sigma_min": 1e-6,                  # 불확실성(표준편차) 최소값 (0 나누기 방지)
    "confidence_formula": "1.0 / (1.0 + sigma)",  # 신뢰도 계산 공식 (sigma가 클수록 신뢰도 하락)
}

# 에이전트별 상세 설정
agents_info = {
    # -----------------------------------------------------------
    # 1. TechnicalAgent: 기술적 지표 기반 예측 (LSTM + Time-Attention)
    # -----------------------------------------------------------
    "TechnicalAgent": {
        "description": "기술적 지표(RSI, MACD 등)와 가격 데이터를 분석하여 예측",
        
        # 모델 아키텍처 정보 (참고용)
        "model_architecture": {
            "type": "LSTM_with_TimeAttention",
            "layers": [
                {"type": "LSTM", "input_dim": 13, "hidden_dim": 64, "name": "lstm1"},
                {"type": "LSTM", "input_dim": 64, "hidden_dim": 32, "name": "lstm2"},
                {"type": "TimeAttention", "hidden_dim": 32, "name": "attn_vec"},
                {"type": "Linear", "input_dim": 32, "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return"
        },
        
        # 사용할 데이터 컬럼 (기술적 지표)
        "data_cols": [
            "weekofyear_sin", "weekofyear_cos", "log_ret_lag1",
            "ret_3d", "mom_10", "ma_200",
            "macd", "bbp", "adx_14",
            "obv", "vol_ma_20", "vol_chg", "vol_20d"
        ],
        "feature_builder": "core.technical_classes.technical:build_features_technical",
        
        # 모델 하이퍼파라미터
        "input_dim": 13,
        "window_size": 20,              # 시계열 윈도우 크기 (Lookback period)
        "rnn_units1": 64,               # LSTM 1층 히든 유닛 수
        "rnn_units2": 32,               # LSTM 2층 히든 유닛 수
        "dropout": 0.18778570103014075, # Dropout 비율
        "epochs": 45,                   # 학습 에포크 수
        "patience": 8,                  # Early Stopping 인내값
        "learning_rate": 4.2471233429729313e-4, # 학습률
        "batch_size": 64,               # 배치 크기
        
        # 설정 및 기타
        "interval": "1d",               # 데이터 주기
        "x_scaler": "MinMaxScaler",     # 입력 데이터 스케일러
        "y_scaler": "StandardScaler",   # 타겟 데이터 스케일러
        "loss_fn": "HuberLoss",         # 손실 함수
        "seed": 1234,                   # 랜덤 시드
        
        # Debate 관련 파라미터
        "gamma": 0.3,                   # 의견 수렴율 (높을수록 타 에이전트 의견 수용도 높음)
        "delta_limit": 0.05,            # 최대 변화 허용 폭
        
        # TechnicalAgent 전용 설정
        "fine_tune_epochs": 20,         # Revise 단계 Fine-tuning 에포크
        "return_clip_min": -0.5,        # 수익률 클리핑 하한 (-50%)
        "return_clip_max": 0.5,         # 수익률 클리핑 상한 (+50%)
        
        # 설명가능성(XAI) 파라미터
        "occlusion_batch_size": 32,     # Occlusion 분석 배치 크기
        "top_k_features": 5,            # 주요 피처 추출 개수
        "shap_weight_time": 0.20,       # 시간 중요도 가중치 (SHAP)
        "shap_weight_feat": 0.30,       # 피처 중요도 가중치 (SHAP)
        "attention_weights": [0.4, 0.25, 0.15],  # 시간 중요도 융합 가중치 [Attn, Grad, Occ]
        "feature_weights": [0.5, 0.2],           # 피처 중요도 융합 가중치 [Grad, Occ]
        "pack_idea_top_time": 8,        # 설명 압축 시 포함할 상위 시간대 수
        "pack_idea_top_feat": 6,        # 설명 압축 시 포함할 상위 피처 수
        "pack_idea_coverage": 0.8,      # 설명 압축 커버리지 비율
    },

    # -----------------------------------------------------------
    # 2. MacroAgent: 거시경제 지표 기반 예측 (Stacked LSTM)
    # -----------------------------------------------------------
    "MacroAgent": {
        "description": "금리, 환율, 유가 등 거시경제 지표와 시장 심리 지수를 분석",
        
        # 모델 아키텍처 정보
        "model_architecture": {
            "type": "LSTM_Stacked_Dense",
            "layers": [
                {"type": "LSTM", "input_dim": "input_dim", "hidden_dim": 128, "name": "lstm1"},
                {"type": "LSTM", "input_dim": 128, "hidden_dim": 64, "name": "lstm2"},
                {"type": "LSTM", "input_dim": 64, "hidden_dim": 32, "name": "lstm3"},
                {"type": "Linear", "input_dim": 32, "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return"
        },
        
        # 데이터 컬럼 (참고용, 실제는 MacroAgent 내부에서 자동 생성)
        "data_cols": [
            "Open", "High", "Low", "Close", "Volume",
            "returns", "sma_5", "sma_20", "rsi", "volume_z",
            "USD_KRW", "NASDAQ", "VIX"
        ],
        
        # 모델 하이퍼파라미터
        "hidden_dims": [128, 64, 32],   # LSTM 3개 층 히든 사이즈
        "dropout_rates": [0.3, 0.3, 0.2], # 각 층별 Dropout 비율
        "window_size": 40,              # 시계열 윈도우 크기
        "epochs": 60,
        "patience": 10,
        "learning_rate": 0.0005,
        "batch_size": 16,
        
        # 설정 및 기타
        "interval": "1d",
        "x_scaler": "StandardScaler",
        "y_scaler": "MinMaxScaler",     # 타겟은 -1 ~ 1 범위로 스케일링
        "loss_fn": "L1Loss",            # L1 Loss (MAE) 사용
        
        # Debate 관련 파라미터
        "gamma": 0.5,
        "delta_limit": 0.1,
        
        # MacroAgent 전용 설정
        "fine_tune_epochs": 5,
        "searcher_buffer_days": 50,     # 데이터 수집 시 여유 기간 (지표 계산용)
        "recent_days": 14,              # 최근 데이터 조회 기간
        "return_clip_min": -0.5,
        "return_clip_max": 0.5,
        "minmax_scaler_range": (-1, 1), # MinMaxScaler 범위
    },

    # -----------------------------------------------------------
    # 3. SentimentalAgent: 뉴스 및 감성 분석 (LSTM)
    # -----------------------------------------------------------
    "SentimentalAgent": {
        "description": "뉴스 헤드라인과 감성 점수, 거래량 변동 등을 분석",
        
        # 모델 아키텍처 정보
        "model_architecture": {
            "type": "SentimentalLSTM",
            "layers": [
                {"type": "LSTM", "input_dim": 8, "hidden_dim": 64, "num_layers": 2, "name": "lstm"},
                {"type": "Linear", "input_dim": 64, "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return"
        },
        
        # 데이터 컬럼
        "data_cols": [
            "returns", "sentiment_mean", "sentiment_vol",
            "Close", "Volume", "Open", "High", "Low"
        ],
        
        # 모델 하이퍼파라미터
        "input_dim": 8,
        "d_model": 64,                  # LSTM 히든 사이즈
        "nhead": 4,                     # (참고용) Attention 헤드 수
        "num_layers": 2,                # LSTM 층 수
        "dropout": 0.2,
        "window_size": 20,
        "epochs": 50,
        "learning_rate": 0.0005,
        "batch_size": 32,
        
        # 설정 및 기타
        "interval": "1d",
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "loss_fn": "HuberLoss",
        
        # Debate 관련 파라미터
        "gamma": 0.3,
        "delta_limit": 0.05,
        
        # SentimentalAgent 전용 설정
        "return_clip_min": -0.5,
        "return_clip_max": 0.5,
    },
}

# 디렉토리 설정
dir_info = {
    "data_dir": "data/processed",       # 전처리된 데이터 저장 경로
    "model_dir": "models",              # 학습된 모델 저장 경로
    "scaler_dir": "models/scalers",     # 스케일러 저장 경로
    "artifacts_dir": "artifacts"        # 기타 결과물 저장 경로
}
