# 🎭 Multi-Agent Debating System

3개의 전문화된 예측 에이전트(Technical/Fundamental/Sentimental)가 상호 학습을 통해 주식 가격을 예측하는 시스템입니다.

## 🏗️ 시스템 아키텍처

```
Stage 1: 사전학습 → Stage 2: 상호학습 → Stage 3: 실시간 디베이트
```

### 🎯 각 Stage별 역할
- **Stage 1**: 각 에이전트가 전문 영역 패턴 학습
- **Stage 2**: 2024 데이터로 모델 간 상호 보정 및 β 신뢰도 추정
- **Stage 3**: Monte Carlo Dropout으로 불확실성 기반 실시간 합의 예측

## 📁 핵심 파일들

### 🚀 실행 스크립트
- `train_agents.py` - Stage 1: 사전학습
- `stage2_trainer.py` - Stage 2: 상호학습  
- `debate_system.py` - Stage 3: 실시간 디베이트
- `streamlit_dashboard.py` - **🆕 실시간 대시보드**
- `single_ticker_builder.py` - **🆕 단일 주식 데이터셋 생성**
- `ticker_input_system.py` - **🆕 티커 입력 통합 시스템**

### 🔧 유틸리티
- `agent_utils.py` - 에이전트 로딩 및 관리
- `dataset_builder.py` - 데이터셋 생성
- `requirements.txt` - 필요한 패키지 목록

### 📊 데이터 & 모델
- `data/` - CSV 데이터셋들 (pretrain/mutual/test)
- `models/` - 훈련된 모델들 (.pt)

## 🚀 실행 순서

```bash
# 1. 데이터셋 생성
python dataset_builder.py

# 2. Stage 1: 사전학습
python train_agents.py

# 3. Stage 2: 상호학습
python stage2_trainer.py

# 4. Stage 3: 실시간 디베이트
python debate_system.py

# 5. 🆕 Streamlit 대시보드 실행
streamlit run streamlit_dashboard.py

# 6. 🆕 티커 입력 시스템 실행 (선택사항)
python ticker_input_system.py
```

### 🆕 대시보드 기능

#### **단일 주식 대시보드** (`streamlit_dashboard.py`)
- 📊 **실시간 성능 메트릭**: 평균 오차, 정확도 등급
- 🎯 **예측 vs 실제 차트**: 인터랙티브 시각화
- 🔄 **β 신뢰도 진화**: 가중치 변화 추이
- 🔍 **불확실성 분석**: 각 에이전트별 σ 분석
- 🎯 **에이전트 성능 비교**: 상세 성능 테이블
- 🔧 **시스템 상태**: 모델/데이터/결과 상태 모니터링

#### **🆕 티커 입력 시스템** (`ticker_input_system.py`)
- 📈 **주식 입력**: 원하는 주식 코드 입력 (AAPL, MSFT, GOOGL 등)
- 🚀 **자동 실행**: 데이터셋 생성 → 훈련 → 상호학습 → 디베이트
- 📊 **인기 주식**: 기술주, 금융주, 헬스케어, 소비재, 에너지, 한국주
- 🎯 **간편 사용**: 한 번의 입력으로 전체 파이프라인 실행

## 📈 성능 지표

- **평균 오차율**: 5.4% ± 2.44%
- **성능 개선**: Stage 2 → Stage 3로 4.7% 향상
- **정확도 등급**: 우수(50%) + 양호(50%) = 100%
- **업계 대비**: 일반 모델 대비 2-4배 우수한 성능

## 🎯 에이전트 전문성

| 에이전트 | 아키텍처 | 특징 | 최종 β 가중치 |
|---------|----------|------|---------------|
| Technical | TCN | 가격+거래량+기술지표 | 0.702 (지배적) |
| Fundamental | LSTM | 재무+거시경제지표 | 0.005 (보조) |
| Sentimental | Transformer | 감성지수+시장반응 | 0.293 (보조) |

## 🔬 핵심 기술

- **Monte Carlo Dropout**: 불확실성 추정 (σ² = Var(yᵢ⁽ᵏ⁾))
- **Peer Correction**: yᵢ' = yᵢ + αβᵢ(peer_meanᵢ - yᵢ)
- **신뢰도 계산**: βᵢ = Softmax(-σᵢ)
- **EMA 피드백**: βᵢ ← λβᵢ + (1-λ)βᵢ(new)

## 📋 요구사항

```bash
# 기본 패키지
pip install -r requirements.txt

# 또는 개별 설치
pip install torch pandas numpy matplotlib scikit-learn yfinance streamlit plotly
```

## 📄 라이선스

MIT License

