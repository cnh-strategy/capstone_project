# Git Main 브랜치 vs 현재 작업 브랜치 변경사항 분석 리포트

**생성일**: 2025-01-XX  
**분석 범위**: Git Main 브랜치와 현재 작업 디렉토리 간 차이점

---

## 📋 목차

1. [전체 변경사항 요약](#1-전체-변경사항-요약)
2. [현재와 이전 버전의 차이점](#2-현재와-이전-버전의-차이점)
3. [각 에이전트별 담당자 확인 사항](#3-각-에이전트별-담당자-확인-사항)
4. [각 에이전트별 개선이 필요한 방향](#4-각-에이전트별-개선이-필요한-방향)

---

## 1. 전체 변경사항 요약

### 1.1 파일 변경 통계
- **수정된 파일**: 15개
- **삭제된 파일**: 28개
- **추가된 파일**: 6개 (추적되지 않은 파일)

### 1.2 주요 변경 카테고리
1. **에이전트 아키텍처 리팩토링**
   - `MacroSentiAgent` → `MacroAgent`로 명칭 변경
   - `TechnicalBaseAgent` 제거 및 `BaseAgent`로 통합
   - 모든 에이전트가 `nn.Module`을 직접 상속하도록 변경

2. **모델 프레임워크 전환**
   - TensorFlow/Keras → PyTorch로 완전 전환
   - 모델 파일 확장자: `.keras` → `.pt`

3. **코드 구조 개선**
   - 중복 코드 제거 및 모듈 통합
   - 설정 기반 하이퍼파라미터 관리 강화

4. **뉴스 데이터 수집 기능 강화**
   - EODHD API 실제 연동 구현
   - 뉴스 캐시 시스템 개선

---

## 2. 현재와 이전 버전의 차이점

### 2.1 아키텍처 변경사항

#### 2.1.1 BaseAgent 변경사항
**이전 버전:**
- API 키가 없을 때 빈 문자열로 처리
- `StockData`에 `MacroSentiAgent` 필드 사용

**현재 버전:**
- API 키가 없을 때 `RuntimeError` 발생 (명시적 에러 처리)
- `StockData`에 `MacroAgent` 필드로 변경
- `feature_cols` 필드 추가 (피처 컬럼 목록 관리)

**주요 변경 코드:**
```python
# 이전
self.api_key = os.getenv("CAPSTONE_OPENAI_API")
if not self.api_key:
    self.api_key = ""

# 현재
self.api_key = os.getenv("CAPSTONE_OPENAI_API")
if not self.api_key:
    raise RuntimeError("환경변수 CAPSTONE_OPENAI_API가 설정되지 않았습니다.")
```

#### 2.1.2 DebateAgent 변경사항
**이전 버전:**
- `MacroSentiAgent` 사용
- `macro_sercher` 함수 직접 호출
- 각 에이전트별로 다른 데이터 로딩 로직

**현재 버전:**
- `MacroAgent` 사용
- 통일된 에이전트 인터페이스 (`searcher`, `pretrain`, `predict`, `reviewer_draft`)
- `_check_agent_ready()` 메서드로 모델 준비 상태 확인
- `_data_built` 플래그로 데이터셋 생성 상태 관리

**주요 변경 코드:**
```python
# 이전
"MacroSentiAgent": MacroPredictor(
    agent_id="MacroSentiAgent",
    ticker=ticker,
    base_date=datetime.today(),
    window=40,
)

# 현재
"MacroAgent": MacroAgent(
    agent_id="MacroAgent",
    ticker=ticker,
    base_date=datetime.today(),
    window=macro_window,  # Config에서 가져옴
)
```

### 2.2 MacroAgent (이전 MacroSentiAgent) 변경사항

#### 2.2.1 프레임워크 전환
**이전 버전:**
- TensorFlow/Keras 기반
- `load_model()` 사용
- 모델 파일: `.keras`

**현재 버전:**
- PyTorch 기반 (`nn.Module` 상속)
- `torch.load()` 사용
- 모델 파일: `.pt`
- 3층 LSTM 구조 (hidden_dims: [128, 64, 32])

#### 2.2.2 모델 구조 변경
**이전:**
```python
# TensorFlow/Keras
self.model = load_model(self.model_path, compile=False)
```

**현재:**
```python
# PyTorch
class MacroAgent(BaseAgent, nn.Module):
    def __init__(self, ...):
        nn.Module.__init__(self)
        self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
        # ...
```

#### 2.2.3 설정 기반 하이퍼파라미터
**변경사항:**
- `window_size`: 14 → 40
- `epochs`: 50 → 60
- `learning_rate`: 1e-4 → 0.0005 (5e-4)
- `y_scaler`: StandardScaler → MinMaxScaler
- `hidden_dims`: 단일 값 → [128, 64, 32] (3층)
- `dropout_rates`: 단일 값 → [0.3, 0.3, 0.2] (레이어별)

### 2.3 TechnicalAgent 변경사항

#### 2.3.1 베이스 클래스 변경
**이전 버전:**
- `TechnicalBaseAgent` 상속
- 별도 베이스 클래스 파일 존재

**현재 버전:**
- `BaseAgent` 직접 상속
- `TechnicalBaseAgent` 파일 삭제 및 기능 통합
- `searcher()` 메서드를 TechnicalAgent 내부로 이동

#### 2.3.2 Attention 메커니즘 개선
**변경사항:**
- `time_attention()` 메서드에서 차원 처리 개선
- 1차원 배열 반환 보장 로직 추가
- `per_time`, `per_feat` 차원 검증 추가

**주요 코드:**
```python
# 현재 버전에 추가된 안전성 검사
w = attn[0].abs().cpu().numpy()
w = w.flatten() if w.ndim > 1 else w
result = w / s if s > 0 else np.ones_like(w) / len(w)
return result.flatten() if result.ndim > 1 else result
```

#### 2.3.3 데이터 관리 개선
- `_last_idea` 인스턴스 변수 추가 (TechnicalAgent 전용 설명 정보 저장)
- `idea` 저장 방식 변경 (Target.idea → self._last_idea)

### 2.4 SentimentalAgent 변경사항

#### 2.4.1 모델 구조 통합
**이전 버전:**
- `SentimentalNet` 별도 클래스 정의

**현재 버전:**
- `SentimentalNet` 제거, `SentimentalAgent`가 `nn.Module` 직접 상속
- 모델 레이어를 `SentimentalAgent` 내부에 정의

#### 2.4.2 뉴스 데이터 수집 강화
**이전 버전:**
- `_fetch_news_from_eodhd_stub()`: 빈 리스트만 반환 (스텁)

**현재 버전:**
- `EODHDNewsClient` 실제 연동 구현
- 뉴스 캐시 시스템 개선 (정확한 기간 → 최신 파일 fallback → 실제 수집)
- 에러 처리 및 로깅 강화

**주요 변경 코드:**
```python
# 이전
def _fetch_news_from_eodhd_stub(...):
    print("[FinBERT] _fetch_news_from_eodhd_stub 호출됨 → 실제 EODHD 연동은 별도 구현 필요")
    return []

# 현재
def _fetch_news_from_eodhd_stub(...):
    from core.sentimental_classes.eodhd_client import EODHDNewsClient
    client = EODHDNewsClient(api_key=api_key)
    news_items = client.fetch_company_news(...)
    # 실제 뉴스 데이터 반환
```

#### 2.4.3 뉴스 피처 생성 프로세스 개선
**변경사항:**
1. 캐시 파일 확인 로직 개선
2. 캐시가 없을 때 실제 EODHD에서 뉴스 수집
3. 디렉토리 자동 생성 (`base.mkdir(parents=True, exist_ok=True)`)
4. 에러 처리 및 traceback 출력 추가

### 2.5 설정 파일 (config/agents.py) 변경사항

#### 2.5.1 에이전트 명칭 변경
- `MacroSentiAgent` → `MacroAgent`

#### 2.5.2 MacroAgent 하이퍼파라미터 업데이트
```python
# 이전
"MacroSentiAgent": {
    "input_dim": 13,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "window_size": 14,
    "epochs": 50,
    "learning_rate": 1e-4,
    "y_scaler": "StandardScaler",
}

# 현재
"MacroAgent": {
    "input_dim": 13,
    "hidden_dims": [128, 64, 32],  # 3층 LSTM
    "dropout_rates": [0.3, 0.3, 0.2],  # 레이어별
    "window_size": 40,
    "epochs": 60,
    "patience": 10,  # Early stopping 추가
    "learning_rate": 0.0005,
    "loss_fn": "L1Loss",  # Loss function 명시
    "y_scaler": "MinMaxScaler",
}
```

### 2.6 삭제된 파일들

#### 2.6.1 에이전트 관련
- `agents/fundamental_agent.py` - Fundamental 에이전트 제거
- `agents/fundamental_reviewer.py` - Fundamental 리뷰어 제거
- `core/technical_classes/technical_base_agent.py` - BaseAgent로 통합

#### 2.6.2 감성 분석 관련
- `core/sentimental_classes/features.py`
- `core/sentimental_classes/lstm.py`
- `core/sentimental_classes/news.py`
- `core/sentimental_classes/price.py`
- `core/sentimental_classes/utils.py`
- `core/sentimental_classes/utils_datetime.py`

#### 2.6.3 유틸리티
- `core/utils_datetime.py`

#### 2.6.4 스크립트 및 데모
- `scripts/__init__.py`
- `scripts/sentimental_demo/__init__.py`
- `scripts/sentimental_demo/build_news_features.py`
- `scripts/sentimental_demo/demo_ctx_prompt.py`
- `scripts/sentimental_demo/test_eodhd_news.py`

#### 2.6.5 모델 파일 (삭제됨)
- 모든 `.pt` 모델 파일 (RZLV, TSLA)
- 모든 스케일러 파일 (`.pkl`)

**⚠️ 주의**: 모델 파일들이 삭제되었으므로 재학습이 필요할 수 있습니다.

#### 2.6.6 노트북 파일
- `macro_test.ipynb`
- `technical_test.ipynb`
- `technical_test_1114.ipynb`
- `test_sentimental_pipeline.ipynb`

### 2.7 추가된 파일들

#### 2.7.1 문서
- `LLM_NEWS_USAGE.md` - LLM에 전달되는 뉴스 데이터 분석 문서
- `NEWS_COLLECTION_FLOW.md` - 뉴스 수집 프로세스 문서
- `NEWS_DATA_USAGE.md` - 뉴스 데이터 사용 가이드

#### 2.7.2 테스트 파일
- `test_debate_nvda.py` - DebateAgent 테스트
- `test_news_collection.py` - 뉴스 수집 테스트

#### 2.7.3 데이터 디렉토리
- `data/` - 데이터 디렉토리 (추적되지 않음)

#### 2.7.4 노트북 디렉토리
- `notebooks/` - 노트북 디렉토리 (추적되지 않음)

### 2.8 기타 변경사항

#### 2.8.1 core/data_set.py
- `MacroSentiAgent` → `MacroAgent` 참조 변경

#### 2.8.2 requirements.txt
- TensorFlow 의존성 제거 예상
- PyTorch 의존성 추가/업데이트 예상

#### 2.8.3 streamlit_dashboard.py
- 에이전트 명칭 변경 반영

---

## 3. 각 에이전트별 담당자 확인 사항

### 3.1 TechnicalAgent 담당자 확인 사항

#### ✅ 필수 확인 항목

1. **모델 파일 재생성 필요 여부**
   - 기존 `.pt` 모델 파일들이 삭제되었습니다
   - `models/TSLA_TechnicalAgent.pt`, `models/RZLV_TechnicalAgent.pt` 등 재학습 필요
   - 스케일러 파일도 함께 재생성 필요

2. **Attention 메커니즘 동작 검증**
   - `time_attention()` 메서드의 차원 처리 로직 변경
   - 실제 예측 시 attention 가중치가 올바르게 계산되는지 확인
   - `per_time`, `per_feat` 차원 검증 로직이 예상대로 동작하는지 테스트

3. **BaseAgent 통합 영향도**
   - `TechnicalBaseAgent` 제거로 인한 기능 누락 여부 확인
   - `searcher()` 메서드가 TechnicalAgent 내부로 이동한 것의 동작 확인
   - 기존에 `TechnicalBaseAgent`에만 있던 기능이 모두 통합되었는지 확인

4. **데이터 로딩 프로세스**
   - `build_dataset_tech()`, `load_dataset_tech()` 함수 사용 확인
   - 데이터셋 경로 및 형식이 변경되지 않았는지 확인

5. **설정 파일 검증**
   - `config/agents.py`의 TechnicalAgent 설정값이 실제 사용값과 일치하는지 확인
   - `window_size: 55`, `rnn_units1: 64`, `rnn_units2: 32` 등

#### ⚠️ 주의사항

- `_last_idea` 저장 방식 변경으로 인한 호환성 문제 가능성
- 기존 코드에서 `target.idea`를 참조하는 부분이 있다면 수정 필요

### 3.2 MacroAgent 담당자 확인 사항

#### ✅ 필수 확인 항목

1. **프레임워크 전환 검증 (최우선)**
   - TensorFlow → PyTorch 전환으로 인한 모델 호환성 문제
   - **기존 `.keras` 모델 파일은 더 이상 사용 불가**
   - **모든 모델 재학습 필수**

2. **모델 구조 변경 검증**
   - 3층 LSTM 구조 (hidden_dims: [128, 64, 32])로 변경
   - 기존 2층 구조와의 성능 비교 필요
   - `dropout_rates: [0.3, 0.3, 0.2]` 적용 검증

3. **하이퍼파라미터 변경 영향도**
   - `window_size: 14 → 40` (약 3배 증가)
   - `learning_rate: 1e-4 → 5e-4` (5배 증가)
   - `epochs: 50 → 60` (20% 증가)
   - `y_scaler: StandardScaler → MinMaxScaler`
   - **이러한 변경이 예측 성능에 미치는 영향 평가 필요**

4. **스케일러 파일 재생성**
   - `xscaler.pkl`, `yscaler.pkl` 파일 재생성 필요
   - 스케일러 경로: `models/scalers/{ticker}_MacroAgent_{x|y}scaler.pkl`

5. **데이터 로딩 프로세스**
   - `macro_dataset()` 함수 호출 경로 확인
   - `core/macro_classes/macro_class_dataset.py` 모듈 동작 확인
   - 데이터 컬럼 수가 `input_dim: 13`과 일치하는지 확인

6. **Config 기반 초기화**
   - `agents_info["MacroAgent"]`에서 하이퍼파라미터를 읽어오는 로직 검증
   - `window` 파라미터가 Config에서 올바르게 가져와지는지 확인

#### ⚠️ 주의사항

- **모델 파일이 완전히 삭제되었으므로 프로덕션 환경에서 즉시 사용 불가**
- 재학습 후 성능 검증 필수
- 기존 모델과의 예측 결과 비교 필요

### 3.3 SentimentalAgent 담당자 확인 사항

#### ✅ 필수 확인 항목

1. **뉴스 데이터 수집 기능 검증 (최우선)**
   - `EODHDNewsClient` 실제 연동 동작 확인
   - `EODHD_API_KEY` 환경변수 설정 확인
   - 뉴스 수집 실패 시 fallback 동작 확인
   - 뉴스 캐시 시스템이 올바르게 동작하는지 확인

2. **FinBERT 분석 프로세스**
   - `build_finbert_news_features()` 함수의 변경사항 검증
   - 뉴스가 없을 때 0 피처 반환 로직 확인
   - 감성 분석 결과가 예측에 올바르게 반영되는지 확인

3. **모델 구조 통합**
   - `SentimentalNet` 클래스 제거로 인한 영향 확인
   - `SentimentalAgent`가 `nn.Module`을 직접 상속하는 구조 검증
   - 모델 레이어 정의가 올바른지 확인

4. **데이터 피처 생성**
   - 뉴스 감성 피처가 올바른 형식으로 생성되는지 확인
   - `feature_cols`에 감성 피처가 포함되는지 확인
   - LLM에 전달되는 뉴스 데이터 형식 확인 (LLM_NEWS_USAGE.md 참조)

5. **캐시 시스템**
   - 뉴스 캐시 파일 경로 및 형식 확인
   - 캐시 파일이 없을 때 자동 수집 동작 확인
   - 캐시 파일 로드 실패 시 처리 로직 확인

#### ⚠️ 주의사항

- EODHD API 키가 없으면 뉴스 수집이 실패하므로 환경변수 설정 필수
- 뉴스 수집 실패 시 0 피처로 대체되므로 예측 성능에 영향 가능
- `LLM_NEWS_USAGE.md` 문서에 따르면 Opinion 생성 시에는 뉴스 데이터가 포함되지 않음 (Rebuttal/Revision에서만 포함)

### 3.4 DebateAgent 담당자 확인 사항

#### ✅ 필수 확인 항목

1. **에이전트 초기화 프로세스**
   - `MacroSentiAgent` → `MacroAgent` 변경 반영 확인
   - Config에서 `window_size`를 가져오는 로직 검증
   - 모든 에이전트가 올바르게 초기화되는지 확인

2. **통일된 인터페이스 사용**
   - 모든 에이전트가 `searcher()`, `pretrain()`, `predict()`, `reviewer_draft()` 메서드를 사용하는지 확인
   - `_check_agent_ready()` 메서드로 모델 준비 상태 확인 로직 검증
   - 에이전트별로 다른 데이터 로딩 로직이 제거되었는지 확인

3. **데이터셋 생성 관리**
   - `_data_built` 플래그로 중복 데이터셋 생성 방지 확인
   - `rebuild` 파라미터 기본값이 `False`로 변경된 영향 확인
   - `force_pretrain` 파라미터 기본값이 `False`로 변경된 영향 확인

4. **에러 처리**
   - 모델 로드 실패 시 처리 로직 확인
   - 에이전트별 예외 처리 검증

5. **Opinion 수집 프로세스**
   - 각 에이전트의 Opinion이 올바르게 수집되는지 확인
   - Opinion 형식이 변경되지 않았는지 확인

#### ⚠️ 주의사항

- `get_opinion()` 메서드의 `rebuild`, `force_pretrain` 기본값 변경으로 인한 동작 차이 가능
- 모든 에이전트의 모델이 준비되어 있어야 정상 동작

### 3.5 공통 확인 사항

#### ✅ 필수 확인 항목

1. **환경변수 설정**
   - `CAPSTONE_OPENAI_API`: 필수 (없으면 RuntimeError 발생)
   - `EODHD_API_KEY`: SentimentalAgent 뉴스 수집에 필요

2. **모델 파일 재생성**
   - 모든 에이전트의 모델 파일 (`.pt`) 재학습 필요
   - 스케일러 파일 (`.pkl`) 재생성 필요

3. **의존성 패키지**
   - `requirements.txt` 변경사항 확인
   - TensorFlow 제거, PyTorch 버전 확인
   - 새로운 패키지 설치 필요 여부 확인

4. **데이터 디렉토리 구조**
   - `data/processed/` 디렉토리 구조 확인
   - 데이터셋 파일 형식 변경 여부 확인

5. **문서화**
   - 새로 추가된 문서 파일 검토:
     - `LLM_NEWS_USAGE.md`
     - `NEWS_COLLECTION_FLOW.md`
     - `NEWS_DATA_USAGE.md`

---

## 4. 각 에이전트별 개선이 필요한 방향

### 4.1 TechnicalAgent 개선 방향

#### 4.1.1 프로세스 개선

1. **모델 재학습 자동화**
   - 현재: 모델 파일이 없으면 수동으로 `pretrain()` 호출 필요
   - 개선: 모델 파일이 없거나 오래된 경우 자동 재학습 트리거
   - 개선: 모델 버전 관리 시스템 도입 (학습 날짜, 하이퍼파라미터 해시 등)

2. **데이터 검증 강화**
   - 현재: 데이터셋 로드 시 기본적인 검증만 수행
   - 개선: 데이터 품질 검증 (결측치, 이상치, 분포 검사)
   - 개선: 데이터 스키마 검증 (필수 컬럼 존재 여부, 데이터 타입 확인)

3. **에러 처리 개선**
   - 현재: 일부 예외 상황에서 기본값 반환
   - 개선: 명시적인 에러 메시지 및 로깅
   - 개선: 에러 복구 메커니즘 (예: 데이터 재수집, 모델 재학습)

4. **로깅 및 모니터링**
   - 현재: `print()` 문으로 로깅
   - 개선: 구조화된 로깅 시스템 (로깅 레벨, 파일 출력)
   - 개선: 예측 성능 메트릭 자동 수집 및 저장

#### 4.1.2 데이터 개선

1. **피처 엔지니어링**
   - 현재: 13개 기술적 지표 사용
   - 개선: 추가 기술적 지표 실험 (MACD 히스토그램, 스토캐스틱 등)
   - 개선: 피처 중요도 분석을 통한 불필요한 피처 제거

2. **데이터 품질 관리**
   - 현재: yfinance에서 데이터 수집, 기본적인 전처리만 수행
   - 개선: 데이터 수집 실패 시 대체 데이터 소스 활용
   - 개선: 데이터 정규화 및 이상치 처리 개선

3. **시계열 데이터 보강**
   - 현재: 5년 데이터 사용
   - 개선: 더 긴 기간 데이터 수집 (10년 이상)
   - 개선: 다양한 시장 상황 포함 (불황, 호황, 변동성 높은 시기)

#### 4.1.3 성능 개선

1. **모델 아키텍처 최적화**
   - 현재: 2층 LSTM + Time-Attention
   - 개선: Attention 메커니즘 개선 (Multi-Head Attention 등)
   - 개선: Transformer 기반 모델 실험
   - 개선: 앙상블 모델 (여러 모델의 예측 결합)

2. **하이퍼파라미터 튜닝**
   - 현재: 고정된 하이퍼파라미터 사용
   - 개선: 자동 하이퍼파라미터 최적화 (Optuna, Ray Tune 등)
   - 개선: 교차 검증을 통한 일반화 성능 평가

3. **예측 불확실성 정량화**
   - 현재: Monte Carlo Dropout 사용
   - 개선: 베이지안 신경망 도입
   - 개선: 예측 구간(confidence interval) 제공

4. **학습 효율성**
   - 현재: 전체 데이터셋으로 학습
   - 개선: 증분 학습 (Incremental Learning) 지원
   - 개선: 전이 학습 (Transfer Learning) 활용

### 4.2 MacroAgent 개선 방향

#### 4.2.1 프로세스 개선

1. **모델 마이그레이션 완료 검증**
   - 현재: TensorFlow → PyTorch 전환 완료
   - 개선: 기존 TensorFlow 모델과의 예측 결과 비교 스크립트 작성
   - 개선: 모델 변환 검증 테스트 자동화

2. **하이퍼파라미터 변경 영향 분석**
   - 현재: window_size, learning_rate 등 대폭 변경
   - 개선: 변경 전후 성능 비교 리포트 생성
   - 개선: 하이퍼파라미터 민감도 분석

3. **데이터 수집 자동화**
   - 현재: 거시경제 데이터 수집 프로세스 확인 필요
   - 개선: 데이터 수집 스케줄링 (일일 자동 업데이트)
   - 개선: 데이터 수집 실패 시 알림 시스템

4. **모델 버전 관리**
   - 현재: 단일 모델 파일만 저장
   - 개선: 모델 버전 관리 (학습 날짜, 성능 메트릭 포함)
   - 개선: A/B 테스트를 위한 다중 모델 관리

#### 4.2.2 데이터 개선

1. **거시경제 지표 확장**
   - 현재: USD_KRW, NASDAQ, VIX 등 기본 지표
   - 개선: 추가 거시경제 지표 도입 (금리, 인플레이션, GDP 등)
   - 개선: 섹터별 지수 추가 (에너지, 기술주 등)

2. **데이터 정규화 개선**
   - 현재: StandardScaler (X), MinMaxScaler (Y)
   - 개선: RobustScaler 실험 (이상치에 강건)
   - 개선: 시계열 특성을 고려한 정규화 방법 (예: Z-score with rolling window)

3. **데이터 품질 관리**
   - 현재: 기본적인 데이터 로드만 수행
   - 개선: 거시경제 데이터의 지연(lag) 처리
   - 개선: 데이터 출처별 신뢰도 가중치 적용

4. **외생 변수 통합**
   - 현재: 주가와 거시경제 지표만 사용
   - 개선: 뉴스 감성 지표 통합
   - 개선: 소셜 미디어 지표 통합

#### 4.2.3 성능 개선

1. **모델 아키텍처 최적화**
   - 현재: 3층 LSTM (128, 64, 32)
   - 개선: Attention 메커니즘 추가 실험
   - 개선: Temporal Convolutional Network (TCN) 실험
   - 개선: Transformer 기반 모델 실험

2. **하이퍼파라미터 최적화**
   - 현재: window_size=40, learning_rate=0.0005
   - 개선: Grid Search 또는 Bayesian Optimization
   - 개선: 교차 검증을 통한 최적 하이퍼파라미터 탐색

3. **앙상블 방법**
   - 현재: 단일 모델 사용
   - 개선: 여러 window_size를 가진 모델 앙상블
   - 개선: 다른 아키텍처 모델 앙상블

4. **학습 안정성**
   - 현재: Early stopping (patience=10) 사용
   - 개선: Learning rate scheduling
   - 개선: Gradient clipping 추가

5. **예측 해석 가능성**
   - 현재: Gradient-based feature importance 사용
   - 개선: SHAP 값 계산 추가
   - 개선: Attention 가중치 시각화

### 4.3 SentimentalAgent 개선 방향

#### 4.3.1 프로세스 개선

1. **뉴스 수집 안정성 강화**
   - 현재: EODHD API 연동 완료
   - 개선: 여러 뉴스 소스 통합 (EODHD, Alpha Vantage, NewsAPI 등)
   - 개선: 뉴스 수집 실패 시 재시도 로직
   - 개선: 뉴스 수집 상태 모니터링 대시보드

2. **캐시 시스템 최적화**
   - 현재: 파일 기반 캐시
   - 개선: Redis 등 인메모리 캐시 도입
   - 개선: 캐시 만료 정책 명확화
   - 개선: 캐시 히트율 모니터링

3. **FinBERT 분석 파이프라인**
   - 현재: 뉴스별로 순차 처리
   - 개선: 배치 처리로 성능 향상
   - 개선: GPU 가속 활용
   - 개선: 분석 결과 캐싱

4. **에러 처리 및 복구**
   - 현재: 뉴스 수집 실패 시 0 피처 반환
   - 개선: 부분 실패 시 기존 캐시 활용
   - 개선: 에러 로깅 및 알림 시스템

#### 4.3.2 데이터 개선

1. **뉴스 데이터 품질 관리**
   - 현재: 기본적인 뉴스 수집만 수행
   - 개선: 뉴스 중복 제거
   - 개선: 뉴스 관련성 필터링 (종목과 직접 관련된 뉴스만)
   - 개선: 뉴스 출처 신뢰도 가중치 적용

2. **감성 분석 정확도 향상**
   - 현재: FinBERT 기본 모델 사용
   - 개선: 도메인 특화 Fine-tuning (금융 뉴스에 특화)
   - 개선: 다중 감성 분석 모델 앙상블
   - 개선: 감성 점수 보정 (과거 예측 성능 기반)

3. **피처 엔지니어링**
   - 현재: 기본 감성 통계 (mean, volatility 등)
   - 개선: 시간 가중 감성 점수 (최근 뉴스에 더 높은 가중치)
   - 개선: 뉴스 볼륨 지표 추가
   - 개선: 섹터별 감성 분석

4. **데이터 보강**
   - 현재: 뉴스 데이터만 사용
   - 개선: 소셜 미디어 데이터 통합 (트위터, 레딧 등)
   - 개선: 애널리스트 리포트 감성 분석
   - 개선: 기업 공시 데이터 분석

#### 4.3.3 성능 개선

1. **모델 아키텍처 최적화**
   - 현재: LSTM 기반 모델
   - 개선: Transformer 기반 모델 실험
   - 개선: Attention 메커니즘을 활용한 뉴스 중요도 가중치

2. **감성 피처 활용 개선**
   - 현재: 일별 집계된 감성 통계만 사용
   - 개선: 시계열 감성 추세 분석
   - 개선: 감성 변화율(rate of change) 피처 추가
   - 개선: 감성과 가격의 상관관계 분석

3. **실시간 처리**
   - 현재: 배치 처리 중심
   - 개선: 실시간 뉴스 스트리밍 처리
   - 개선: 증분 학습 (새로운 뉴스 데이터로 모델 업데이트)

4. **예측 정확도 향상**
   - 현재: 단일 모델 사용
   - 개선: 앙상블 모델 (여러 window_size, 여러 아키텍처)
   - 개선: 메타 학습 (Meta-Learning) 활용

5. **LLM 통합 개선**
   - 현재: Opinion 생성 시 뉴스 데이터 미포함
   - 개선: Opinion 생성 시에도 주요 뉴스 요약 포함
   - 개선: 뉴스 기반 근거 생성 자동화

### 4.4 DebateAgent 개선 방향

#### 4.4.1 프로세스 개선

1. **에이전트 관리 자동화**
   - 현재: 수동으로 에이전트 초기화 및 관리
   - 개선: 에이전트 상태 모니터링 시스템
   - 개선: 에이전트별 성능 메트릭 자동 수집
   - 개선: 에이전트 장애 시 자동 복구

2. **토론 프로세스 최적화**
   - 현재: 고정된 라운드 수 (기본 3라운드)
   - 개선: 동적 라운드 수 (수렴 시 조기 종료)
   - 개선: 에이전트별 가중치 자동 조정
   - 개선: 토론 품질 평가 메트릭

3. **데이터셋 관리**
   - 현재: `_data_built` 플래그로 중복 방지
   - 개선: 데이터셋 버전 관리
   - 개선: 데이터셋 무결성 검증
   - 개선: 데이터셋 업데이트 스케줄링

4. **에러 처리 및 복구**
   - 현재: 기본적인 예외 처리
   - 개선: 에이전트별 에러 분류 및 처리
   - 개선: 부분 실패 시 다른 에이전트로 대체
   - 개선: 에러 로깅 및 알림 시스템

#### 4.4.2 데이터 개선

1. **에이전트 간 데이터 공유**
   - 현재: 각 에이전트가 독립적으로 데이터 수집
   - 개선: 공통 데이터 캐시 시스템
   - 개선: 에이전트 간 데이터 일관성 검증

2. **예측 결과 통합 개선**
   - 현재: 단순 평균 또는 가중 평균
   - 개선: 베이지안 모델 평균 (BMA)
   - 개선: 에이전트 신뢰도 기반 동적 가중치
   - 개선: 앙상블 메타 학습

3. **토론 컨텍스트 관리**
   - 현재: 라운드별 Opinion, Rebuttal 저장
   - 개선: 토론 히스토리 분석 및 학습
   - 개선: 유사한 상황에서의 과거 토론 결과 활용

#### 4.4.3 성능 개선

1. **토론 효율성 향상**
   - 현재: 모든 에이전트가 모든 라운드에 참여
   - 개선: 에이전트별 참여 전략 (높은 신뢰도 에이전트 우선)
   - 개선: 토론 조기 종료 조건 최적화

2. **예측 정확도 향상**
   - 현재: 단순 앙상블
   - 개선: 스태킹(Stacking) 메타 모델 도입
   - 개선: 에이전트별 예측 불확실성 고려한 통합

3. **실시간 처리**
   - 현재: 배치 처리 중심
   - 개선: 스트리밍 데이터 처리
   - 개선: 실시간 예측 업데이트

4. **해석 가능성**
   - 현재: 각 에이전트의 Opinion만 제공
   - 개선: 토론 과정 시각화
   - 개선: 최종 예측에 기여한 에이전트 및 근거 명시

### 4.5 공통 개선 방향

#### 4.5.1 프로세스 개선

1. **CI/CD 파이프라인 구축**
   - 현재: 수동 테스트 및 배포
   - 개선: 자동화된 테스트 스위트
   - 개선: 모델 재학습 파이프라인 자동화
   - 개선: 성능 모니터링 및 알림

2. **문서화 강화**
   - 현재: 기본적인 코드 주석
   - 개선: API 문서 자동 생성
   - 개선: 아키텍처 다이어그램
   - 개선: 사용자 가이드 및 튜토리얼

3. **테스트 커버리지**
   - 현재: 수동 테스트 중심
   - 개선: 단위 테스트 작성
   - 개선: 통합 테스트 작성
   - 개선: 성능 벤치마크 테스트

#### 4.5.2 데이터 개선

1. **데이터 품질 관리 시스템**
   - 현재: 기본적인 데이터 검증
   - 개선: 데이터 품질 메트릭 정의 및 모니터링
   - 개선: 데이터 드리프트 감지
   - 개선: 자동 데이터 정제 파이프라인

2. **데이터 버전 관리**
   - 현재: 파일 기반 저장
   - 개선: DVC (Data Version Control) 도입
   - 개선: 데이터 라인지 추적

#### 4.5.3 성능 개선

1. **인프라 최적화**
   - 현재: 단일 머신 실행
   - 개선: 분산 학습 지원
   - 개선: GPU 활용 최적화
   - 개선: 모델 서빙 최적화 (ONNX 변환 등)

2. **모니터링 및 관찰 가능성**
   - 현재: 기본적인 로깅
   - 개선: 구조화된 로깅 (JSON 형식)
   - 개선: 메트릭 수집 시스템 (Prometheus 등)
   - 개선: 대시보드 구축 (Grafana 등)

---

## 5. 결론 및 권장사항

### 5.1 즉시 조치 필요 사항

1. **모델 재학습**
   - 모든 에이전트의 모델 파일이 삭제되었으므로 즉시 재학습 필요
   - 특히 MacroAgent는 TensorFlow → PyTorch 전환으로 인해 필수

2. **환경변수 설정**
   - `CAPSTONE_OPENAI_API`: 필수
   - `EODHD_API_KEY`: SentimentalAgent 사용 시 필수

3. **의존성 확인**
   - `requirements.txt` 변경사항 확인 및 패키지 재설치
   - TensorFlow 제거, PyTorch 버전 확인

### 5.2 단기 개선 사항 (1-2주)

1. **테스트 및 검증**
   - 각 에이전트별 기능 테스트
   - 모델 재학습 후 성능 검증
   - 통합 테스트 (DebateAgent 전체 플로우)

2. **문서화 보완**
   - 변경사항 반영된 사용자 가이드 작성
   - API 문서 업데이트

### 5.3 중장기 개선 사항 (1-3개월)

1. **성능 최적화**
   - 하이퍼파라미터 튜닝
   - 모델 아키텍처 개선
   - 앙상블 방법 도입

2. **인프라 개선**
   - CI/CD 파이프라인 구축
   - 모니터링 시스템 구축
   - 자동화된 재학습 파이프라인

---

**리포트 작성 완료**

