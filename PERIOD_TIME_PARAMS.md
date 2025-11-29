# Period/Time 관련 파라미터 정리

## 현재 상황: 파라미터가 너무 많고 혼란스러움

### 1. 기본 Period 파라미터 (각 에이전트별)

```python
# config/agents.py
"TechnicalAgent": {
    "period": "2y",              # 기본 데이터 수집 기간
    "period_searcher": "5y",     # searcher에서 사용 (왜 다름?)
    "period_pretrain": "2y",     # pretrain에서 사용 (period와 동일)
}

"MacroAgent": {
    "period": "2y",              # 기본 데이터 수집 기간 (사용 안 함?)
    "normal_years": 2,           # 일반 모드 연수
    "backtest_years": 5,         # 백테스팅 모드 연수
    "searcher_buffer_days": 100, # searcher에서 window * 2 + buffer_days
}

"SentimentalAgent": {
    "period": "2y",              # 기본 데이터 수집 기간
    "run_dataset_days": 365,     # run_dataset에서 사용하는 일수 (period와 다름!)
}
```

### 2. 백테스터에서 사용하는 날짜 파라미터

```python
# core/backtester.py
self.start_date      # 테스트 시작일
self.end_date        # 테스트 종료일
self.data_start_date # 데이터 수집 시작일 (start_date - 2년)
days                 # 테스트 기간 (일수)
```

### 3. Window 관련 파라미터

```python
window_size          # 각 에이전트의 lookback window 크기
recent_days          # MacroAgent에서 최근 며칠치 데이터 사용 (14일)
```

## 문제점

1. **중복/불일치**: 
   - `period`와 `period_searcher`, `period_pretrain`이 다름
   - `period`와 `normal_years`, `backtest_years`가 다름
   - `period`와 `run_dataset_days`가 다름

2. **명확하지 않은 용도**:
   - `period`: 어디서 사용되는지 불명확
   - `period_searcher`: 왜 5년인지 불명확
   - `searcher_buffer_days`: 왜 필요한지 불명확

3. **일관성 부족**:
   - TechnicalAgent: period, period_searcher, period_pretrain
   - MacroAgent: normal_years, backtest_years, searcher_buffer_days
   - SentimentalAgent: period, run_dataset_days

## 제안: 통합 및 정리

### 옵션 1: 단순화 (권장)

```python
# 공통 파라미터
common_params = {
    "data_collection_years": 2,      # 일반 모드 데이터 수집 기간 (년)
    "backtest_data_years": 2,         # 백테스팅 모드 데이터 수집 기간 (년)
}

# 각 에이전트별
"TechnicalAgent": {
    "window_size": 55,
    # period 제거, common_params 사용
}

"MacroAgent": {
    "window_size": 40,
    "searcher_buffer_days": 100,     # window * 2 + buffer_days (필요시만)
    # normal_years, backtest_years 제거, common_params 사용
}

"SentimentalAgent": {
    "window_size": 40,
    # run_dataset_days 제거, common_params 사용
}
```

### 옵션 2: 명확한 네이밍

```python
# 데이터 수집 기간
"data_collection": {
    "normal_mode_years": 2,          # 일반 모드
    "backtest_mode_years": 2,        # 백테스팅 모드
    "searcher_years": 5,             # searcher 전용 (필요시만)
}

# 시계열 윈도우
"time_series": {
    "window_size": 55,               # lookback window
    "recent_days": 14,               # 최근 며칠치 (LLM context용)
}
```

## 현재 파라미터 사용 현황

### TechnicalAgent
- `period`: pretrain에서 사용 (기본값)
- `period_searcher`: searcher에서 사용 (5년)
- `period_pretrain`: pretrain에서 사용 (2년, period와 동일)

### MacroAgent
- `period`: 사용 안 함 (무시됨)
- `normal_years`: pretrain 일반 모드에서 사용 (2년)
- `backtest_years`: pretrain 백테스팅 모드에서 사용 (5년)
- `searcher_buffer_days`: searcher에서 사용 (100일)

### SentimentalAgent
- `period`: run_dataset에서 fallback으로 사용 (2년)
- `run_dataset_days`: run_dataset에서 우선 사용 (365일)

## 권장 사항

1. **공통 파라미터로 통합**: `common_params`에 데이터 수집 기간 통합
2. **불필요한 파라미터 제거**: `period_searcher`, `period_pretrain` 등 중복 제거
3. **명확한 네이밍**: `normal_years` → `data_collection_years_normal`
4. **일관성 유지**: 모든 에이전트가 동일한 방식으로 데이터 수집 기간 설정

