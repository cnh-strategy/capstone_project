# 파라미터 통합 제안 분석

## 제안 내용

1. **pretrain 기간과 searcher 기간은 동일한 파라미터로 수행**
2. **period, period_test 두 개만으로 공통 파라미터로 변환**

## 현재 상황 분석

### 현재 파라미터 분산 상태

#### TechnicalAgent
- `period: "2y"` - 기본값 (사용 안 함?)
- `period_searcher: "5y"` - searcher에서 사용
- `period_pretrain: "2y"` - pretrain에서 사용

#### MacroAgent
- `period: "2y"` - 사용 안 함
- `normal_years: 2` - pretrain 일반 모드
- `backtest_years: 5` - pretrain 백테스팅 모드
- `searcher_buffer_days: 50` - searcher에서 window + buffer_days

#### SentimentalAgent
- `period: "2y"` - fallback으로 사용
- `run_dataset_days: 365` - run_dataset/searcher에서 사용

## 제안된 구조

```python
# common_params에 추가
common_params = {
    "period": "2y",              # 일반 모드 데이터 수집 기간
    "period_test": "2y",         # 백테스팅 모드 데이터 수집 기간
}

# 각 에이전트에서 제거할 파라미터:
# - period_searcher, period_pretrain (TechnicalAgent)
# - normal_years, backtest_years (MacroAgent)
# - run_dataset_days (SentimentalAgent)
```

## 장점 분석

### ✅ 1. 단순화
- **현재**: 7개 파라미터 (period, period_searcher, period_pretrain, normal_years, backtest_years, run_dataset_days, searcher_buffer_days)
- **제안**: 2개 파라미터 (period, period_test)
- **감소율**: 약 71% 감소

### ✅ 2. 일관성
- 모든 에이전트가 동일한 방식으로 데이터 수집
- searcher와 pretrain이 동일한 기간 사용
- 일반 모드와 백테스팅 모드 구분 명확

### ✅ 3. 유지보수성
- 한 곳(common_params)에서 관리
- 변경 시 모든 에이전트에 일괄 적용
- 버그 발생 가능성 감소

### ✅ 4. 명확성
- `period`: 일반 모드
- `period_test`: 백테스팅 모드
- 직관적이고 이해하기 쉬움

## 잠재적 문제점 분석

### ⚠️ 1. TechnicalAgent의 period_searcher="5y" 이유
**현재 상황:**
- searcher에서 5년치 데이터 수집
- pretrain에서 2년치 데이터 수집

**가능한 이유:**
- searcher는 데이터셋을 한 번만 생성하므로 넉넉하게 수집?
- pretrain은 매번 재학습하므로 최소 기간만 사용?

**제안 적용 시:**
- searcher와 pretrain 모두 동일한 기간 사용
- **영향**: searcher에서 더 적은 데이터 수집 (5년 → 2년)
- **판단**: 문제 없음 (pretrain에서 2년으로 충분히 학습하고 있음)

### ⚠️ 2. MacroAgent의 searcher_buffer_days
**현재 상황:**
- searcher: `window + buffer_days` (40 + 50 = 90일)
- pretrain: `normal_years` 또는 `backtest_years` (2년 또는 5년)

**제안 적용 시:**
- searcher: `period` 또는 `period_test` 사용 (2년)
- pretrain: `period` 또는 `period_test` 사용 (2년)

**영향:**
- searcher: 90일 → 2년 (더 많은 데이터 수집)
- pretrain: 기존과 동일 (2년)

**판단:**
- searcher는 이미 최적화되어 있음 (90일)
- 하지만 pretrain과 통일하면 일관성 확보
- **해결책**: searcher는 `window + buffer_days` 유지, pretrain만 `period` 사용?

### ⚠️ 3. SentimentalAgent의 run_dataset_days=365
**현재 상황:**
- `run_dataset_days: 365` (1년)
- `period: "2y"` (2년, fallback)

**제안 적용 시:**
- `period: "2y"` 사용

**영향:**
- 365일 → 2년 (더 많은 데이터 수집)
- **판단**: 문제 없음 (더 많은 데이터가 학습에 도움)

## 제안 구조 상세

### Option A: 완전 통일 (권장)

```python
common_params = {
    "period": "2y",              # 일반 모드: searcher + pretrain 모두
    "period_test": "2y",         # 백테스팅 모드: searcher + pretrain 모두
}

# 모든 에이전트에서:
# - searcher: period 또는 period_test 사용
# - pretrain: period 또는 period_test 사용
```

**장점:**
- 완전한 일관성
- 가장 단순한 구조

**단점:**
- MacroAgent의 searcher 최적화(90일) 포기

### Option B: searcher 예외 허용

```python
common_params = {
    "period": "2y",              # 일반 모드: pretrain
    "period_test": "2y",         # 백테스팅 모드: pretrain
}

# MacroAgent만 searcher_buffer_days 유지
"MacroAgent": {
    "searcher_buffer_days": 50,  # searcher 전용 (최적화 유지)
}
```

**장점:**
- MacroAgent의 searcher 최적화 유지
- pretrain은 통일

**단점:**
- 완전한 통일은 아님

## 최종 판단

### ✅ 제안은 매우 합리적입니다

**이유:**
1. **단순화**: 파라미터 수 대폭 감소 (7개 → 2개)
2. **일관성**: 모든 에이전트가 동일한 방식
3. **유지보수성**: 한 곳에서 관리
4. **명확성**: period와 period_test만으로 이해 가능

### ⚠️ 고려사항

1. **MacroAgent searcher 최적화**
   - 현재: `window + buffer_days = 90일` (최적화됨)
   - 제안: `period = 2년` (더 많은 데이터)
   - **권장**: Option B (searcher_buffer_days 유지)

2. **TechnicalAgent period_searcher="5y"**
   - 현재: 5년 수집하지만 실제로는 마지막 window만 사용
   - 제안: 2년으로 통일
   - **판단**: 문제 없음 (pretrain에서 2년으로 충분)

3. **SentimentalAgent run_dataset_days=365**
   - 현재: 1년 수집
   - 제안: 2년으로 통일
   - **판단**: 더 많은 데이터가 학습에 도움

## 권장 구현 방안

### Option B (searcher 예외 허용) - 권장

```python
common_params = {
    # ... 기존 파라미터 ...
    "period": "2y",              # 일반 모드: pretrain + searcher (MacroAgent 제외)
    "period_test": "2y",         # 백테스팅 모드: pretrain + searcher (MacroAgent 제외)
}

# 각 에이전트에서 제거:
# TechnicalAgent: period_searcher, period_pretrain 제거
# MacroAgent: normal_years, backtest_years 제거 (searcher_buffer_days는 유지)
# SentimentalAgent: run_dataset_days 제거
```

**이유:**
- MacroAgent의 searcher 최적화 유지
- 나머지는 완전 통일
- 실용적이고 효율적

## 결론

**제안은 매우 합리적이며, 구현을 권장합니다.**

다만 MacroAgent의 searcher 최적화를 고려하여 `searcher_buffer_days`는 유지하는 것을 권장합니다.

