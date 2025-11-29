# 백테스팅 데이터 흐름 분석: test_date = 10일

## 전체 구조 개요

### 초기 설정 (days=10일 경우)

```
end_date = 오늘 (예: 2025-11-29)
start_date = end_date - 10일 = 2025-11-19 (테스트 시작일)
data_start_date = start_date - 730일 = 2023-11-19 (전체 데이터 시작일)

전체 데이터 범위: 2023-11-19 ~ 2025-11-29 (약 2년 + 10일)
테스트 기간: 2025-11-19 ~ 2025-11-29 (10일)
```

### 1단계: 전체 데이터 1회 생성

**생성되는 데이터:**
- **TechnicalAgent 데이터**: `data_start_date` ~ `end_date` (약 2년 + 10일)
  - Config: period="3y" (최대 3년치)
  - Window size: 55일
  - 예상 샘플 수: 약 500~600개 (거래일 기준)

- **MacroAgent 데이터**: `searcher()` 호출 시 `datetime.today()` 기준으로 수집
  - Window size: 40일
  - 예상 샘플 수: 약 500~600개

- **SentimentalAgent 데이터**: `run_dataset(days_needed)` 호출
  - days_needed = (end_date - data_start_date).days + 365 = 약 1100일
  - Window size: 14일 (기본값)
  - 예상 샘플 수: 약 1000~1100개

---

## 각 시점별 상세 분석

### 시점 1: Day 1 (2025-11-19)

**슬라이싱:**
- TechnicalAgent: `tech_X_all`에서 `date <= 2025-11-19` 필터링
  - 사용 가능 데이터: 2023-11-19 ~ 2025-11-19 (약 2년)
  - 샘플 수: 약 500개

- MacroAgent: `macro_full_df_all`에서 `Date <= 2025-11-19` 필터링
  - 사용 가능 데이터: 과거 ~ 2025-11-19
  - 샘플 수: 약 500개

- SentimentalAgent: `senti_raw_all`에서 `date <= 2025-11-19` 필터링
  - 사용 가능 데이터: 과거 ~ 2025-11-19
  - 샘플 수: 약 1000개

**Pretrain:**
- TechnicalAgent.pretrain():
  - `simulation_date = "2025-11-19"` 확인
  - `load_dataset_tech()` → 날짜 필터링 → 2025-11-19 이전 데이터만 사용
  - 학습 데이터: 약 500개 중 80% = 400개
  - 검증 데이터: 약 100개
  - 모델 학습 후 저장

- MacroAgent.pretrain():
  - `simulation_date = "2025-11-19"` 확인
  - `end_date = 2025-11-19` 사용
  - `start_date = 2025-11-19 - 1825일 = 2020-11-19`
  - 학습 데이터: 2020-11-19 ~ 2025-11-19 (5년치)
  - 시퀀스 생성 후 학습

- SentimentalAgent.pretrain():
  - `simulation_date = "2025-11-19"` 확인
  - `build_pretrain_dataset()` → `super().pretrain()`
  - 학습 데이터: 2025-11-19 이전 데이터만 사용

**Predict:**
- TechnicalAgent: 2025-11-19 날짜의 윈도우 데이터로 예측
  - 입력: `tech_X_filtered[2025-11-19]` (55일 윈도우)
  - 출력: `pred_tech`, `conf_tech`, `unc_tech`

- MacroAgent: 2025-11-19 날짜의 윈도우 데이터로 예측
  - 입력: `macro_full_df`의 마지막 40일
  - 출력: `pred_macro`, `conf_macro`, `unc_macro`

- SentimentalAgent: 2025-11-19 날짜의 윈도우 데이터로 예측
  - 입력: `senti_raw`의 마지막 14일
  - 출력: `pred_senti`, `conf_senti`, `unc_senti`

**결과물:**
```python
{
    "Date": 2025-11-19,
    "Last_Close": 150.0,  # 2025-11-19 종가
    "Next_Close": 152.0,  # 2025-11-20 실제 종가 (Target)
    "Tech_Pred": 151.5, "Tech_Conf": 0.8, "Tech_Unc": 0.05,
    "Macro_Pred": 151.2, "Macro_Conf": 0.75, "Macro_Unc": 0.06,
    "Senti_Pred": 151.8, "Senti_Conf": 0.7, "Senti_Unc": 0.07
}
```

---

### 시점 2: Day 2 (2025-11-20)

**슬라이싱:**
- TechnicalAgent: `date <= 2025-11-20` 필터링
  - 사용 가능 데이터: 2023-11-19 ~ 2025-11-20
  - 샘플 수: 약 501개 (Day 1보다 1개 증가)

- MacroAgent: `Date <= 2025-11-20` 필터링
  - 사용 가능 데이터: 과거 ~ 2025-11-20
  - 샘플 수: 약 501개

- SentimentalAgent: `date <= 2025-11-20` 필터링
  - 사용 가능 데이터: 과거 ~ 2025-11-20
  - 샘플 수: 약 1001개

**Pretrain:**
- 각 에이전트: Day 1과 동일한 로직
  - **차이점**: 이제 2025-11-20 이전 데이터까지 포함
  - TechnicalAgent: 약 501개 → 학습 400개, 검증 101개
  - MacroAgent: 2020-11-19 ~ 2025-11-20 (5년 + 1일)
  - SentimentalAgent: 2025-11-20 이전 데이터

**Predict:**
- 각 에이전트: 2025-11-20 날짜의 윈도우 데이터로 예측

**결과물:**
```python
{
    "Date": 2025-11-20,
    "Last_Close": 152.0,
    "Next_Close": 153.5,  # 2025-11-21 실제 종가
    "Tech_Pred": 152.8, "Tech_Conf": 0.82, "Tech_Unc": 0.048,
    "Macro_Pred": 152.5, "Macro_Conf": 0.77, "Macro_Unc": 0.055,
    "Senti_Pred": 153.0, "Senti_Conf": 0.72, "Senti_Unc": 0.065
}
```

---

### 시점 3~9: Day 3 ~ Day 9

**패턴:**
- 각 날짜마다 동일한 로직 반복
- **데이터 증가**: 매일 1일씩 데이터가 추가됨
- **Pretrain**: 매일 실행 (최소 30개 데이터 요구사항 충족 시)
- **Predict**: 각 날짜의 윈도우 데이터로 예측

**데이터 증가 추이:**
```
Day 1: 약 500개 → Day 2: 약 501개 → ... → Day 9: 약 508개
```

---

### 시점 10: Day 10 (2025-11-29) - 마지막 날

**슬라이싱:**
- TechnicalAgent: `date <= 2025-11-29` 필터링
  - 사용 가능 데이터: 2023-11-19 ~ 2025-11-29
  - 샘플 수: 약 509개

- MacroAgent: `Date <= 2025-11-29` 필터링
  - 사용 가능 데이터: 과거 ~ 2025-11-29
  - 샘플 수: 약 509개

- SentimentalAgent: `date <= 2025-11-29` 필터링
  - 사용 가능 데이터: 과거 ~ 2025-11-29
  - 샘플 수: 약 1009개

**Pretrain:**
- 각 에이전트: 2025-11-29 이전 데이터로 최종 학습
  - TechnicalAgent: 약 509개 → 학습 407개, 검증 102개
  - MacroAgent: 2020-11-19 ~ 2025-11-29 (5년 + 10일)
  - SentimentalAgent: 2025-11-29 이전 데이터

**Predict:**
- 각 에이전트: 2025-11-29 날짜의 윈도우 데이터로 예측
- **주의**: `Next_Close`는 다음 거래일이 없으면 `np.nan`

**결과물:**
```python
{
    "Date": 2025-11-29,
    "Last_Close": 160.0,
    "Next_Close": np.nan,  # 다음 거래일 없음
    "Tech_Pred": 161.2, "Tech_Conf": 0.85, "Tech_Unc": 0.04,
    "Macro_Pred": 160.8, "Macro_Conf": 0.8, "Macro_Unc": 0.05,
    "Senti_Pred": 161.5, "Senti_Conf": 0.75, "Senti_Unc": 0.06
}
```

---

## 최종 결과물

### full_data (DataFrame)
```python
# 10일치 예측 결과
full_data = pd.DataFrame([
    {Day 1 결과},
    {Day 2 결과},
    ...
    {Day 10 결과}
])

# Shape: (10, 11)
# Columns: Date, Last_Close, Next_Close, 
#          Tech_Pred, Tech_Conf, Tech_Unc,
#          Macro_Pred, Macro_Conf, Macro_Unc,
#          Senti_Pred, Senti_Conf, Senti_Unc
```

### train_model() 입력 데이터

**Train 데이터:**
- `full_data`에서 `Date < start_date` 필터링
- **문제**: `start_date = 2025-11-19`이므로, `Date < 2025-11-19`인 데이터는 없음
- **결과**: Train 데이터 = 0행

**해결 방법:**
- `start_date` 이전에 학습 데이터가 필요하므로, `data_start_date` ~ `start_date` 구간의 데이터를 사용해야 함
- 또는 `full_data`의 앞부분을 Train으로 사용

### run_simulation() 입력 데이터

**Test 데이터:**
- `full_data`에서 `Date >= start_date` 필터링
- **결과**: 10일치 데이터 모두 사용
- LightGBM Meta Model로 앙상블 예측
- 매수/매도 신호 생성 및 포트폴리오 시뮬레이션

---

## 핵심 포인트

### 1. 데이터 누적 효과
- 각 날짜마다 1일씩 데이터가 추가됨
- Pretrain은 매일 증가하는 데이터로 재학습
- **장점**: 점진적으로 더 많은 데이터로 학습
- **단점**: 계산 비용 증가

### 2. Look-ahead Bias 방지
- ✅ 각 날짜마다 해당 날짜 이전 데이터만 사용
- ✅ `simulation_date` 설정으로 자동 필터링
- ✅ Pretrain도 해당 날짜 이전 데이터만 사용

### 3. Train/Test Split 문제
- ⚠️ 현재 구조에서는 Train 데이터가 0행이 될 수 있음
- `start_date` 이전 데이터가 `full_data`에 없음
- 해결: `data_start_date` ~ `start_date` 구간의 데이터도 `full_data`에 포함 필요

### 4. 결과물의 특징
- 각 날짜마다 3개 에이전트의 예측값 + 신뢰도 + 불확실성
- 총 10일치 예측 결과
- Meta Model 학습을 위한 앙상블 데이터
- 백테스트 시뮬레이션을 위한 입력 데이터

---

## 개선 제안

### Train 데이터 확보 방법
1. `prepare_data()`에서 `data_start_date` ~ `start_date` 구간도 예측 수행
2. 또는 `train_model()`에서 `full_data`의 앞부분을 Train으로 사용
3. 최소 Train 데이터 요구사항 확인 및 경고

### 성능 최적화
1. Pretrain을 매일 하지 않고 주기적으로만 수행 (예: 매 5일마다)
2. 데이터가 충분히 증가했을 때만 재학습
3. 이전 모델과 성능 비교 후 재학습 여부 결정

