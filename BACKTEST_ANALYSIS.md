# 백테스팅 스크립트 상세 분석

## 📋 목차
1. [전체 아키텍처](#전체-아키텍처)
2. [실행 플로우](#실행-플로우)
3. [핵심 메서드 상세 분석](#핵심-메서드-상세-분석)
4. [데이터 흐름](#데이터-흐름)
5. [분석 알고리즘](#분석-알고리즘)
6. [시각화 프로세스](#시각화-프로세스)

---

## 전체 아키텍처

### 클래스 구조
```
RollingBacktester
├── __init__()          # 초기화 및 설정
├── prepare_data()      # 데이터 준비
├── run_loop()          # 백테스팅 메인 루프
├── _collect_result()   # 결과 수집
├── save_results()      # CSV 저장
└── analyze()           # 결과 분석 및 시각화
```

### 주요 의존성
- **DebateAgent**: Multi-Agent Debate 시스템 (예측 생성)
- **core.metrics**: 성능 지표 계산 함수들
- **core.data_set**: 데이터셋 빌드 유틸리티

---

## 실행 플로우

### 1. 초기화 단계 (`__init__`)

```python
RollingBacktester(ticker, start_date, end_date, output_dir, auto_analyze)
```

**동작 과정:**
1. **티커 정규화**: `ticker.upper()`로 대문자 변환
2. **디렉토리 생성**: `data/backtests` 디렉토리 자동 생성
3. **DebateAgent 초기화**: 
   - TechnicalAgent, MacroAgent, SentimentalAgent 생성
   - 각 에이전트는 독립적인 모델과 데이터 처리 로직 보유
4. **Test Mode 활성화**: 
   - 모든 에이전트에 `set_test_mode(True)` 호출
   - 백테스팅 시 미래 데이터 누출 방지
   - `simulation_date` 이전 데이터만 사용하도록 필터링

**핵심 변수:**
- `self.results`: 일별 예측 결과 저장 리스트
- `self.csv_path`: 최종 CSV 저장 경로
- `self.auto_analyze`: 분석 자동 실행 여부

---

### 2. 데이터 준비 단계 (`prepare_data`)

**목적**: 백테스팅 기간 + Lookback 기간 전체 데이터를 한 번에 수집

**계산 로직:**
```python
max_period_days = 365 * 3  # 최대 Lookback: 3년
test_days = (end_date - start_date).days  # 백테스팅 기간
total_days = max_period_days + test_days + 30  # 여유분 30일
period = f"{int(total_days/365) + 2}y"  # 넉넉하게 설정
```

**데이터 수집:**
- `build_dataset()` 호출로 전체 기간 데이터 생성
- 각 Agent별 데이터셋 자동 생성:
  - TechnicalAgent: 기술적 지표 데이터
  - MacroAgent: 거시경제 데이터
  - SentimentalAgent: 감성 분석 데이터

**왜 한 번에 수집?**
- 매일 데이터를 새로 수집하면 비효율적
- Lookback 기간이 필요하므로 미리 준비
- 네트워크/API 호출 최소화

---

### 3. 백테스팅 루프 (`run_loop`)

#### 3.1 날짜 순회
```python
current_dt = start_date
while current_dt <= end_date:
    # 주말 제외 (weekday >= 5)
    # 각 거래일마다 시뮬레이션 수행
    current_dt += timedelta(days=1)
```

#### 3.2 Time Travel 설정
```python
for name, ag in self.agent.agents.items():
    if hasattr(ag, "set_simulation_date"):
        ag.set_simulation_date(sim_date)
```

**핵심 개념:**
- **Time Travel**: 특정 날짜로 "시간 여행"
- 각 에이전트는 `simulation_date` 이전 데이터만 사용
- `BaseAgent._apply_date_filter()` 메서드로 데이터 필터링
- 미래 정보 누출 방지 (Look-ahead bias 제거)

**데이터 필터링 로직:**
```python
def _apply_date_filter(self, X, dates):
    if not self.test_mode or not self.simulation_date:
        return X  # 필터링 없음
    
    sim_date = datetime.strptime(self.simulation_date, "%Y-%m-%d")
    valid_indices = []
    for i, date_seq in enumerate(dates):
        last_date = datetime.strptime(date_seq[-1], "%Y-%m-%d")
        if last_date <= sim_date:  # 시뮬레이션 날짜 이전만
            valid_indices.append(i)
    return X[valid_indices]
```

#### 3.3 Debate 실행
```python
result = self.agent.run()
```

**DebateAgent.run() 내부 프로세스:**

1. **Round 0: 초기 Opinion 수집**
   ```
   TechnicalAgent.get_opinion()  → 예측값 + 근거
   MacroAgent.get_opinion()      → 예측값 + 근거
   SentimentalAgent.get_opinion() → 예측값 + 근거
   ```

2. **Round 1~N: Rebuttal & Revise**
   ```
   Round 1:
   - get_rebuttal(): 각 에이전트가 다른 에이전트의 의견에 반박/지지
   - get_revise(): 반박 내용을 바탕으로 예측 수정
   
   Round 2, 3... (기본 3라운드)
   ```

3. **최종 Ensemble**
   ```
   get_ensemble():
   - 각 에이전트의 최종 예측값 수집
   - 가중 평균 또는 앙상블 모델로 통합
   - ensemble_next_close 계산
   ```

**반환 데이터 구조:**
```python
{
    "last_price": float,              # 현재 종가
    "ensemble_next_close": float,      # 앙상블 예측 종가
    "mean_next_close": float,          # 단순 평균 예측
    "agents": {
        "TechnicalAgent_next_close": float,
        "MacroAgent_next_close": float,
        "SentimentalAgent_next_close": float
    }
}
```

#### 3.4 결과 수집 (`_collect_result`)

**데이터 변환:**
```python
row = {
    "Date": "2024-01-15",
    "Ticker": "TSLA",
    "Actual_Close": 250.50,           # 실제 종가
    "Ensemble_Pred": 252.30,          # 앙상블 예측
    "Mean_Pred": 251.80,              # 평균 예측
    "TechnicalAgent_Pred": 251.50,    # 각 에이전트별 예측
    "MacroAgent_Pred": 252.10,
    "SentimentalAgent_Pred": 251.80
}
self.results.append(row)
```

**중간 저장:**
- 매 거래일마다 `save_results()` 호출
- 데이터 유실 방지 (장애 시 복구 가능)
- CSV 파일에 append 방식으로 저장

---

### 4. 분석 단계 (`analyze`)

#### 4.1 데이터 로드 및 전처리

```python
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df_valid = df.dropna(subset=['Actual_Close', 'Ensemble_Pred'])
```

**전처리:**
- 날짜 형식 변환
- 날짜순 정렬
- NaN 값 제거 (예측 실패한 날짜 제외)

#### 4.2 기본 성능 지표 계산

**MAE (Mean Absolute Error)**
```python
mae = np.mean(np.abs(y_true - y_pred))
```
- 평균 절대 오차
- 단위: 달러 ($)
- 낮을수록 좋음

**RMSE (Root Mean Squared Error)**
```python
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
```
- 평균 제곱근 오차
- 큰 오차에 더 큰 페널티
- 단위: 달러 ($)

**MAPE (Mean Absolute Percentage Error)**
```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```
- 평균 절대 백분율 오차
- 단위: %
- 가격 스케일과 무관한 상대적 오차

#### 4.3 방향 정확도 계산

**알고리즘:**
```python
prev_close = df['Actual_Close'].shift(1)  # 전일 종가
actual_diff = y_true - prev_close         # 실제 변동
pred_diff = y_pred - prev_close           # 예측 변동

actual_sign = np.sign(actual_diff)       # 실제 방향 (+1, 0, -1)
pred_sign = np.sign(pred_diff)            # 예측 방향

match = (actual_sign == pred_sign)       # 방향 일치 여부
direction_accuracy = np.mean(match) * 100  # 정확도 (%)
```

**의미:**
- 상승/하락 방향을 맞춘 비율
- 가격 예측보다 방향 예측이 더 중요할 수 있음
- 50% 이상이면 랜덤보다 우수

#### 4.4 수익률 분석 (`calculate_profitability`)

**Buy & Hold 전략:**
```python
bh_shares = initial_capital / actual_prices[0]  # 첫날 매수
bh_final_value = bh_shares * actual_prices[-1]  # 마지막날 평가
bh_return = (bh_final_value / initial_capital - 1) * 100
```

**예측 기반 전략:**
```python
for i in range(1, len(actual_prices)):
    prev_close = actual_prices[i-1]      # 전일 종가
    curr_close = actual_prices[i]         # 오늘 종가
    pred_close = predicted_prices[i]      # 예측 종가
    
    if pred_close > prev_close:
        # 상승 예측 → 보유 → 수익률 적용
        daily_ret = curr_close / prev_close
        current_capital *= daily_ret
    else:
        # 하락 예측 → 현금 보유 → 수익률 1.0 (변화 없음)
        pass
```

**전략 로직:**
- **Long Only**: 매도 시 현금 보유 (공매도 없음)
- **매매 시점**: T-1 시점에 T 시점 종가 예측을 보고 결정
- **거래 가격**: 전일 종가 기준 (실제로는 시가에 거래하지만 근사)

**수익률 비교:**
- Strategy Return vs Buy & Hold Return
- Strategy가 더 높으면 예측이 유용함을 의미

---

### 5. 시각화 프로세스

#### 5.1 Price Chart (가격 예측 차트)

**구성 요소:**
- **Actual Close** (검은색 실선): 실제 종가
- **Ensemble Pred** (파란색 점선): 앙상블 예측
- **Agent별 예측** (점선, 반투명): 각 에이전트 개별 예측

**용도:**
- 예측값과 실제값의 추세 비교
- 에이전트별 예측 차이 확인
- 시계열 패턴 분석

#### 5.2 Cumulative Return Chart (누적 수익률 차트)

**계산 과정:**
```python
# 일일 수익률
df['Daily_Ret'] = df['Actual_Close'].pct_change()

# 전략 신호
df['Signal'] = np.where(df['Ensemble_Pred'] > df['Prev_Close'], 1, 0)

# 전략 일일 수익률
df['Strat_Daily_Ret'] = df['Signal'] * df['Daily_Ret']

# 누적 수익률
df['Cum_BH'] = (1 + df['Daily_Ret']).cumprod()      # Buy & Hold
df['Cum_Strat'] = (1 + df['Strat_Daily_Ret']).cumprod()  # Strategy
```

**시각화:**
- X축: 날짜
- Y축: 누적 수익률 (1.0 = 0%)
- 두 선의 차이가 전략의 성과

**해석:**
- Strategy 선이 Buy & Hold 위에 있으면 → 전략 우수
- 차이가 크면 → 예측이 유용함
- 교차 지점 → 전략이 Buy & Hold를 추월/추월당함

#### 5.3 Error Histogram (오차 분포 히스토그램)

**계산:**
```python
errors = (pred - actual) / actual * 100  # 백분율 오차
plt.hist(errors, bins=30)
```

**분석:**
- 정규분포에 가까우면 → 예측이 일관적
- 왜도(skewness) 확인:
  - 양의 왜도: 과소 예측 경향
  - 음의 왜도: 과대 예측 경향
- 꼬리(tail) 확인: 극단적 오차 발생 빈도

---

## 데이터 흐름 다이어그램

```
[Command Line]
    ↓
[RollingBacktester.__init__]
    ├─→ DebateAgent 초기화
    │   ├─→ TechnicalAgent
    │   ├─→ MacroAgent
    │   └─→ SentimentalAgent
    └─→ Test Mode 활성화
    ↓
[prepare_data()]
    └─→ build_dataset() → 전체 기간 데이터 수집
    ↓
[run_loop()]
    ├─→ 날짜 순회 (start_date ~ end_date)
    │   ├─→ 주말 제외
    │   ├─→ set_simulation_date() → Time Travel
    │   ├─→ DebateAgent.run()
    │   │   ├─→ Round 0: Opinion 수집
    │   │   ├─→ Round 1~N: Rebuttal & Revise
    │   │   └─→ Ensemble 예측
    │   ├─→ _collect_result() → 결과 저장
    │   └─→ save_results() → CSV 저장
    └─→ analyze() (auto_analyze=True인 경우)
        ├─→ CSV 로드
        ├─→ 지표 계산
        │   ├─→ MAE, RMSE, MAPE
        │   ├─→ Direction Accuracy
        │   └─→ Profitability
        └─→ 시각화
            ├─→ Price Chart
            ├─→ Return Chart
            └─→ Error Histogram
```

---

## 핵심 알고리즘 상세

### 1. Time Travel 메커니즘

**문제:** 백테스팅 시 미래 정보 누출 방지

**해결:**
1. `test_mode = True` 설정
2. `simulation_date` 설정 (예: "2024-01-15")
3. 데이터 로드 시 `simulation_date` 이전만 사용

**예시:**
```
시뮬레이션 날짜: 2024-01-15
사용 가능 데이터: 2024-01-15 이전 모든 데이터
사용 불가 데이터: 2024-01-16 이후 데이터
```

### 2. Ensemble 예측 생성

**방법 1: 가중 평균**
```python
weights = {
    "TechnicalAgent": 0.4,
    "MacroAgent": 0.3,
    "SentimentalAgent": 0.3
}
ensemble = sum(agent_pred * weight for agent_pred, weight in zip(predictions, weights))
```

**방법 2: 앙상블 모델**
- 메타 학습 모델 사용
- 각 에이전트 예측을 입력으로 받아 최종 예측 생성

### 3. 수익률 계산의 시간 정렬

**핵심:** 예측 시점과 거래 시점의 정렬

```
T-1 시점 (예: 2024-01-14 장 마감 후)
├─→ T 시점 종가 예측 (2024-01-15 종가 예측)
└─→ T 시점 시가에 거래 결정

T 시점 (2024-01-15)
├─→ 시가: 전일 종가 근사
├─→ 종가: 실제 거래 종료 가격
└─→ 수익률 = (종가 / 시가) - 1
```

**코드 구현:**
```python
# T-1 시점 예측값: predicted_prices[i]
# T 시점 실제 종가: actual_prices[i]
# T-1 시점 종가 (거래 가격): actual_prices[i-1]

if predicted_prices[i] > actual_prices[i-1]:
    # 상승 예측 → T 시점 보유
    return = actual_prices[i] / actual_prices[i-1]
else:
    # 하락 예측 → 현금 보유
    return = 1.0
```

---

## 성능 최적화 포인트

### 1. 데이터 중복 수집 방지
- `prepare_data()`에서 한 번에 전체 데이터 수집
- 매일 데이터를 새로 수집하지 않음

### 2. 중간 저장
- 매 거래일마다 CSV 저장
- 장애 시 복구 가능
- 진행 상황 모니터링 가능

### 3. 에러 처리
- 특정 날짜에서 에러 발생해도 다음 날짜로 진행
- 전체 백테스팅이 중단되지 않음

---

## 제한사항 및 개선 가능 영역

### 현재 제한사항

1. **거래 비용 미고려**
   - 수수료, 슬리피지 미반영
   - 실제 수익률보다 높게 평가될 수 있음

2. **단순 전략**
   - Long Only (공매도 없음)
   - 예측값 > 전일 종가면 보유, 아니면 현금
   - 더 복잡한 전략 가능 (포지션 크기 조절, 리스크 관리 등)

3. **주말 처리**
   - 단순히 제외 (weekday >= 5)
   - 공휴일 미고려

4. **데이터 품질**
   - NaN 값 제외만 수행
   - 이상치(outlier) 처리 없음

### 개선 가능 영역

1. **고급 지표 추가**
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate
   - Calmar Ratio

2. **전략 고도화**
   - 포지션 크기 조절
   - 리스크 관리 (손절, 익절)
   - 공매도 허용

3. **병렬 처리**
   - 여러 날짜 동시 처리
   - 멀티프로세싱 활용

4. **실시간 모니터링**
   - 진행률 표시 (Progress Bar)
   - 예상 완료 시간 계산

---

## 사용 예시

### 기본 사용
```bash
python3 scripts/rolling_backtest.py \
    --ticker TSLA \
    --start 2024-01-01 \
    --end 2024-12-31
```

### 분석 스킵
```bash
python3 scripts/rolling_backtest.py \
    --ticker TSLA \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --no-analyze
```

### Python 코드에서 사용
```python
from scripts.rolling_backtest import RollingBacktester

runner = RollingBacktester(
    ticker="TSLA",
    start_date="2024-01-01",
    end_date="2024-12-31",
    auto_analyze=True
)

runner.prepare_data()
runner.run_loop()  # 자동으로 analyze() 호출됨

# 또는 수동 분석
runner.analyze(csv_path="custom_path.csv")
```

---

## 결론

이 백테스팅 시스템은:
- ✅ **Time Travel 메커니즘**으로 미래 정보 누출 방지
- ✅ **Multi-Agent Debate**로 다양한 관점의 예측 통합
- ✅ **자동 분석 및 시각화**로 결과 해석 용이
- ✅ **견고한 에러 처리**로 장애 시에도 진행 가능

실전 투자 전략 검증에 활용할 수 있는 완전한 백테스팅 시스템입니다.


