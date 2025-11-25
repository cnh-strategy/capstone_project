# Rolling Backtest 스크립트 문제점 분석

## 🔴 구조적 문제점

### 1. **하드코딩된 매직 넘버**
- **위치**: `prepare_data()` 메서드
- **문제**: 
  - `max_period_days = 365 * 3` (3년 고정)
  - `total_days = max_period_days + test_days + 30` (30일 여유분 고정)
  - `period=f"{int(total_days/365) + 2}y"` (계산 로직이 복잡하고 불명확)
- **영향**: 다른 lookback 기간이 필요한 경우 수정 어려움
- **개선**: 설정 파일이나 파라미터로 분리

### 2. **에러 처리 부족**
- **위치**: `run_loop()` 메서드의 try-except 블록
- **문제**:
  - 너무 넓은 예외 처리 (`except Exception as e`)
  - 에러 발생 시 단순히 print만 하고 계속 진행
  - 에러 타입별 처리 없음 (네트워크 에러, 데이터 에러, 모델 에러 등)
  - 에러 발생한 날짜가 결과에 기록되지 않음
- **영향**: 디버깅 어려움, 부분 실패 추적 불가
- **개선**: 
  - 구체적인 예외 타입별 처리
  - 에러 로깅 및 재시도 로직
  - 실패한 날짜 기록

### 3. **상태 관리 문제**
- **위치**: `run_loop()` 메서드
- **문제**:
  - `DebateAgent`가 루프 간에 상태를 유지함 (`self.opinions`, `self.rebuttals` 등)
  - 매일 `agent.run()`을 호출하지만 이전 라운드의 상태가 남아있을 수 있음
  - `_data_built` 플래그가 한 번만 True로 설정되어 재사용됨
- **영향**: 날짜 간 데이터 오염 가능성, 예측 정확도 저하
- **개선**: 
  - 각 날짜마다 DebateAgent를 새로 생성하거나
  - 상태를 명시적으로 리셋하는 메서드 추가

### 4. **주말/공휴일 처리 부족**
- **위치**: `run_loop()` 메서드
- **문제**:
  - 단순히 `weekday() >= 5`로 주말만 체크
  - 공휴일 미고려 (미국 주식시장 기준)
  - 실제 거래일(Trading Day) 확인 없음
- **영향**: 불필요한 계산, 실제 거래일과 불일치
- **개선**: 
  - `pandas_market_calendars` 같은 라이브러리 사용
  - 또는 yfinance로 실제 거래일 확인

### 5. **데이터 일관성 문제**
- **위치**: `prepare_data()`와 `run_loop()` 사이
- **문제**:
  - `prepare_data()`에서 한 번만 데이터를 수집
  - 루프 중에 데이터가 업데이트되지 않음
  - 시뮬레이션 날짜가 변경되어도 데이터는 고정
- **영향**: 실제로는 각 날짜마다 새로운 데이터가 필요할 수 있음
- **개선**: 
  - 각 날짜마다 필요한 데이터만 동적으로 로드
  - 또는 데이터 캐싱 전략 명확화

### 6. **메모리 누수 가능성**
- **위치**: 전체 루프 구조
- **문제**:
  - 매일 `agent.run()` 호출 시 내부 상태 누적
  - 결과 리스트(`self.results`)가 계속 증가
  - 중간 저장은 하지만 메모리에서 제거하지 않음
- **영향**: 장기 백테스트 시 메모리 부족
- **개선**: 
  - 배치 단위로 결과 저장 후 메모리 정리
  - 또는 스트리밍 방식으로 CSV에 직접 쓰기

### 7. **입력 검증 부족**
- **위치**: `__init__()` 메서드
- **문제**:
  - 날짜 형식 검증 없음
  - `start_date > end_date` 체크 없음
  - ticker 유효성 검증 없음
- **영향**: 런타임 에러 발생 가능
- **개선**: 
  - 입력 검증 로직 추가
  - 명확한 에러 메시지

### 8. **결과 수집 로직의 취약성**
- **위치**: `_collect_result()` 메서드
- **문제**:
  - `result.get()`으로 안전하게 가져오지만, 키가 없을 때 None이 저장됨
  - `Actual_Close`가 None일 수 있음 (실제 가격을 가져오지 못한 경우)
  - 에이전트별 예측값 추출 로직이 문자열 치환에 의존 (`k.replace("_next_close", "")`)
- **영향**: 분석 단계에서 NaN 처리 필요, 데이터 품질 저하
- **개선**: 
  - 필수 필드 검증
  - 타입 체크

### 9. **분석 메서드의 중복 로직**
- **위치**: `analyze()` 메서드
- **문제**:
  - `calculate_profitability()` 호출 시 `dates` 리스트를 생성하는데, 이미 `df_valid['Date']`가 있음
  - `prev_close` 계산이 `fillna(method='bfill')` 사용 (deprecated)
- **영향**: pandas 버전 호환성 문제, 불필요한 변환
- **개선**: 
  - pandas 최신 API 사용
  - 중복 로직 제거

### 10. **파일 경로 하드코딩**
- **위치**: 전체
- **문제**:
  - `output_dir = "data/backtests"` 기본값이 하드코딩
  - 상대 경로 사용으로 인한 작업 디렉토리 의존성
- **영향**: 다른 디렉토리에서 실행 시 문제 발생 가능
- **개선**: 
  - 절대 경로 또는 프로젝트 루트 기준 경로 사용

---

## 🧪 테스트 문제점

### 1. **테스트 불가능한 구조**
- **문제**:
  - `DebateAgent` 직접 인스턴스화 (하드 의존성)
  - 파일 시스템 직접 사용 (`os.makedirs`, `pd.to_csv`)
  - 외부 API 호출 (yfinance, OpenAI) 포함
  - 날짜 기반 로직이 복잡하여 모킹 어려움
- **영향**: 단위 테스트 작성 불가능
- **개선**: 
  - 의존성 주입 (Dependency Injection)
  - 인터페이스 추상화
  - 파일 I/O 래퍼 클래스

### 2. **재현성 문제**
- **문제**:
  - 랜덤 시드 설정 없음
  - 날짜 순회 로직이 복잡하여 재현 어려움
  - LLM 호출로 인한 비결정적 결과
- **영향**: 동일 입력에 대해 다른 결과 가능
- **개선**: 
  - 시드 설정
  - 결정적 로직 분리
  - LLM 호출 결과 캐싱 (테스트 모드)

### 3. **검증 로직 부족**
- **문제**:
  - 결과 데이터 검증 없음 (예: 가격이 음수인지, 예측값이 합리적인 범위인지)
  - 날짜 순서 검증 없음
  - 데이터 완전성 검증 없음
- **영향**: 잘못된 결과가 저장될 수 있음
- **개선**: 
  - 결과 검증 로직 추가
  - 어서션(assertion) 추가

### 4. **모킹 어려움**
- **문제**:
  - `DebateAgent.run()`이 복잡한 내부 로직 수행
  - 여러 에이전트와 상호작용
  - 파일 시스템과 네트워크 호출 혼재
- **영향**: 통합 테스트 작성 어려움
- **개선**: 
  - 작은 단위로 분리
  - Mock 객체 사용 가능한 구조

### 5. **테스트 데이터 부재**
- **문제**:
  - 실제 데이터에 의존
  - 테스트용 샘플 데이터 생성 로직 없음
- **영향**: 테스트 실행 시 실제 API 호출 필요
- **개선**: 
  - 테스트용 데이터 생성 유틸리티
  - Fixture 데이터 제공

### 6. **에러 시나리오 테스트 불가**
- **문제**:
  - 네트워크 에러, 파일 시스템 에러 등 시뮬레이션 어려움
  - 부분 실패 시나리오 테스트 불가
- **영향**: 실제 운영 환경에서 예상치 못한 에러 발생
- **개선**: 
  - 에러 인젝션 메커니즘
  - 실패 시나리오 테스트

### 7. **성능 테스트 불가**
- **문제**:
  - 실행 시간 측정 로직 없음
  - 메모리 사용량 추적 없음
  - 병목 지점 식별 불가
- **영향**: 성능 최적화 어려움
- **개선**: 
  - 프로파일링 도구 통합
  - 성능 메트릭 수집

### 8. **통합 테스트 어려움**
- **문제**:
  - 전체 파이프라인이 하나의 클래스에 집중
  - 단계별 검증 포인트 없음
- **영향**: 부분 기능 검증 불가
- **개선**: 
  - 단계별 검증 로직
  - 중간 결과 저장 및 검증

---

## 📋 개선 우선순위

### 높음 (Critical)
1. ✅ 상태 관리 문제 (각 날짜마다 상태 리셋)
2. ✅ 에러 처리 개선 (구체적인 예외 처리)
3. ✅ 입력 검증 추가
4. ✅ 주말/공휴일 처리 개선

### 중간 (Important)
5. ✅ 하드코딩된 값 설정 파일로 분리
6. ✅ 메모리 누수 방지 (배치 저장)
7. ✅ 결과 검증 로직 추가
8. ✅ 의존성 주입으로 테스트 가능성 향상

### 낮음 (Nice to have)
9. ✅ 분석 메서드 리팩토링
10. ✅ 파일 경로 처리 개선
11. ✅ 성능 프로파일링 추가
12. ✅ 테스트 데이터 생성 유틸리티

---

## 💡 구체적인 개선 제안

### 1. 설정 파일 분리
```python
# config/backtest_config.py
BACKTEST_CONFIG = {
    "lookback_years": 3,
    "buffer_days": 30,
    "output_dir": "data/backtests",
    "max_retries": 3,
    "retry_delay": 5,
}
```

### 2. 상태 리셋 메서드
```python
def reset_agent_state(self):
    """각 날짜 시뮬레이션 전에 에이전트 상태 리셋"""
    for name, ag in self.agent.agents.items():
        if hasattr(ag, "reset_state"):
            ag.reset_state()
    # DebateAgent 상태도 리셋 필요
    self.agent.opinions.clear()
    self.agent.rebuttals.clear()
```

### 3. 거래일 확인 유틸리티
```python
def is_trading_day(date: datetime) -> bool:
    """실제 거래일인지 확인"""
    # pandas_market_calendars 사용 또는
    # yfinance로 해당 날짜 데이터 존재 여부 확인
    pass
```

### 4. 결과 검증
```python
def _validate_result(self, result: Dict) -> bool:
    """결과 데이터 검증"""
    if not result.get("last_price") or result["last_price"] <= 0:
        return False
    if not result.get("ensemble_next_close") or result["ensemble_next_close"] <= 0:
        return False
    # 예측값이 현재가의 ±50% 범위 내인지 확인
    price = result["last_price"]
    pred = result["ensemble_next_close"]
    if pred < price * 0.5 or pred > price * 1.5:
        return False
    return True
```

### 5. 의존성 주입
```python
class RollingBacktester:
    def __init__(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        agent_factory: Callable = None,  # 테스트용 모킹 가능
        data_loader: Callable = None,    # 테스트용 데이터 로더
        file_writer: Callable = None,     # 테스트용 파일 라이터
    ):
        self.agent = agent_factory(ticker) if agent_factory else DebateAgent(ticker)
        # ...
```

