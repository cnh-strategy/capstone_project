# Backtest 모듈

백테스팅 관련 코드, 데이터, 모델을 별도로 관리하는 폴더입니다.

## 폴더 구조

```
backtest/
├── scripts/
│   └── rolling_backtest.py    # 롤링 백테스트 실행 스크립트
├── data/
│   ├── raw/                    # Raw CSV 파일 (searcher에서 생성)
│   │   └── backtest_temp/      # 필터링된 임시 데이터셋
│   ├── processed/               # 전처리된 데이터셋
│   └── backtests/               # 백테스트 결과 CSV
│       └── analysis/            # 분석 차트 및 리포트
└── models/                      # 백테스트용 모델 파일
    └── scalers/                 # 스케일러 파일
```

## 사용법

### 기본 실행

```bash
cd /home/ubuntu/Projects/ml-ai/capstone
python backtest/scripts/rolling_backtest.py --ticker TSLA --predict-days 5
```

### 옵션

- `--ticker`: 분석할 티커 (필수)
- `--start`: 첫 번째 예측일 (YYYY-MM-DD). 지정하지 않으면 자동 계산
- `--predict-days`: 예측할 거래일 수 (기본: 5일)
- `--rounds`: 디베이트 라운드 수 (기본: 3회)
- `--no-analyze`: 자동 분석 스킵

### 예시

```bash
# TSLA에 대해 최근 5거래일 백테스트
python backtest/scripts/rolling_backtest.py --ticker TSLA --predict-days 5

# 특정 날짜부터 시작
python backtest/scripts/rolling_backtest.py --ticker TSLA --start 2024-01-01 --predict-days 10

# 분석 없이 실행만
python backtest/scripts/rolling_backtest.py --ticker TSLA --no-analyze
```

## 주요 특징

1. **독립된 디렉토리**: 백테스트 관련 모든 파일이 `backtest/` 폴더에 저장됩니다
2. **Searcher 사용**: `prepare_data()`에서 각 agent의 `searcher()` 메서드를 사용하여 데이터를 준비합니다 (agent 코드 수정 없이)
3. **자동 정리**: 각 날짜 처리 후 임시 데이터셋과 모델 파일이 자동으로 삭제됩니다
4. **자동 분석**: 백테스트 완료 후 성능 지표 계산 및 시각화가 자동으로 수행됩니다

## 데이터 준비

`prepare_data()` 메서드는 다음과 같이 동작합니다:

1. `DebateSystem`을 임시로 생성
2. 각 agent의 `data_dir`과 `model_dir`을 backtest 전용으로 설정
3. 각 agent의 `searcher(ticker, rebuild=True)` 호출
4. Raw CSV 파일이 `backtest/data/raw/`에 생성됨

이 방식으로 agent 코드를 수정하지 않고도 searcher를 사용할 수 있습니다.

## 출력 파일

- **결과 CSV**: `backtest/data/backtests/rolling_{TICKER}_{START}_{END}.csv`
- **분석 차트**: `backtest/data/backtests/analysis/`
  - `*_price.png`: 가격 예측 차트
  - `*_return.png`: 누적 수익률 차트
  - `*_error.png`: 오차 분포 히스토그램

