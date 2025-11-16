# 뉴스 데이터 활용 구조 분석

## 현재 구조에서 뉴스 데이터의 활용 방식

### 1. 데이터 수집 및 처리 플로우

```
EODHD API
  ↓
load_or_fetch_news() (finbert_utils.py)
  ↓
뉴스 캐시 저장 (data/raw/news/{TICKER.US}_{start}_{end}.json)
  ↓
build_finbert_news_features() (sentimental_agent.py)
  ↓
FinBERT 감성 분석
  ↓
일별 피처 집계 (compute_finbert_features)
  ↓
감성 피처 딕셔너리 생성
```

### 2. 뉴스 데이터가 사용되는 곳

#### ✅ **사용됨: LLM 컨텍스트 생성**

**위치**: `SentimentalAgent.build_ctx()`

```python
# 1. 뉴스 감성 피처 생성
news_feats = build_finbert_news_features(
    self.ticker, asof_date_kst, base_dir=os.path.join("data", "raw", "news")
)

# 2. feature_importance에 포함
feature_importance = {
    "sentiment_score": news_feats["sentiment_summary"]["mean_7d"],
    "sentiment_summary": news_feats["sentiment_summary"],
    "sentiment_volatility": {"vol_7d": news_feats["sentiment_volatility"].get("vol_7d", 0.0)},
    "trend_7d": news_feats["trend_7d"],
    "news_count": news_feats["news_count"],
    "has_news": news_feats.get("has_news", False),
    ...
}

# 3. LLM 컨텍스트에 포함
ctx = {
    "agent_id": self.agent_id,
    "ticker": self.ticker,
    "snapshot": snapshot,
    "prediction": {...},
    "feature_importance": feature_importance,  # ← 뉴스 데이터 포함
}
```

**활용 시점**:
- `reviewer_draft()`: Opinion의 `reason` 생성 시
- `reviewer_rebuttal()`: Rebuttal 생성 시
- `reviewer_revise()`: Revision 생성 시

**예시** (`_build_messages_opinion`):
```python
ctx = self.build_ctx()
# ctx["feature_importance"]["sentiment_summary"] 등을 LLM에 전달
# LLM이 이를 바탕으로 reason 생성
```

#### ❌ **사용되지 않음: 모델 학습 및 예측**

**위치**: `core/data_set.py`의 `build_dataset()`

```python
# 데이터셋 생성 시 사용되는 피처
feature_cols = [
    "returns", "sma_5", "sma_20", "rsi", "volume_z",
    "USD_KRW", "NASDAQ", "VIX",
    "sentiment_mean", "sentiment_vol",  # ← 단순 rolling mean/std
    "Open", "High", "Low", "Close", "Volume",
]

# sentiment_mean, sentiment_vol은 returns의 rolling 통계일 뿐
# 실제 FinBERT 뉴스 감성 분석 결과는 사용되지 않음
df["sentiment_mean"] = df["returns"].rolling(3).mean().fillna(0)
df["sentiment_vol"] = df["returns"].rolling(3).std().fillna(0)
```

**LSTM 모델 입력**:
- `input_dim`: 8 (config에서 정의)
- 실제 피처: `data_cols`에 정의된 8개 피처만 사용
- **FinBERT 뉴스 감성 피처는 포함되지 않음**

**예측 시** (`_predict_next_close`):
- 가격 데이터만 사용
- 뉴스 데이터는 사용되지 않음

### 3. 뉴스 데이터의 역할

현재 구조에서 뉴스 데이터는:

1. **모델 예측에는 직접 사용되지 않음**
   - LSTM 모델의 입력 피처에 포함되지 않음
   - 예측값 계산에 영향을 주지 않음

2. **LLM의 컨텍스트로만 사용됨**
   - Opinion의 `reason` 생성 시 참고
   - Rebuttal 생성 시 참고
   - Revision 생성 시 참고
   - **해석 가능성(Interpretability) 향상**

### 4. 뉴스 감성 피처 구조

```python
{
    "sentiment_summary": {
        "mean_7d": float,      # 7일 평균 감성 점수
        "mean_30d": float,     # 30일 평균 감성 점수
        "pos_ratio_7d": float, # 7일 양성 뉴스 비율
        "neg_ratio_7d": float, # 7일 음성 뉴스 비율
    },
    "sentiment_volatility": {
        "vol_7d": float,       # 7일 감성 변동성
    },
    "news_count": {
        "count_1d": int,       # 1일 뉴스 개수
        "count_7d": int,       # 7일 뉴스 개수
    },
    "trend_7d": float,         # 7일 감성 추세 (선형 회귀 기울기)
    "has_news": bool,          # 뉴스 데이터 존재 여부
}
```

### 5. 개선 가능한 점

현재 구조의 한계:
- 뉴스 데이터가 모델 학습/예측에 직접 사용되지 않음
- `sentiment_mean`, `sentiment_vol`은 단순 통계값일 뿐

개선 방안:
1. **모델 입력에 뉴스 피처 추가**
   - `data_cols`에 FinBERT 감성 피처 추가
   - 시계열 데이터로 변환하여 LSTM 입력에 포함

2. **멀티모달 접근**
   - 가격 데이터 + 뉴스 감성 데이터를 결합
   - Attention 메커니즘으로 가격과 뉴스의 상관관계 학습

3. **실시간 뉴스 반영**
   - 예측 시점의 최신 뉴스 데이터를 즉시 반영
   - 캐시 없이도 실시간 수집 가능하도록 개선

