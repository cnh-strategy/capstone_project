# LLM에 전달되는 뉴스 데이터 분석

## 1. LLM 호출 시 뉴스 데이터 전달 방식

### 전달 경로

**Opinion 생성 시** (`reviewer_draft`):
```
reviewer_draft()
  ↓
_build_messages_opinion()  # ⚠️ build_ctx() 호출 안 함
  ↓
ctx에 뉴스 데이터 없음 (기본 정보만)
  ↓
LLM 프롬프트에 뉴스 데이터 미포함
  ↓
(또는 fallback에서 build_ctx() 호출)
  ↓
간단한 reason 생성 (감성 통계만 포함)
```

**Rebuttal 생성 시** (`reviewer_rebuttal`):
```
reviewer_rebuttal()
  ↓
_build_messages_rebuttal()
  ↓
build_ctx()  # ⭐ 뉴스 감성 피처 생성
  ↓
build_finbert_news_features()  # 뉴스 수집 + FinBERT 분석
  ↓
feature_importance 딕셔너리에 포함
  ↓
LLM 프롬프트에 감성 통계 포함
```

**Revision 생성 시** (`reviewer_revise`):
```
reviewer_revise()
  ↓
_build_messages_revision()
  ↓
build_ctx()  # ⭐ 뉴스 감성 피처 생성
  ↓
build_finbert_news_features()  # 뉴스 수집 + FinBERT 분석
  ↓
feature_importance 딕셔너리에 포함
  ↓
LLM 프롬프트에 감성 통계 포함
```

### LLM에 전달되는 데이터 구조

```python
ctx = {
    "agent_id": "SentimentalAgent",
    "ticker": "NVDA",
    "snapshot": {
        "asof_date": "2025-11-16",
        "last_price": 123.45,
        "currency": "USD",
        ...
    },
    "prediction": {
        "pred_close": 125.00,
        "pred_return": 0.0125,
        "uncertainty": {"std": 0.05, "ci95": 0.098},
        "confidence": 0.95,
        ...
    },
    "feature_importance": {  # ⭐ 뉴스 감성 데이터
        "sentiment_score": -0.1435,  # 7일 평균 감성 점수
        "sentiment_summary": {
            "mean_7d": -0.1435,      # 7일 평균 감성
            "mean_30d": -0.1435,     # 30일 평균 감성
            "pos_ratio_7d": 0.49,    # 양성 뉴스 비율
            "neg_ratio_7d": 0.51,    # 음성 뉴스 비율
        },
        "sentiment_volatility": {
            "vol_7d": 0.0,           # 7일 감성 변동성
        },
        "trend_7d": 0.0,             # 7일 감성 추세
        "news_count": {
            "count_1d": 100,         # 1일 뉴스 개수
            "count_7d": 100,         # 7일 뉴스 개수
        },
        "has_news": True,
        ...
    }
}
```

### LLM 프롬프트 예시

**Opinion 생성 시** (`_build_messages_opinion`):
```python
system_text = "너는 감성/뉴스 중심의 단기 주가 분석가다."

user_text = """
ctx(JSON):
{
  "agent_id": "SentimentalAgent",
  "ticker": "NVDA",
  "feature_importance": {
    "sentiment_summary": {
      "mean_7d": -0.1435,
      "pos_ratio_7d": 0.49,
      "neg_ratio_7d": 0.51
    },
    "news_count": {"count_7d": 100},
    ...
  },
  ...
}

위 ctx를 바탕으로 reason을 생성하라.
"""
```

**Rebuttal 생성 시** (`_build_messages_rebuttal`):
```python
user_text = """
티커: NVDA
상대 에이전트: TechnicalAgent
상대 의견: ...

우리 예측:
- next_close: 125.00
- 예상 변화율: 1.25%

감성 근거:
- mean7=-0.1435, mean30=-0.1435
- pos7=0.49, neg7=0.51
- vol7=0.0, trend7=0.0, news7=100

요청: 위 정보를 바탕으로 상대 의견의 약점을 반박하세요.
"""
```

## 2. 영향이 큰 시점의 뉴스 수집 여부

### ❌ 현재는 영향이 큰 시점의 뉴스를 선별하지 않음

**뉴스 수집 방식**:
- 최근 40일간의 **모든 뉴스**를 수집 (`lookback_days=40`)
- 날짜별로 **단순 집계**만 수행
- 영향이 큰 뉴스를 선별하는 로직 없음

**집계 방식**:
```python
# compute_finbert_features()에서:
# 1. 모든 뉴스에 FinBERT 점수 부여
# 2. 날짜별로 그룹화
# 3. 날짜별 평균 점수 계산
# 4. 7일/30일 윈도우로 집계

day_scores[d] = float(sum(scores) / n)  # 단순 평균
```

### 문제점

1. **모든 뉴스를 동일하게 취급**
   - 중요한 뉴스와 일반 뉴스의 가중치가 동일
   - 시장에 큰 영향을 준 뉴스가 묻힐 수 있음

2. **시간 가중치 없음**
   - 최근 뉴스와 오래된 뉴스의 가중치가 동일
   - 최근 뉴스가 더 중요할 수 있음

3. **뉴스 중요도 측정 없음**
   - 조회수, 공유 수, 출처 신뢰도 등 고려 안 함
   - 단순히 감성 점수만 사용

### 개선 방안

1. **시간 가중치 적용**
   ```python
   # 최근 뉴스에 더 높은 가중치
   weight = exp(-days_ago / decay_factor)
   weighted_score = score * weight
   ```

2. **영향이 큰 뉴스 선별**
   - 감성 점수의 절댓값이 큰 뉴스
   - 특정 키워드 포함 뉴스 (예: "earnings", "FDA approval")
   - 출처 신뢰도가 높은 뉴스

3. **뉴스 중요도 점수**
   - FinBERT 점수 + 추가 메트릭
   - 예: `importance = abs(sentiment_score) * source_credibility * recency_weight`

4. **상위 N개 뉴스만 사용**
   - 중요도 점수 기준으로 상위 뉴스만 선별
   - 집계 시 가중 평균 사용

## 3. 현재 뉴스 데이터 활용 요약

### ✅ LLM에 전달되는 내용

- **집계된 감성 통계**: 평균, 비율, 변동성, 추세
- **뉴스 개수**: 1일/7일 뉴스 개수
- **전체 컨텍스트**: 예측값, 불확실성, 가격 스냅샷 등

### ❌ LLM에 전달되지 않는 내용

- **개별 뉴스 기사**: 원문 제목/내용은 전달 안 됨
- **뉴스 출처**: 출처 정보는 전달 안 됨
- **뉴스 URL**: 링크는 전달 안 됨
- **뉴스 날짜별 상세**: 날짜별 상세 정보는 전달 안 됨

### 결론

현재 구조는:
- **집계된 통계만** LLM에 전달
- **개별 뉴스 기사는 전달 안 됨**
- **영향이 큰 뉴스를 선별하지 않음**
- **모든 뉴스를 동일하게 취급**

LLM은 집계된 감성 통계를 바탕으로 Opinion/Rebuttal/Revision을 생성합니다.

