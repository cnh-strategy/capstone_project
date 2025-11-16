# 뉴스 데이터 수집 실행 플로우

## 현재 구조에서 뉴스 데이터 수집 시점

### ❌ searcher()에서는 실행되지 않음

`SentimentalAgent.searcher()` 메서드는:
- 가격 데이터셋만 로드 (`load_dataset`)
- StockData 객체 생성
- **뉴스 데이터 수집은 하지 않음**

```python
def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
    # 1. 데이터셋 빌드/로드
    build_dataset(...)  # 가격 데이터만
    load_dataset(...)   # 가격 데이터만
    
    # 2. StockData 생성
    self.stockdata = StockData(...)
    
    # 3. 뉴스 수집 없음!
    return torch.tensor(X_latest, ...)
```

### ✅ build_ctx()에서 실행됨

뉴스 데이터는 `build_ctx()` 메서드에서 수집됩니다:

```python
def build_ctx(self, asof_date_kst: Optional[str] = None):
    # 1. 예측값 생성
    pred_close, ... = self._predict_next_close()
    
    # 2. 뉴스 데이터 수집 및 감성 분석 ⭐
    news_feats = build_finbert_news_features(
        self.ticker, asof_date_kst, base_dir=os.path.join("data", "raw", "news")
    )
    
    # 3. 컨텍스트 구성
    ctx = {
        "feature_importance": {
            "sentiment_summary": news_feats["sentiment_summary"],
            ...
        }
    }
    return ctx
```

### 실행 플로우

```
DebateAgent.get_opinion()
  ↓
agent.searcher()           # 가격 데이터만 로드, 뉴스 수집 없음
  ↓
agent.predict()            # 예측 수행
  ↓
agent.reviewer_draft()     # Opinion 생성
  ↓
self.build_ctx()           # ⭐ 여기서 뉴스 수집 실행!
  ↓
build_finbert_news_features()
  ↓
load_or_fetch_news()       # EODHD에서 뉴스 수집
  ↓
FinBERT 감성 분석
  ↓
피처 집계 및 반환
```

### 호출 시점

`build_ctx()`는 다음 메서드에서 호출됩니다:

1. **reviewer_draft()**: Opinion 생성 시
   ```python
   def reviewer_draft(self, stock_data, target):
       ctx = self.build_ctx()  # ← 뉴스 수집 실행
       # LLM으로 reason 생성
   ```

2. **reviewer_rebuttal()**: Rebuttal 생성 시
   ```python
   def reviewer_rebuttal(self, ...):
       ctx = self.build_ctx()  # ← 뉴스 수집 실행
       # LLM으로 rebuttal 생성
   ```

3. **reviewer_revise()**: Revision 생성 시
   ```python
   def reviewer_revise(self, ...):
       ctx = self.build_ctx()  # ← 뉴스 수집 실행
       # LLM으로 revised reason 생성
   ```

### 문제점

현재 구조의 문제:
- **searcher()에서 뉴스 수집이 없음**
- 뉴스 수집은 `build_ctx()`가 호출될 때만 실행됨
- 즉, Opinion/Rebuttal/Revision 생성 시에만 뉴스 데이터가 수집됨
- **예측(predict) 단계에서는 뉴스 데이터를 사용하지 않음**

### 개선 방안

1. **searcher()에서 뉴스 수집 추가**
   ```python
   def searcher(self, ticker, rebuild=False):
       # 기존 코드...
       
       # 뉴스 데이터 수집 추가
       asof_date = datetime.now().strftime("%Y-%m-%d")
       news_feats = build_finbert_news_features(
           ticker, asof_date, base_dir=os.path.join("data", "raw", "news")
       )
       
       # StockData에 뉴스 피처 저장
       self.stockdata.news_features = news_feats
       
       return torch.tensor(X_latest, ...)
   ```

2. **예측 시 뉴스 데이터 활용**
   - 뉴스 피처를 모델 입력에 포함
   - 또는 예측 confidence 조정에 활용

