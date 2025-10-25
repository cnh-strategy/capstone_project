# 🚀 MVP 하이브리드 주식 예측 시스템

## 📋 시스템 개요
**ML 예측 + LLM 해석**의 간단하고 효율적인 MVP 모델입니다.

## 🎯 핵심 특징

### ✅ **간단함**
- 복잡한 LLM 토론 제거
- 핵심 기능만 유지
- 직관적인 사용법

### ✅ **효율성**
- ML 예측 + LLM 해석의 최적 조합
- 빠른 실행 속도
- 명확한 결과 제공

### ✅ **투명성**
- 각 단계별 결과를 명확히 표시
- 에이전트별 예측값과 신뢰도 제공
- 해석 가능한 AI

### ✅ **실용성**
- 실제 투자 결정에 도움이 되는 정보 제공
- 주의사항 및 리스크 안내

## 🔄 간단한 분석 흐름

### 1️⃣ **TICKER 입력**
- 사용자가 주식 티커 입력 (예: RZLV, AAPL, TSLA)

### 2️⃣ **데이터 수집**
- **Technical Agent**: 기술적 지표 데이터 (OHLCV, RSI, MACD 등)
- **Fundamental Agent**: 재무 데이터 (P/E, P/B, ROE 등)
- **Sentimental Agent**: 감정 분석 데이터 (뉴스, 소셜미디어 등)

### 3️⃣ **모델 학습** (선택사항)
- 2022~2024년 데이터로 개별 Agent 학습
- 기존 모델이 있으면 로드, 없으면 새로 학습
- TCN, LSTM, Transformer 기반 모델

### 4️⃣ **ML 예측**
- 각 Agent별로 다음날 종가 예측
- Monte Carlo Dropout으로 불확실성 측정
- 가중평균으로 최종 합의 도출

### 5️⃣ **LLM 해석**
- ML 예측 결과에 대한 간단한 해석 제공
- 투자 의견 및 주의사항 안내
- 각 에이전트별 예측 요약

## 🏗️ 파일 구조

```
testing3/
├── mvp_main.py              # MVP 메인 시스템
├── mvp_dashboard.py         # MVP Streamlit 대시보드
├── run_mvp.py              # MVP 실행 스크립트
├── agents/                 # 에이전트들 (ML 기능 통합)
│   ├── base_agent.py      # 기본 에이전트 + ML 공통 기능
│   ├── technical_agent.py  # 기술적 에이전트 + TCN 모델
│   ├── fundamental_agent.py # 펀더멘털 에이전트 + LSTM 모델
│   └── sentimental_agent.py # 감정적 에이전트 + Transformer 모델
├── data/                   # 생성된 데이터
├── ml_modules/models/      # 학습된 모델들
└── requirements.txt        # 의존성
```

## 🚀 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. MVP 시스템 실행

#### 🖥️ **Streamlit 대시보드** (권장)
```bash
python3 run_mvp.py --mode dashboard
```

**사용법:**
1. 왼쪽 사이드바에서 주식 티커 입력
2. "🚀 전체 파이프라인 실행" 버튼 클릭
3. 자동으로 모든 단계가 순차 실행됩니다:
   - 🔍 데이터 수집 → 🎯 모델 학습 → 📈 ML 예측 → 💭 LLM 해석
4. 결과를 탭별로 확인

#### 💻 **CLI 모드**
```bash
# 전체 분석
python3 run_mvp.py --mode cli --ticker RZLV --step all

# 개별 단계
python3 run_mvp.py --mode cli --ticker RZLV --step search    # 데이터 수집만
python3 run_mvp.py --mode cli --ticker RZLV --step train     # 모델 학습만
python3 run_mvp.py --mode cli --ticker RZLV --step predict   # ML 예측만
python3 run_mvp.py --mode cli --ticker RZLV --step interpret # LLM 해석만
```

#### 🔧 **직접 실행**
```bash
# MVP 메인 시스템
python3 mvp_main.py --ticker RZLV --step all

# MVP 대시보드
streamlit run mvp_dashboard.py
```

## 📊 예상 결과

### 🎯 **최종 예측**
- ML 모델들의 가중평균 예측값
- 각 에이전트별 신뢰도 (β) 제공

### 📈 **에이전트별 예측**
- **Technical Agent**: 기술적 분석 기반 예측
- **Fundamental Agent**: 펀더멘털 분석 기반 예측
- **Sentimental Agent**: 감정 분석 기반 예측

### 💭 **LLM 해석**
- 예측 결과에 대한 투자 의견
- 각 에이전트별 예측 요약
- 주의사항 및 리스크 안내

## 🎯 지원 종목

- **기술주**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA
- **금융주**: JPM, BAC, WFC, GS, MS
- **에너지주**: XOM, CVX, COP, EOG
- **기타**: RZLV, SPY, QQQ

## ⚠️ 주의사항

1. **투자 조언 아님**: 이 시스템은 교육 및 연구 목적으로만 사용되어야 합니다.
2. **과거 데이터 기반**: 예측은 과거 데이터를 기반으로 하며, 미래 성과를 보장하지 않습니다.
3. **추가 분석 필요**: 실제 투자 결정 시 추가적인 분석과 전문가 상담이 필요합니다.
4. **리스크 관리**: 투자에는 항상 손실 위험이 따르므로 신중한 판단이 필요합니다.

## 🔧 기술 스택

- **ML**: PyTorch, scikit-learn, pandas, numpy
- **데이터**: yfinance
- **시각화**: Plotly, Matplotlib
- **UI**: Streamlit
- **언어**: Python 3.8+

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로만 사용되어야 합니다.

---

**🚀 MVP 하이브리드 주식 예측 시스템 | ML 예측 + LLM 해석 | 간단하고 효율적**
