# 🧠 AI Stock Debate System (MVP)

> **신뢰도 기반 Revise를 수행하는 AI 주식 토론 시스템**

## 📋 프로젝트 개요

여러 전문 AI 에이전트가 주식에 대해 토론하고, 신뢰도 기반으로 의견을 수정하여 최종 예측을 도출하는 시스템입니다.

### 🎯 핵심 기능
- **다중 전문 에이전트**: Technical, Sentimental, MacroSenti 분석가
- **신뢰도 기반 Revise**: 각 에이전트의 불확실성(σ)을 기반으로 가중치 계산
- **실시간 대시보드**: Streamlit을 통한 인터랙티브 시각화
- **실시간 주가 연동**: Yahoo Finance API를 통한 현재가 정보

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ TechnicalAgent  │    │ SentimentalAgent│    │ MacroSentiAgent │
│   (공격적)      │    │   (중립적)      │    │   (거시경제)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   DebateAgent   │
                    │  (토론 관리)    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Streamlit 대시보드│
                    │  (시각화)       │
                    └─────────────────┘
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:
```bash
CAPSTONE_OPENAI_API=your_openai_api_key_here
EODHD_API_KEY=your_eodhd_api_key_here  # SentimentalAgent 사용 시
```

### 3. Streamlit 대시보드 실행
```bash
streamlit run streamlit_dashboard.py
```

### 4. 토론 실행
1. 사이드바에서 종목 티커 입력 (예: AAPL, TSLA, NVDA)
2. 라운드 수 설정 (1-5)
3. "■ 토론 시작" 버튼 클릭

## 🎭 에이전트 소개

### 📈 TechnicalAgent (기술적 분석가)
- **특징**: 공격적, 차트 패턴 분석
- **예측 범위**: ±15%
- **분석 요소**: 이동평균, RSI, MACD, 볼린저밴드 등

### 💭 SentimentalAgent (센티멘탈 분석가)
- **특징**: 중립적, 시장 심리 분석
- **예측 범위**: ±10%
- **분석 요소**: 뉴스, 소셜미디어, 시장 분위기 등

### 📊 MacroSentiAgent (매크로 센티멘탈 분석가)
- **특징**: 거시경제 지표 기반 분석
- **예측 범위**: ±12%
- **분석 요소**: SPY, QQQ, VIX, 금리, 환율, 원자재 가격 등
- **모델**: LSTM 기반 다중 자산 예측 파이프라인

## 🔄 토론 프로세스

### Round 0: 초기 의견 생성
각 에이전트가 독립적으로 주식 분석 및 예측 수행

### Round 1-N: 토론 및 수정
1. **Rebuttal**: 다른 에이전트의 의견에 대한 반박/지지
2. **Revise**: 신뢰도 기반으로 자신의 의견 수정
3. **Ensemble**: 최종 예측가 계산

### 신뢰도 계산 공식
```
β_i = (1/σ_i) / Σ(1/σ_j)
revised_price = β_i × my_price + (1-β_i) × weighted_others
```

## 📊 대시보드 기능

### 🎯 주요 탭
- **최종의견 표**: 각 에이전트의 최종 예측가와 근거
- **투자의견 표**: 라운드별 의견 변화 상세 내역
- **최종 예측 비교**: 에이전트별 예측가 막대차트
- **라운드별 의견 변화**: 시간에 따른 예측가 변화 추이
- **반박/지지 패턴**: 에이전트 간 상호작용 분석

### 📈 시각화 기능
- 실시간 주가 차트 (7일)
- 에이전트별 예측가 비교
- 라운드별 의견 변화 추이
- 반박/지지 패턴 분석

## 🛠️ 기술 스택

- **AI/ML**: PyTorch, TensorFlow, Transformers
- **데이터**: Yahoo Finance API, yfinance, EODHD API
- **시각화**: Streamlit, Plotly
- **언어**: Python 3.10+
- **LLM**: OpenAI API (GPT-4, GPT-4o-mini)

## 📁 프로젝트 구조

```
├── agents/
│   ├── base_agent.py          # 기본 에이전트 클래스
│   ├── debate_agent.py        # 토론 관리 에이전트
│   ├── macro_agent.py         # 매크로 센티멘탈 분석가
│   ├── technical_agent.py     # 기술적 분석가
│   └── sentimental_agent.py   # 센티멘탈 분석가
├── config/
│   └── agents.py              # 에이전트 하이퍼파라미터 설정
├── core/
│   ├── data_set.py            # 데이터셋 관리
│   ├── utils_datetime.py      # 날짜 유틸리티
│   ├── macro_classes/         # 매크로 분석 클래스
│   │   ├── macro_llm.py       # LLM 설명기
│   │   ├── macro_sub.py       # 매크로 서브루틴
│   │   └── ...
│   ├── sentimental_classes/   # 센티멘탈 분석 클래스
│   │   ├── eodhd_client.py    # EODHD API 클라이언트
│   │   ├── finbert_utils.py   # FinBERT 유틸리티
│   │   └── ...
│   └── technical_classes/     # 기술적 분석 클래스
│       ├── technical_base_agent.py
│       └── ...
├── scripts/
│   └── sentimental_demo/     # 센티멘탈 데모 스크립트
├── notebooks/                  # 테스트 및 실험 노트북
│   ├── macro_test.ipynb       # 매크로 에이전트 테스트
│   ├── technical_test.ipynb   # 기술적 분석 테스트
│   ├── technical_test_1114.ipynb # 기술적 분석 테스트 (버전 관리)
│   └── test_sentimental_pipeline.ipynb # 센티멘탈 파이프라인 테스트
├── data/                      # 데이터 저장소
├── models/                    # 학습된 모델
├── prompts.py                # LLM 프롬프트
├── streamlit_dashboard.py    # 메인 대시보드
└── requirements.txt           # 의존성 패키지
```

## 🎯 주요 특징

### ✨ MVP 완성
- 신뢰도 기반 Revise 알고리즘 구현
- 다중 전문 에이전트 완전 구현 (Technical, Sentimental, MacroSenti)
- 실시간 대시보드 완성
- Gradient 기반 Feature Importance 분석 (MacroSentiAgent)

### 🔬 과학적 접근
- 불확실성(uncertainty) 기반 신뢰도 계산
- 베이지안 접근법을 통한 의견 수정
- 앙상블 학습을 통한 예측 정확도 향상

### 🎨 사용자 경험
- 직관적인 Streamlit 인터페이스
- 실시간 진행 상황 표시
- 인터랙티브 차트 및 시각화

## 🚀 향후 계획

- [ ] 더 많은 에이전트 추가 (Quantitative, ESG 등)
- [ ] 백테스팅 기능 추가
- [ ] 모바일 앱 개발
- [ ] API 서비스 제공

