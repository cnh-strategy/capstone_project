# 🚀 새로운 하이브리드 주식 예측 시스템

## 📋 시스템 개요
원래 `@capstone/` 구조를 기반으로 ML과 LLM을 통합한 새로운 하이브리드 시스템입니다.

## 🔄 새로운 분석 흐름

### 1️⃣ **TICKER 입력**
- 사용자가 주식 티커 입력 (예: RZLV, AAPL, TSLA)

### 2️⃣ **각 Agent의 Searcher**
- **목적**: 2022~2025년 CSV 파일 생성
- **기능**: 
  - Fundamental Agent: 재무 데이터 수집
  - Technical Agent: 기술적 지표 데이터 수집  
  - Sentimental Agent: 감정 분석 데이터 수집
- **출력**: `{TICKER}_{agent_type}_data.csv`

### 3️⃣ **각 Agent의 Trainer** (선택사항)
- **목적**: 2022~2024년 데이터로 개별 Agent 학습
- **기능**:
  - 이미 학습된 `.pt` 파일이 있으면 선택하여 로드
  - 없으면 새로 학습 실행
  - 실행하면 기존 모델 업데이트
- **출력**: `{TICKER}_{agent_type}_model.pt`

### 4️⃣ **각 Agent의 Predicter**
- **목적**: 상호학습 + 예측
- **기능**:
  - 최근 1년 데이터로 상호학습 진행
  - 상호학습 후 최근 7일 데이터로 다음날 종가 예측
- **출력**: 각 Agent별 예측값과 신뢰도

### 5️⃣ **Debate Round 진행**
#### 5-1. **각 Agent의 Reviewer Draft**
- **목적**: Opinion 생성
- **기능**: Predicter 출력 종가 데이터를 바탕으로 Opinion(종가, 이유, 신뢰도) 생성

#### 5-2. **각 Agent의 Reviewer Rebut**
- **목적**: 반론/지지 의견 형성
- **기능**: LLM을 통해 다른 Agent의 Opinion에 대해 신뢰도 및 결과값 비교하여 반론/지지 의견 형성

#### 5-3. **각 Agent의 Reviewer Revise**
- **목적**: 예측 수정
- **기능**: 다른 Agent들의 신뢰도를 바탕으로 예측 종가 수정, 이유 형성

## 🏗️ 파일 구조

```
testing3/
├── new_main.py              # 새로운 메인 진입점 (간단한 구조)
├── agents/                  # 에이전트들 (원래 capstone 구조 + ML 기능 통합)
│   ├── base_agent.py       # 기본 에이전트 클래스 + ML 공통 기능
│   ├── fundamental_agent.py # 펀더멘털 에이전트 + ML 기능
│   ├── technical_agent.py   # 기술적 에이전트 + ML 기능
│   └── sentimental_agent.py # 감정적 에이전트 + ML 기능
├── debate_agent.py         # 토론 시스템 (원래 capstone 구조)
├── prompts.py              # 프롬프트 정의
├── streamlit_dashboard.py  # 대시보드
├── data/                   # 생성된 데이터
├── ml_modules/models/      # 학습된 모델들
└── requirements.txt        # 의존성
```

## 🎯 핵심 장점

1. **깔끔한 구조**: 원래 capstone의 체계적인 구조 유지
2. **통합된 에이전트**: 각 Agent에 ML 기능이 통합되어 일관성 있음
3. **단계별 진행**: 각 단계가 명확히 분리되어 이해하기 쉬움
4. **선택적 실행**: Trainer는 선택사항으로 유연성 제공
5. **ML + LLM 통합**: 정량적 예측과 정성적 토론의 완벽한 결합
6. **공통 기능**: BaseAgent에 ML 공통 기능이 있어 코드 중복 최소화
7. **실시간 업데이트**: 기존 모델을 업데이트할 수 있는 유연성

## 🚀 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 새로운 간단한 시스템 실행
python3 new_main.py --ticker RZLV --step all

# 3. 개별 단계 실행
python3 new_main.py --ticker RZLV --step search    # 데이터 수집만
python3 new_main.py --ticker RZLV --step train     # 모델 학습만
python3 new_main.py --ticker RZLV --step predict   # 예측만
python3 new_main.py --ticker RZLV --step debate    # 토론만

# 4. Streamlit 대시보드 실행 (기존)
streamlit run streamlit_dashboard.py
```

## 📊 예상 결과

- **ML 예측**: 각 Agent별 정량적 예측값
- **LLM 토론**: 3라운드 토론을 통한 정성적 분석
- **최종 합의**: ML과 LLM 결과의 종합적 판단
