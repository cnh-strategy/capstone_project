# AI Stock Analysis Debate System

## 🎯 프로젝트 개요

Multi-Agent 시스템을 통한 주식 분석 및 예측 플랫폼입니다. 세 개의 전문 에이전트(Sentimental, Technical, Fundamental)가 토론을 통해 더 정확한 투자 의견을 도출합니다.

## 🚀 주요 기능

### 🤖 Multi-Agent Debate System
- **Sentimental Agent**: 센티멘탈 분석 (뉴스, 소셜미디어, 투자자 심리)
- **Technical Agent**: 기술적 분석 (차트 패턴, 지표, 거래량)
- **Fundamental Agent**: 펀더멘털 분석 (재무제표, 밸류에이션, 성장성)

### 📊 시각화 기능
- **🌐 Streamlit 웹 대시보드**: 인터랙티브한 웹 인터페이스로 토론 실행 및 결과 확인
- **라운드별 의견 변화**: 에이전트들의 의견이 라운드를 거치며 어떻게 변화하는지 시각화
- **의견 일치도 분석**: 에이전트들 간의 의견 일치도를 측정하고 시각화
- **반박/지지 네트워크**: 에이전트들 간의 반박과 지지 패턴을 네트워크로 표현
- **주식 컨텍스트**: yfinance를 활용한 주식 기본 정보 및 차트
- **인터랙티브 대시보드**: Plotly를 활용한 상호작용 가능한 대시보드
- **종합 리포트**: 모든 분석 결과를 종합한 리포트 생성

## 📁 프로젝트 구조

```
capstone/
├── 📁 agents/                    # 에이전트 모듈들
│   ├── __init__.py              # 에이전트 패키지 초기화
│   ├── base_agent.py            # 기본 에이전트 클래스
│   ├── fundamental_agent.py     # 펀더멘털 분석 에이전트
│   ├── sentimental_agent.py     # 센티멘탈 분석 에이전트
│   ├── strategy_agent.py        # 전략 에이전트
│   └── technical_agent.py       # 기술적 분석 에이전트
├── 📁 demos/                     # 데모 및 테스트 파일들
│   ├── notebooks/               # Jupyter 노트북 데모
│   ├── testing/                 # 테스트 파일들
│   └── visualization_demos/     # 시각화 데모
├── 📁 outputs/                   # 출력 파일들
│   ├── demo_reports/            # 데모 리포트
│   └── reports/                 # 생성된 리포트
├── 📄 main.py                   # 🎯 메인 실행 파일 (Streamlit 자동 실행)
├── 📄 debate_agent.py           # 토론 시스템 코어 로직
├── 📄 streamlit_dashboard.py    # 🌐 Streamlit 웹 대시보드
├── 📄 visualization.py          # 🎨 시각화 모듈
├── 📄 prompts.py                # LLM 프롬프트 템플릿
├── 📄 requirements.txt          # 의존성 패키지 목록
└── 📄 README.md                # 프로젝트 문서
```

## 🛠️ 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
export CAPSTONE_OPENAI_API="your_openai_api_key_here"

# ML 모듈 사용시 (선택적)
export FINNHUB_API_KEY="your_finnhub_api_key_here"
```

또는 `.env` 파일 생성:
```bash
# .env 파일 생성
cp env_example.txt .env
# .env 파일을 편집하여 API 키 설정
```

### 3. 실행 방법

#### 🌐 Streamlit 웹 대시보드 (권장)
```bash
python main.py
```
- 자동으로 Streamlit 대시보드가 시작됩니다
- 브라우저에서 `http://localhost:8501` 접속
- 인터랙티브한 웹 인터페이스로 토론 실행 및 결과 확인

#### 📊 직접 Streamlit 실행
```bash
streamlit run streamlit_dashboard.py --server.port 8501
```

#### 💻 커맨드라인 실행 (기본 토론만)
```bash
python debate_agent.py
```

## 🎨 시각화 기능 사용법

### 🌐 Streamlit 웹 대시보드 (권장)
```bash
python main.py
```
- **인터랙티브 웹 인터페이스**: 브라우저에서 직관적인 조작
- **실시간 토론 진행**: 진행 상황을 실시간으로 확인
- **종합 시각화**: 모든 분석 결과를 한 번에 확인
- **탭 기반 구성**: 
  - 최종의견 표
  - 투자의견 표  
  - 최종 예측 비교
  - 라운드별 의견 변화
  - 반박/지지 패턴

### 📊 프로그래밍 방식 사용
```python
from visualization import DebateVisualizer

# 시각화 객체 생성
visualizer = DebateVisualizer()

# 라운드별 의견 변화
visualizer.plot_round_progression(logs, final)

# 인터랙티브 대시보드
visualizer.create_interactive_dashboard(logs, final, "AAPL")

# 전체 리포트 생성
visualizer.generate_report(logs, final, "AAPL")
```

## 📊 시각화 종류

### 🌐 Streamlit 웹 대시보드 기능

#### 1. 최종의견 표
- 각 에이전트의 최종 예측 가격과 투자의견을 표 형태로 표시
- 에이전트별 예측 가격과 근거를 한눈에 비교

#### 2. 투자의견 표
- 라운드별 에이전트 의견 상세 내역
- 각 라운드에서의 예측 가격과 분석 근거
- 반박/지지 결과와 메시지

#### 3. 최종 예측 비교
- 에이전트별 최종 예측 가격을 막대 차트로 시각화
- 평균선과 현재가 비교
- 예측 범위와 변동성 분석

#### 4. 라운드별 의견 변화
- 에이전트들의 예측 가격이 라운드를 거치며 어떻게 변화하는지 시각화
- 현재가 기준선과 함께 추이 확인
- 에이전트별 색상 구분

#### 5. 반박/지지 패턴
- 에이전트들 간의 반박과 지지 패턴을 막대 차트로 표현
- 라운드별 반박/지지 상세 내역
- 반박/지지 비율과 통계

#### 6. 주식 컨텍스트
- yfinance를 활용한 실제 주식 데이터 시각화
- 최근 7일 주가 차트
- 시가총액, PER, 거래량 등 기본 정보

## 🔧 주요 클래스

### DebateSystem (main.py)
```python
class DebateSystem:
    def __init__(self)
    def create_agents(self) -> Dict[str, Any]
    def run_debate(self, ticker: str, rounds: int = 1) -> tuple
    def show_visualization_options(self, logs: List, final: Dict, ticker: str)
```

### Debate (debate_agent.py)
```python
class Debate:
    def __init__(self, agents: Dict[str, BaseAgent], verbose: bool = False)
    def run(self, ticker: str, rounds: int = 1) -> Tuple[List[RoundLog], Dict]
```

### DebateVisualizer
```python
class DebateVisualizer:
    def plot_round_progression(self, logs, final, save_path=None)
    def plot_consensus_analysis(self, logs, final, save_path=None)
    def plot_rebuttal_network(self, logs, save_path=None)
    def plot_stock_context(self, ticker, period="1mo", save_path=None)
    def create_interactive_dashboard(self, logs, final, ticker)
    def generate_report(self, logs, final, ticker, save_dir="./reports")
```

## 🌟 Streamlit 대시보드 주요 특징

### 🎯 사용자 친화적 인터페이스
- **직관적인 사이드바**: 종목 선택, 라운드 수 조정, 차트 옵션 설정
- **실시간 진행 상황**: 토론 진행 과정을 실시간으로 표시
- **탭 기반 구성**: 다양한 분석 결과를 체계적으로 정리

### 📊 종합적인 시각화
- **인터랙티브 차트**: Plotly 기반의 상호작용 가능한 차트
- **실시간 데이터**: yfinance를 통한 최신 주식 정보
- **다양한 분석**: 예측 비교, 의견 변화, 반박/지지 패턴 등

### 🚀 성능 최적화
- **캐싱 시스템**: 동일한 토론 결과 재사용
- **진행 상황 표시**: 사용자 경험 향상을 위한 실시간 피드백
- **반응형 디자인**: 다양한 화면 크기에 최적화

## 📈 사용 예제

### 🌐 Streamlit 웹 대시보드 사용법

#### 1. 대시보드 시작
```bash
$ python main.py
🚀 자동으로 Streamlit 웹 대시보드를 시작합니다...
📱 브라우저에서 http://localhost:8501 에 접속하세요
⏹️  중지하려면 Ctrl+C를 누르세요
```

#### 2. 웹 인터페이스에서 토론 실행
1. **사이드바 설정**:
   - 종목 티커 입력 (예: AAPL, TSLA, MSFT)
   - 라운드 수 선택 (1-5)
   - 차트 옵션 설정

2. **토론 시작**:
   - "■ 토론 시작" 버튼 클릭
   - 실시간 진행 상황 확인
   - 각 라운드별 에이전트 의견 생성 과정 표시

3. **결과 확인**:
   - 토론 완료 후 자동으로 결과 표시
   - 탭을 통해 다양한 시각화 확인

#### 3. 예상 결과
```
■ 토론이 성공적으로 완료되었습니다!

■ 토론 결과 요약
- 분석 종목: AAPL
- 실행 라운드: 3
- 참여 에이전트: 3명
- 최종 평균 예측: 151.20
- 중앙값: 150.80
- 현재가: 150.00
```

### 💻 커맨드라인 사용법
```bash
$ python debate_agent.py
분석할 티커를 입력하세요 (예: PLTR, AAPL, TSLA): AAPL
라운드 수를 입력하세요 (기본=1): 2

=== Debate 결과 요약 ===
□ round 1 :
- sentimental : 150.25 / 센티멘탈 분석 결과...
- technical : 152.10 / 기술적 분석 결과...
- fundamental : 148.90 / 펀더멘털 분석 결과...

□ round 2 :
- sentimental : 151.20 / 수정된 센티멘탈 분석...
- technical : 151.80 / 수정된 기술적 분석...
- fundamental : 150.10 / 수정된 펀더멘털 분석...

□ 결론
- 목표가 : 151.20
- 이유 : 에이전트 의견을 종합한 결과(중앙값 기준)
```

## 🤖 ML 모듈 통합 (NEW!)

### ML 모듈 활성화
Sentimental 브랜치의 searcher와 predictor를 메인 브랜치에 통합할 수 있습니다:

```python
# ML 모듈 통합 모드
debate_system = DebateSystem(use_ml_modules=True)
logs, final = debate_system.run_debate("AAPL", rounds=3)
```

### ML 모듈 기능
- **Sentimental Agent**: 
  - 실시간 뉴스 수집 (Finnhub API)
  - FINBERT 임베딩으로 텍스트 분석
  - MLP 신경망 모델로 주가 예측
- **Technical Agent**:
  - FRED API로 매크로 경제 데이터 수집
  - 기술적 지표 자동 계산 (RSI, MA, 볼린저밴드)
  - Keras 딥러닝 모델로 예측
- **Fundamental Agent**:
  - 분기 재무제표 자동 수집
  - 시장 지수 데이터 통합
  - LightGBM 모델로 펀더멘털 예측
- **하이브리드 분석**: GPT + ML 모델 결합으로 정확도 향상

### 설정 방법
1. **API 키 설정**:
   ```bash
   export FINNHUB_API_KEY="your_finnhub_api_key"
   ```

2. **ML 모델 파일 복사**:
   ```bash
   # 각 브랜치에서 모델 파일 복사
   cp feature/sentimental/mlp_stock_model.pt ./
   cp -r feature/technical/model_artifacts ./
   cp -r feature/yezi-fundamental/fundamental_model_maker ./
   ```

3. **필요 패키지 설치**:
   ```bash
   pip install torch transformers lightgbm tensorflow scikit-learn
   ```

4. **사용 예제 실행**:
   ```bash
   python example_ml_integration.py
   ```

## 🎯 향후 계획

- [x] 웹 인터페이스 추가 (Streamlit 대시보드 완료)
- [x] ML 모듈 통합 (모든 브랜치 통합 완료)
  - [x] Sentimental 브랜치 통합
  - [x] Technical 브랜치 통합  
  - [x] Fundamental 브랜치 통합
- [ ] 실시간 데이터 스트리밍
- [ ] 백테스팅 기능
- [ ] 포트폴리오 최적화
- [ ] 알림 시스템
- [ ] 데이터베이스 연동
- [ ] 모바일 반응형 대시보드
- [ ] 사용자 인증 및 세션 관리

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.
