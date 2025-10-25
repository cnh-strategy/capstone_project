# 🎯 핵심 실행 파일 목록

## 📁 최종 파일 구조 (실행 필수)

```
testing2/
├── 🚀 hybrid_main.py          # 하이브리드 시스템 메인 (필수)
├── 📊 hybrid_dashboard.py     # Streamlit 대시보드 (필수)
├── 🎮 run_hybrid.py          # 실행 스크립트 (필수)
├── 📖 QUICK_START.md         # 빠른 시작 가이드
├── 📋 ESSENTIAL_FILES.md     # 이 파일
├── 📦 requirements.txt       # 의존성 (필수)
├── 🚫 .gitignore            # Git 무시 파일
├── agents/                   # LLM 에이전트들 (필수)
│   ├── __init__.py
│   ├── base_agent.py
│   ├── fundamental_agent.py
│   ├── fundamental_modules.py
│   ├── sentimental_agent.py
│   ├── sentimental_modules.py
│   ├── technical_agent.py
│   └── technical_modules.py
├── ml_models/                # ML 모델들 (필수)
│   ├── __init__.py
│   ├── train_agents.py
│   ├── stage2_trainer.py
│   ├── debate_system.py
│   └── agent_utils.py
├── data/                     # RZLV 데이터 (필수)
│   ├── RZLV_*_pretrain.csv
│   ├── RZLV_*_mutual.csv
│   ├── RZLV_*_test.csv
│   └── single_ticker_builder.py
└── models/                   # 훈련된 모델들 (필수)
    ├── fundamental_agent.pt
    ├── sentimental_agent.pt
    └── technical_agent.pt
```

## 🗑️ 제거된 파일들

### ❌ 중복/불필요한 파일들
- `main.py` (원본 LLM 시스템)
- `debate_agent.py` (원본 토론 시스템)
- `prompts.py` (프롬프트 템플릿)
- `README.md` (상세 문서)
- `agents/strategy_agent.py` (사용 안함)
- `data/dataset_builder.py` (중복)

## 🚀 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Streamlit 실행
```bash
python3 run_hybrid.py --mode dashboard
```

### 3. 브라우저 접속
- **로컬**: http://localhost:8503
- **외부**: http://54.144.206.12:8503

## ✅ 핵심 기능

### 🎯 하이브리드 분석
- **ML 예측**: 3단계 학습 (사전훈련 → 상호학습 → 실시간 토론)
- **LLM 토론**: 3개 전문 에이전트의 다중 라운드 토론
- **최종 합의**: ML과 LLM 결과의 가중평균

### 📊 실시간 모니터링
- **실시간 로그**: 모든 과정 모니터링
- **결과 시각화**: 예측 결과 및 토론 과정
- **데이터 품질**: 수집된 데이터 검증

## 🎮 사용 시나리오

1. **티커 입력**: RZLV (또는 다른 주식)
2. **분석 모드**: 하이브리드 분석 선택
3. **토론 라운드**: 3 (권장)
4. **결과 확인**: 실시간 로그 및 시각화

## 🔧 문제 해결

### Streamlit 접속 불가
```bash
# 포트 확인
ss -tlnp | grep 8503

# 재시작
pkill -f streamlit
python3 run_hybrid.py --mode dashboard
```

### 의존성 오류
```bash
# 의존성 확인
python3 run_hybrid.py --check

# 재설치
pip install -r requirements.txt --force-reinstall
```

## 📈 예상 결과
- **ML 예측**: ~8.23 (기술적/재무/감정 분석)
- **LLM 예측**: ~5.95 (3라운드 토론 합의)
- **최종 합의**: ~6.86 (가중평균)

## 🎉 완료!

이제 **최소한의 핵심 파일들로 구성된 완전한 하이브리드 시스템**을 사용할 수 있습니다!
