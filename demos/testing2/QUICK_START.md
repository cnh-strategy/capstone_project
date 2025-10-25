# 🚀 하이브리드 AI 주식 예측 시스템 - 빠른 시작 가이드

## 📋 시스템 개요
- **ML 기반 예측**: 3단계 학습 (사전훈련 → 상호학습 → 실시간 토론)
- **LLM 기반 토론**: 3개 전문 에이전트의 다중 라운드 토론
- **하이브리드 합의**: ML과 LLM 결과의 가중평균

## 🎯 핵심 파일 구조
```
testing2/
├── hybrid_main.py          # 하이브리드 시스템 메인
├── hybrid_dashboard.py     # Streamlit 대시보드
├── run_hybrid.py          # 실행 스크립트
├── agents/                # LLM 에이전트들
├── ml_models/             # ML 모델들
├── data/                  # 데이터 파일들
├── models/                # 훈련된 모델들
└── requirements.txt       # 의존성
```

## 🚀 빠른 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Streamlit 대시보드 실행
```bash
python3 run_hybrid.py --mode dashboard
```

### 3. 브라우저에서 접속
- **로컬**: http://localhost:8503
- **외부**: http://54.144.206.12:8503

## 🎮 사용 방법

### 1. 대시보드 설정
- **티커 입력**: RZLV (또는 다른 주식 티커)
- **분석 모드**: 하이브리드 분석 선택
- **토론 라운드**: 3 (권장)

### 2. 분석 실행
1. "🚀 분석 시작" 버튼 클릭
2. "📝 실시간 로그" 탭에서 진행 상황 모니터링
3. "📊 결과 시각화" 탭에서 결과 확인

## 📊 예상 결과
- **ML 예측**: ~8.23 (기술적/재무/감정 분석)
- **LLM 예측**: ~5.95 (3라운드 토론 합의)
- **최종 합의**: ~6.86 (가중평균)

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

## 📈 데이터 분할
- **사전훈련**: 2022-2024년 (753개 샘플)
- **상호학습**: 2025년 (193개 샘플)
- **실시간 토론**: 최근 1주일 (5개 샘플)

## 🎯 핵심 기능
- ✅ **실시간 로깅**: 모든 과정 모니터링
- ✅ **하이브리드 예측**: ML + LLM 결합
- ✅ **가중평균 합의**: 합리적인 최종 예측
- ✅ **yfinance 데이터**: 안정적인 데이터 수집
- ✅ **3단계 ML 학습**: 체계적인 모델 훈련

## 🆘 지원
문제가 발생하면 실시간 로그를 확인하고, 필요시 시스템을 재시작하세요.
