#!/usr/bin/env python3
"""
ML 모듈 통합 사용 예제
Sentimental 브랜치의 searcher와 predictor를 메인 브랜치에 통합하는 방법을 보여줍니다.
"""

import os
import sys
from main import DebateSystem

def main():
    print("🎯 ML 모듈 통합 예제")
    print("=" * 50)
    
    # 1. 기본 모드 (GPT만 사용)
    print("\n📊 1. 기본 모드 (GPT 기반 분석만)")
    print("-" * 30)
    
    debate_system_basic = DebateSystem(use_ml_modules=False)
    print("✅ 기본 모드로 DebateSystem 초기화 완료")
    print("   - GPT 기반 분석만 사용")
    print("   - 기존 방식과 동일한 동작")
    
    # 2. ML 모듈 통합 모드 (모든 에이전트)
    print("\n🤖 2. ML 모듈 통합 모드 (모든 에이전트)")
    print("-" * 30)
    
    # 환경변수 확인
    openai_key = os.getenv('CAPSTONE_OPENAI_API')
    finnhub_key = os.getenv('FINNHUB_API_KEY')
    
    if not openai_key:
        print("❌ CAPSTONE_OPENAI_API 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 API 키를 설정하세요.")
        return
    
    if not finnhub_key:
        print("⚠️  FINNHUB_API_KEY가 설정되지 않았습니다.")
        print("   ML 뉴스 수집 기능이 제한됩니다.")
    
    # ML 모델 파일 확인
    model_path = "mlp_stock_model.pt"
    if not os.path.exists(model_path):
        print(f"⚠️  ML 모델 파일을 찾을 수 없습니다: {model_path}")
        print("   Sentimental 브랜치에서 모델 파일을 복사하세요.")
    
    try:
        debate_system_ml = DebateSystem(use_ml_modules=True)
        print("✅ ML 모듈 통합 모드로 DebateSystem 초기화 완료")
        print("   - GPT + ML 모델 결합 분석")
        print("   - Sentimental: FINBERT + MLP 모델 + Finnhub 뉴스")
        print("   - Technical: Keras 모델 + FRED 매크로 데이터")
        print("   - Fundamental: LightGBM 모델 + 분기 보고서")
        
        # 3. 토론 실행 예제
        print("\n🚀 3. 토론 실행 예제")
        print("-" * 30)
        
        ticker = "AAPL"
        rounds = 1
        
        print(f"📈 {ticker} 종목 분석 시작...")
        print("   (실제 실행을 원하면 아래 주석을 해제하세요)")
        
        # 실제 토론 실행 (주석 해제하여 사용)
        # logs, final = debate_system_ml.run_debate(ticker, rounds)
        # print(f"✅ 토론 완료! 최종 예측: {final.get('mean_next_close', 'N/A')}")
        
    except Exception as e:
        print(f"❌ ML 모듈 초기화 실패: {e}")
        print("   필요한 패키지가 설치되어 있는지 확인하세요:")
        print("   pip install torch transformers requests")
    
    # 4. 설정 가이드
    print("\n📋 4. 설정 가이드")
    print("-" * 30)
    print("ML 모듈을 사용하려면:")
    print("1. .env 파일에 API 키 설정:")
    print("   CAPSTONE_OPENAI_API=your_key")
    print("   FINNHUB_API_KEY=your_key")
    print("   FRED_API_KEY=your_key")
    print()
    print("2. 필요한 패키지 설치:")
    print("   pip install torch transformers requests lightgbm scikit-learn")
    print()
    print("3. ML 모델 파일 복사:")
    print("   - Sentimental: mlp_stock_model.pt")
    print("   - Technical: model_artifacts/final_best.keras")
    print("   - Fundamental: fundamental_model_maker/2025/models22/final_lgbm.pkl")
    print()
    print("4. 코드에서 ML 모듈 활성화:")
    print("   debate_system = DebateSystem(use_ml_modules=True)")

def test_ml_modules():
    """ML 모듈 개별 테스트"""
    print("\n🧪 ML 모듈 개별 테스트")
    print("-" * 30)
    
    try:
        from agents.sentimental_modules import SentimentalModuleManager
        
        # 모듈 매니저 초기화
        manager = SentimentalModuleManager(
            use_ml_searcher=True,
            use_ml_predictor=True,
            finnhub_api_key=os.getenv('FINNHUB_API_KEY'),
            model_path="mlp_stock_model.pt"
        )
        
        # 테스트 실행
        ticker = "AAPL"
        current_price = 150.0
        
        print(f"📊 {ticker} 센티멘탈 분석 테스트...")
        enhanced_data = manager.get_enhanced_sentimental_data(ticker, current_price)
        
        print("✅ ML 모듈 테스트 완료!")
        print(f"   센티멘탈: {enhanced_data.get('sentiment', 'N/A')}")
        print(f"   ML 예측: {enhanced_data.get('ml_prediction', 'N/A')}")
        print(f"   신뢰도: {enhanced_data.get('ml_confidence', 0.0):.2f}")
        
    except Exception as e:
        print(f"❌ ML 모듈 테스트 실패: {e}")

if __name__ == "__main__":
    main()
    test_ml_modules()
