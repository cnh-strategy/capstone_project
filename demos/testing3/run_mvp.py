#!/usr/bin/env python3
"""
MVP 하이브리드 주식 예측 시스템 실행 스크립트
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """의존성 확인"""
    print("🔍 의존성 확인 중...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn',
        'yfinance', 'matplotlib', 'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 의존성이 설치되어 있습니다.")
    return True

def run_streamlit(port=8501):
    """Streamlit 대시보드 실행"""
    print(f"🚀 MVP Streamlit 대시보드 실행 중... (포트: {port})")
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "mvp_dashboard.py", 
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit 실행 실패: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Streamlit 종료")

def run_cli(ticker, step, force_retrain):
    """CLI 모드 실행"""
    print(f"🎯 MVP CLI 모드 실행: {ticker}")
    
    try:
        cmd = [
            sys.executable, "mvp_main.py",
            "--ticker", ticker,
            "--step", step
        ]
        
        if force_retrain:
            cmd.append("--force-retrain")
        
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ CLI 실행 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description='MVP 하이브리드 주식 예측 시스템')
    parser.add_argument('--mode', choices=['dashboard', 'cli', 'check'], 
                       default='dashboard', help='실행 모드')
    parser.add_argument('--port', type=int, default=8501, help='Streamlit 포트')
    parser.add_argument('--ticker', type=str, default='RZLV', help='주식 티커')
    parser.add_argument('--step', type=str, 
                       choices=['search', 'train', 'predict', 'interpret', 'all'],
                       default='all', help='실행할 단계')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='모델 강제 재학습')
    
    args = parser.parse_args()
    
    print("🚀 MVP 하이브리드 주식 예측 시스템")
    print("=" * 50)
    
    if args.mode == 'check':
        check_dependencies()
    elif args.mode == 'dashboard':
        if check_dependencies():
            run_streamlit(args.port)
    elif args.mode == 'cli':
        if check_dependencies():
            run_cli(args.ticker, args.step, args.force_retrain)

if __name__ == "__main__":
    main()
