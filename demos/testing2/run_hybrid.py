#!/usr/bin/env python3
"""
시스템을 실행 통합 스크립트
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 필요 패키지 설치 유무 확인
def check_dependencies():
    print("# 의존성 패키지 확인 중...")
    
    # 필요 패키지 정의
    required_packages = [
        'streamlit', 'torch', 'pandas', 'numpy', 
        'plotly', 'yfinance', 'sklearn'
    ]

    # 누락된 패키지 정의
    missing_packages = []
    
    # 필요 패키지 확인
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    # 누락된 패키지 출력
    if missing_packages:
        print(f"\n >> 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print(">> 모든 필수 패키지가 설치되어 있습니다.")
    return True

# 환경 설정
def setup_environment():
    print("# 환경 설정 중...")
    
    # 필요한 디렉토리 생성
    directories = ['models', 'data', 'ml_data']

    # 디렉토리 생성
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"## {directory}/ 디렉토리 확인")
    
    # 환경 변수 확인
    env_vars = {
        'CAPSTONE_OPENAI_API': 'OpenAI API 키 (LLM 토론용)',
    }
    missing_env = []

    # 환경 변수 확인
    for var, description in env_vars.items():
        if not os.getenv(var):
            missing_env.append(f"{var}: {description}")
    
    # 누락된 환경 변수 출력
    if missing_env:
        print("\n >> 환경 변수 설정 권장:")
        for var in missing_env:
            print(f"  - {var}")
        print("\n >> 설정하지 않으면 해당 기능이 비활성화됩니다.")
    
    print(" >> 환경 설정 완료")

# Streamlit 대시보드 실행
def run_dashboard(port=8501):
    print(f"# Streamlit 대시보드 시작 (포트: {port})...")
    
    try:
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            'hybrid_dashboard.py', 
            '--server.port', str(port),
            '--server.headless', 'true'
        ]
        
        print(f"🚀 명령어: {' '.join(cmd)}")
        print(f"📱 브라우저에서 http://localhost:{port} 에 접속하세요")
        print("⏹️  중지하려면 Ctrl+C를 누르세요")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n >> 대시보드를 종료합니다.")
    except Exception as e:
        print(f" >> 대시보드 실행 실패: {e}")

# 커맨드라인 인터페이스 실행
def run_cli():
    print("💻 커맨드라인 인터페이스 시작...")
    
    try:
        subprocess.run([sys.executable, 'hybrid_main.py'])
    except Exception as e:
        print(f">> CLI 실행 실패: {e}")

# 티커 입력 시스템 실행
def run_ticker_system():
    print("# 티커 입력 시스템 시작...")
    
    try:
        subprocess.run([sys.executable, 'ticker_input_system.py'])
    except Exception as e:
        print(f">> 티커 시스템 실행 실패: {e}")

# 메인 함수
def main():

    # 인자 파서 생성
    parser = argparse.ArgumentParser(
        description='Hybrid Multi-Agent Debate System Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            사용 예제:
            python run_hybrid.py                    # 대시보드 실행 (기본)
            python run_hybrid.py --mode dashboard   # 대시보드 실행
            python run_hybrid.py --mode cli         # 커맨드라인 실행
            python run_hybrid.py --mode ticker      # 티커 입력 시스템 실행
            python run_hybrid.py --port 8502        # 다른 포트로 대시보드 실행
            python run_hybrid.py --check            # 의존성만 확인
        """
    )
    
    # 실행 모드 선택
    parser.add_argument(
        '--mode', 
        choices=['dashboard', 'cli', 'ticker'], 
        default='dashboard',
        help='실행 모드 선택 (기본: dashboard)'
    )
    
    # 대시보드 포트 번호
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='대시보드 포트 번호 (기본: 8501)'
    )
    
    # 의존성만 확인하고 종료
    parser.add_argument(
        '--check', 
        action='store_true',
        help='의존성만 확인하고 종료'
    )
    
    # 환경 설정 건너뛰기
    parser.add_argument(
        '--skip-setup', 
        action='store_true',
        help='환경 설정 건너뛰기'
    )

    # 인자 파싱
    args = parser.parse_args()
    
    print(">> Hybrid Multi-Agent Debate System")
    print("=" * 60)
    
    # 의존성 확인
    if not check_dependencies():
        if not args.check:
            print("\n >> 필수 패키지가 누락되어 실행할 수 없습니다.")
            print("requirements.txt를 사용하여 설치하세요:")
            print("pip install -r requirements.txt")
        return 1
    
    # 의존성만 확인하고 종료
    if args.check:
        print("\n >> 의존성 확인 완료")
        return 0
    
    # 환경 설정
    if not args.skip_setup:
        setup_environment()
    
    # 실행 모드에 따른 분기
    print(f"\n🎯 실행 모드: {args.mode}")
    
    if args.mode == 'dashboard':
        run_dashboard(args.port)
    elif args.mode == 'cli':
        run_cli()
    elif args.mode == 'ticker':
        run_ticker_system()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
