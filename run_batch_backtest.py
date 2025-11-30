import argparse
import os
import sys
import traceback
from typing import List

# 현재 디렉토리(프로젝트 루트)를 경로에 추가하여 backtest 모듈을 찾을 수 있게 함
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from backtest import RollingBacktester

def run_batch_backtest(
    tickers: List[str], 
    predict_days: int, 
    rounds: int, 
    start_date: str = None
):
    """
    여러 종목에 대해 순차적으로 Rolling Backtest를 수행합니다.
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting Batch Backtest Simulation")
    print(f"🎯 Target Tickers: {tickers}")
    print(f"📅 Predict Days (N): {predict_days}")
    print(f"🔄 Debate Rounds: {rounds}")
    print(f"{'='*80}\n")
    
    failed_tickers = []
    success_tickers = []
    
    for ticker in tickers:
        print(f"\n{'#'*80}")
        print(f"▶️ Processing Ticker: {ticker}")
        print(f"{'#'*80}\n")
        
        try:
            # RollingBacktester 인스턴스 생성
            runner = RollingBacktester(
                ticker=ticker,
                start_date=start_date,
                predict_days=predict_days,
                rounds=rounds,
                auto_analyze=True  # 분석 및 그래프 저장 자동 수행
            )
            
            # 1. 데이터 준비
            runner.prepare_data()
            
            # 2. 백테스트 루프 실행
            runner.run_loop()
            
            print(f"\n✅ Successfully finished backtest for {ticker}")
            success_tickers.append(ticker)
            
        except Exception as e:
            print(f"\n❌ Failed backtest for {ticker}: {e}")
            traceback.print_exc()
            failed_tickers.append(ticker)
            
    # 최종 리포트
    print(f"\n{'='*80}")
    print("📊 Batch Backtest Execution Report")
    print(f"{'='*80}")
    print(f"✅ Success ({len(success_tickers)}): {', '.join(success_tickers)}")
    if failed_tickers:
        print(f"❌ Failed ({len(failed_tickers)}): {', '.join(failed_tickers)}")
    else:
        print("✨ All tickers completed successfully!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch backtests for multiple tickers")
    
    # 기본값으로 NVDL, AAPL, TSLA 설정
    parser.add_argument(
        "--tickers", 
        type=str, 
        nargs="+", 
        default=["NVDL", "AAPL", "TSLA"], 
        help="Target Tickers list (default: NVDL AAPL TSLA)"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=5, 
        help="Number of days to predict (N)"
    )
    
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=3, 
        help="Debate rounds per day"
    )
    
    parser.add_argument(
        "--start", 
        type=str, 
        default=None, 
        help="Start date (YYYY-MM-DD). If not set, auto-calculated."
    )
    
    args = parser.parse_args()
    
    run_batch_backtest(
        tickers=args.tickers,
        predict_days=args.days,
        rounds=args.rounds,
        start_date=args.start
    )

