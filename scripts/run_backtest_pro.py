import sys
import os
import argparse
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtester import Backtester

def main():
    parser = argparse.ArgumentParser(description="Unified Backtesting Module")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Target Ticker Symbol")
    parser.add_argument("--days", type=int, default=5, help="Backtest Period (Days)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial Capital ($)")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f" Backtest Simulation: {args.ticker}")
    print("="*60)
    print(f"[INFO] 실제 데이터 사용: yfinance API를 통해 실제 주가 데이터를 다운로드합니다.")
    print(f"[INFO] Look-ahead bias 방지: 각 거래일마다 해당 날짜 이전 데이터만 사용합니다.")
    print("="*60 + "\n")
    
    # Backtester 인스턴스 생성
    bt = Backtester(
        ticker=args.ticker,
        days=args.days,
        initial_capital=args.capital
    )
    
    try:
        # 1. 데이터 준비 & 에이전트 예측
        print("\n[1/4] 데이터 준비 및 에이전트 예측")
        print("-" * 60)
        bt.prepare_data()
        
        if bt.full_data is None or bt.full_data.empty:
            print("\n[ERROR] 데이터 준비에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        # 2. 모델 학습 (Meta Model)
        print("\n[2/4] Meta Model 학습")
        print("-" * 60)
        bt.train_model()
        
        if bt.model is None:
            print("\n[ERROR] 모델 학습에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        # 3. 시뮬레이션 실행
        print("\n[3/4] 백테스트 시뮬레이션 실행")
        print("-" * 60)
        bt.run_simulation(buy_threshold=0.005, sell_threshold=0.005)
        
        if bt.results is None or bt.results.empty:
            print("\n[ERROR] 시뮬레이션 실행에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        # 4. 결과 계산 및 저장
        print("\n[4/4] 결과 계산 및 저장")
        print("-" * 60)
        metrics = bt.calculate_metrics()
        bt.save_results()
        
        # 5. 리포트 출력
        print("\n" + "="*60)
        print(f"[Backtest Report: {args.ticker}]")
        print("="*60)
        print(f"백테스트 기간: {bt.start_date.date()} ~ {bt.end_date.date()} ({args.days}일)")
        print("-" * 60)
        print(f"초기 자본: ${metrics.get('Initial_Capital', 0):,.2f}")
        print(f"최종 자본: ${metrics.get('Final_Capital', 0):,.2f}")
        
        ret = metrics.get('Total_Return', 0)
        print(f"누적 수익률: {'+' if ret > 0 else ''}{ret:.2f}%")
        print(f"최대 낙폭(MDD): {metrics.get('MDD', 0):.2f}%")
        print(f"매매 횟수: {metrics.get('Trade_Count', 0)}회")
        if 'Avg_Daily_Return' in metrics:
            print(f"평균 일일 수익률: {metrics.get('Avg_Daily_Return', 0):.4f}%")
            print(f"변동성(표준편차): {metrics.get('Volatility', 0):.4f}%")
        print("-" * 60)
        print("상세 로그 및 차트가 저장되었습니다.")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] 백테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

