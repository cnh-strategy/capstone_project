
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.metrics import calculate_metrics, calculate_direction_accuracy, calculate_profitability

def analyze_results(csv_path: str, output_dir: str = "data/backtests/analysis"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # 1. 기본 지표 계산
    # Ensemble_Pred vs Actual_Close
    # NaN 제거
    df_valid = df.dropna(subset=['Actual_Close', 'Ensemble_Pred'])
    
    y_true = df_valid['Actual_Close'].values
    y_pred = df_valid['Ensemble_Pred'].values
    
    metrics = calculate_metrics(y_true, y_pred)
    print("\n[Ensemble Performance]")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 방향 정확도
    # 전일 종가(Actual_Close shifted) 필요
    prev_close = df_valid['Actual_Close'].shift(1).bfill().values
    # 하지만 여기선 df_valid가 연속되지 않을 수도 있으니, 원본 df에서 계산하는 게 나음
    # 다만 Actual_Close가 이미 있는 데이터만 의미 있음.
    
    dir_acc = calculate_direction_accuracy(y_true, y_pred, prev_close)
    print(f"Direction Accuracy: {dir_acc:.2f}%")
    
    # 2. 수익률 분석
    # 수익률 계산을 위해 전체 시계열 사용
    dates = df_valid['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    prof_res = calculate_profitability(dates, y_true, y_pred)
    print("\n[Profitability]")
    print(f"Strategy Return: {prof_res['Strategy_Return']:.2f}%")
    print(f"Buy & Hold Return: {prof_res['BuyHold_Return']:.2f}%")
    
    # 3. 시각화
    base_name = os.path.basename(csv_path).replace(".csv", "")
    
    # A. Price Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df_valid['Date'], df_valid['Actual_Close'], label='Actual Close', color='black')
    plt.plot(df_valid['Date'], df_valid['Ensemble_Pred'], label='Ensemble Pred', color='blue', linestyle='--')
    
    # Agent별 예측 추가 (있는 경우)
    colors = ['red', 'green', 'orange']
    for i, col in enumerate([c for c in df.columns if c.endswith('_Pred') and c != 'Ensemble_Pred']):
        if col in df_valid.columns:
            plt.plot(df_valid['Date'], df_valid[col], label=col, alpha=0.5, linestyle=':', color=colors[i%len(colors)])
            
    plt.title(f"Price Prediction: {base_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{base_name}_price.png"))
    print(f"Saved price chart to {output_dir}")
    
    # B. Cumulative Return
    # 간단히 매일의 수익률을 누적
    # Strategy: (Today Close / Prev Close) if Pred > Prev Close else 1.0
    # BuyHold: (Today Close / Prev Close)
    
    df_valid['Daily_Ret'] = df_valid['Actual_Close'].pct_change().fillna(0)
    # Prev Close (Align)
    # Shift Actual Close by 1
    df_valid['Prev_Close'] = df_valid['Actual_Close'].shift(1)
    
    # Strategy Signal: Pred > Prev_Close (Yesterday's Close, known today)
    # Rolling Backtest logic: Pred is for Today. Comparison is with Yesterday.
    df_valid['Signal'] = np.where(df_valid['Ensemble_Pred'] > df_valid['Prev_Close'], 1, 0)
    
    # Strategy Daily Return (Shift Signal to align? No.
    # We decide at T-1 (or T open) to hold for T based on Pred(T).
    # So if Pred(T) > Close(T-1), we get Return(T).
    # Signal is computed row-wise.
    df_valid['Strat_Daily_Ret'] = df_valid['Signal'] * df_valid['Daily_Ret']
    
    df_valid['Cum_BH'] = (1 + df_valid['Daily_Ret']).cumprod()
    df_valid['Cum_Strat'] = (1 + df_valid['Strat_Daily_Ret']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_valid['Date'], df_valid['Cum_BH'], label='Buy & Hold', color='gray')
    plt.plot(df_valid['Date'], df_valid['Cum_Strat'], label='Strategy', color='red')
    plt.title(f"Cumulative Return: {base_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{base_name}_return.png"))
    print(f"Saved return chart to {output_dir}")
    
    # C. Error Histogram
    errors = (df_valid['Ensemble_Pred'] - df_valid['Actual_Close']) / df_valid['Actual_Close'] * 100
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=30, color='purple', alpha=0.7)
    plt.title(f"Error Distribution (%) : {base_name}")
    plt.xlabel("Error %")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{base_name}_error.png"))
    print(f"Saved error hist to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Result Analyzer")
    parser.add_argument("csv_file", type=str, help="Path to rolling backtest csv result")
    args = parser.parse_args()
    
    analyze_results(args.csv_file)



