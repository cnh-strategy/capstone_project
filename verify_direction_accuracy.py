import pandas as pd
import numpy as np
import os

def calculate_direction_accuracy(df):
    """Actual_Close와 TechnicalAgent_Pred의 방향 정확도 계산
    
    tech_tuning.py 방식:
    - Current_Close 컬럼이 있으면: prev_close = Current_Close (예측 시점 종가)
    - Current_Close 컬럼이 없으면: prev_close = Actual_Close.shift(1) (run_tuning.py 방식)
    - dir_match = ((actual - prev) * (pred - prev)) > 0
    """
    if len(df) < 2:
        return None
    
    df_res = df.dropna(subset=['Actual_Close', 'TechnicalAgent_Pred']).reset_index(drop=True)
    
    if len(df_res) == 0:
        return None
    
    y_true = df_res['Actual_Close']
    y_pred = df_res['TechnicalAgent_Pred']
    
    # Current_Close 컬럼이 있으면 tech_tuning.py 방식, 없으면 run_tuning.py 방식
    if 'Current_Close' in df_res.columns:
        # tech_tuning.py 방식: Current_Close를 기준으로 계산
        prev_close = df_res['Current_Close']
        mask = ~prev_close.isna()
    else:
        # run_tuning.py 방식: 이전 행의 Actual_Close를 기준으로 계산
        prev_close = y_true.shift(1)
        mask = ~prev_close.isna()
    
    if mask.sum() == 0:
        return None
    
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    prev_close_valid = prev_close[mask]
    
    # 방향 정확도 계산 (동일한 공식)
    dir_match = ((y_true_valid - prev_close_valid) * (y_pred_valid - prev_close_valid)) > 0
    dir_acc = dir_match.mean() * 100
    
    correct = dir_match.sum()
    total = len(dir_match)
    
    # 상세 분석용
    actual_directions = np.sign(y_true_valid - prev_close_valid).tolist()
    pred_directions = np.sign(y_pred_valid - prev_close_valid).tolist()
    
    return {
        'correct': correct,
        'total': total,
        'accuracy': dir_acc,
        'actual_directions': actual_directions,
        'pred_directions': pred_directions
    }

# Summary 파일 읽기 (가장 최근 파일 찾기)
import glob
summary_dir = '/home/ubuntu/Projects/ml-ai/capstone/backtest/tech_tuning_results/MSFT'
summary_files = glob.glob(os.path.join(summary_dir, 'tech_tuning_summary_*.csv'))
if not summary_files:
    print("❌ Summary 파일을 찾을 수 없습니다.")
    exit(1)
summary_file = sorted(summary_files)[-1]  # 가장 최근 파일
print(f"📄 사용할 Summary 파일: {os.path.basename(summary_file)}")
summary_df = pd.read_csv(summary_file)

print("=" * 80)
print("방향 정확도 검증 결과")
print("=" * 80)
print()

# 각 실험에 대해 검증
for idx, row in summary_df.iterrows():
    if pd.isna(row['experiment_id']):
        continue
    
    exp_id = int(row['experiment_id'])
    reported_accuracy = row['direction_accuracy']
    n_samples = int(row['n_samples'])
    
    # Rolling 파일 읽기
    rolling_file = f'/home/ubuntu/Projects/ml-ai/capstone/backtest/tech_tuning_results/MSFT/exp_{exp_id}/rolling_MSFT_2025-10-22_2025-12-01.csv'
    
    if not os.path.exists(rolling_file):
        print(f"실험 {exp_id}: 파일을 찾을 수 없습니다 - {rolling_file}")
        continue
    
    df = pd.read_csv(rolling_file)
    result = calculate_direction_accuracy(df)
    
    if result:
        calculated_accuracy = result['accuracy']
        calculated_total = result['total']
        correct_count = result['correct']
        
        print(f"실험 {exp_id}:")
        print(f"  Summary 보고된 정확도: {reported_accuracy:.2f}%")
        print(f"  실제 계산된 정확도: {calculated_accuracy:.2f}%")
        print(f"  샘플 수 (Summary): {n_samples}")
        print(f"  샘플 수 (실제 계산): {calculated_total}")
        print(f"  정확히 일치: {correct_count} / {calculated_total}")
        print(f"  차이: {abs(reported_accuracy - calculated_accuracy):.4f}%")
        
        if abs(reported_accuracy - calculated_accuracy) < 0.01:
            print(f"  ✅ 정확도가 일치합니다!")
        else:
            print(f"  ❌ 정확도가 일치하지 않습니다!")
        
        # 상세 분석
        print(f"  상세 분석:")
        print(f"    - 실제 상승: {result['actual_directions'].count(1)}회")
        print(f"    - 실제 하락: {result['actual_directions'].count(0)}회")
        print(f"    - 예측 상승: {result['pred_directions'].count(1)}회")
        print(f"    - 예측 하락: {result['pred_directions'].count(0)}회")
        print()
    else:
        print(f"실험 {exp_id}: 데이터가 부족하여 계산할 수 없습니다.")
        print()

print("=" * 80)
