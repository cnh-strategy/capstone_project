import os
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def manual_summary_generator(ticker="MSFT"):
    base_dir = f"backtest/tuning_results/{ticker}"
    output_file = f"{base_dir}/tuning_summary_{ticker}_manual.csv"
    
    print(f"🔍 Searching for results in: {base_dir}")
    
    exp_dirs = glob.glob(os.path.join(base_dir, "exp_*"))
    print(f"📂 Found {len(exp_dirs)} experiment directories.")
    
    results = []
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        
        csv_files = glob.glob(os.path.join(exp_dir, "rolling_*.csv"))
        if not csv_files:
            continue
            
        # 모든 CSV 파일을 읽어서 합치기 (여러 파일이 있을 경우)
        all_dfs = []
        for csv_path in sorted(csv_files):
            try:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
            except:
                pass
        
        if not all_dfs:
            continue
            
        # 모든 데이터 합치기
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
        print(f"[DEBUG] {exp_name}: Combined {len(csv_files)} CSV files, {len(df)} total rows")
        
        try:
            df = pd.read_csv(csv_path)
            df_valid = df.dropna(subset=['Actual_Close', 'Ensemble_Pred']).copy()
            
            if len(df_valid) == 0:
                print(f"[DEBUG] {exp_name}: No valid rows after dropna")
                continue
                
            print(f"[DEBUG] {exp_name}: {len(df_valid)} valid rows")
                
            # 공통 변수
            y_true = df_valid['Actual_Close']
            prev_close = df_valid['Actual_Close'].shift(1)
            daily_ret = df_valid['Actual_Close'].pct_change().fillna(0)
            
            # ★ [수정] 첫 번째 행은 prev_close가 없으므로 제외하고 계산
            # prev_close가 NaN인 행을 제거
            mask = ~prev_close.isna()
            print(f"[DEBUG] {exp_name}: Valid rows (after mask): {mask.sum()}")
            
            if mask.sum() == 0:
                # prev_close가 모두 NaN이면 계산 불가
                print(f"[DEBUG] {exp_name}: No valid rows for direction calculation")
                ens_acc = 0.0
            else:
                y_true_valid = y_true[mask]
                y_pred_valid = df_valid['Ensemble_Pred'][mask]
                prev_close_valid = prev_close[mask]
                
                # 기본 결과 (Ensemble) - 첫 번째 행 제외
                product = (y_true_valid - prev_close_valid) * (y_pred_valid - prev_close_valid)
                ens_acc = (product > 0).mean() * 100
                print(f"[DEBUG] {exp_name}: Calculated acc = {ens_acc:.2f}%")
            
            # Strategy Return (Ensemble 기준) - 첫 번째 행 제외
            if mask.sum() > 0:
                ens_signal = np.where(df_valid['Ensemble_Pred'][mask] > prev_close[mask], 1, 0)
                ens_ret = (daily_ret[mask] * pd.Series(ens_signal, index=df_valid[mask].index).shift(1).fillna(0)).cumsum().iloc[-1] * 100
            else:
                ens_ret = 0.0
            
            row = {
                "experiment_id": exp_name,
                "Ensemble_Acc": ens_acc,
                "Ensemble_Ret": ens_ret,
            }
            
            # ★ [추가] 개별 에이전트 성능 계산
            # 컬럼명 패턴: "*_Pred" 로 끝나는 모든 컬럼 찾기
            pred_cols = [c for c in df_valid.columns if c.endswith('_Pred') and c != 'Ensemble_Pred']
            
            for col in pred_cols:
                agent_name = col.replace('_Pred', '') # 예: TechnicalAgent_R0
                y_agent = df_valid[col]
                
                # Direction Accuracy 계산 - 첫 번째 행 제외
                if mask.sum() > 0:
                    y_agent_valid = y_agent[mask]
                    agent_acc = (((y_true_valid - prev_close_valid) * (y_agent_valid - prev_close_valid)) > 0).mean() * 100
                else:
                    agent_acc = 0.0
                row[f"{agent_name}_Acc"] = agent_acc
            
            row["rows_count"] = len(df_valid)
            results.append(row)
            
            print(f"✅ {exp_name}: Ens_Acc={ens_acc:.2f}%")
            
        except Exception as e:
            print(f"❌ Error {exp_name}: {e}")
    
    if results:
        df_summary = pd.DataFrame(results)
        
        # 컬럼 순서: ID -> Ensemble -> Agents... -> Rows
        base_cols = ["experiment_id", "Ensemble_Acc", "Ensemble_Ret"]
        agent_cols = sorted([c for c in df_summary.columns if c not in base_cols and c != "rows_count"])
        final_cols = base_cols + agent_cols + ["rows_count"]
        
        df_summary = df_summary[final_cols].sort_values(by=["Ensemble_Ret", "Ensemble_Acc"], ascending=[False, False])
        
        df_summary.to_csv(output_file, index=False)
        print(f"\n🎉 Summary saved to: {output_file}")
        print(df_summary.to_string(index=False))
    else:
        print("\n❌ No valid results found.")

if __name__ == "__main__":
    manual_summary_generator("MSFT")
