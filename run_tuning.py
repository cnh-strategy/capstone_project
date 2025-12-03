import os
import sys
import itertools
import pandas as pd
import copy
import glob
import random
from datetime import datetime
import traceback
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import config.agents as cfg_agents
from backtest import RollingBacktester


class TeeOutput:
    """터미널 출력과 파일 출력을 동시에 처리하는 클래스"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 즉시 파일에 쓰기
    
    def flush(self):
        for f in self.files:
            f.flush()


def calculate_metrics_from_csv(csv_path: str) -> dict:
    """
    CSV 파일에서 지표를 계산하는 헬퍼 함수
    
    Args:
        csv_path: 결과 CSV 파일 경로
        
    Returns:
        metrics 딕셔너리 (실패 시 None)
    """
    try:
        df_res = pd.read_csv(csv_path)
        df_res = df_res.dropna(subset=['Actual_Close', 'Ensemble_Pred']).reset_index(drop=True)
        
        if len(df_res) == 0:
            return None
        
        y_true = df_res['Actual_Close']
        y_pred = df_res['Ensemble_Pred']
        
        # 기본 지표
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # 방향 정확도: 첫 번째 행 제외 (prev_close가 NaN인 행 제외)
        prev_close = df_res['Actual_Close'].shift(1)
        mask = ~prev_close.isna()
        
        if mask.sum() == 0:
            return None
        
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]
        prev_close_valid = prev_close[mask]
        
        # 방향 정확도 계산
        dir_match = ((y_true_valid - prev_close_valid) * (y_pred_valid - prev_close_valid)) > 0
        dir_acc = dir_match.mean() * 100
        
        # 수익률 계산
        daily_ret = df_res['Actual_Close'].pct_change().fillna(0)[mask]
        signal = np.where(y_pred_valid > prev_close_valid, 1, 0)
        signal_series = pd.Series(signal, index=df_res[mask].index)
        
        strategy_ret = (daily_ret * signal_series.shift(1).fillna(0)).cumsum().iloc[-1] * 100
        bh_ret = daily_ret.cumsum().iloc[-1] * 100
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "direction_acc": dir_acc,
            "strategy_return": strategy_ret,
            "buy_hold_return": bh_ret
        }
        
        # 개별 에이전트 성능 계산
        pred_cols = [c for c in df_res.columns if c.endswith('_Pred') and c != 'Ensemble_Pred']
        for col in pred_cols:
            try:
                agent_name = col.replace('_Pred', '')
                y_agent = df_res[col][mask]
                agent_dir_match = ((y_true_valid - prev_close_valid) * (y_agent - prev_close_valid)) > 0
                metrics[f"{agent_name}_acc"] = agent_dir_match.mean() * 100
            except Exception:
                pass
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculating metrics from CSV: {e}")
        return None

def run_hyperparameter_tuning(
    ticker="TSLA",
    base_output_dir="backtest/tuning_results"
):
    # 1. 검증 기간 및 실험 횟수 설정
    PREDICT_DAYS = 30
    MAX_EVALS = 50  # 랜덤 서치 최대 시도 횟수
    
    # 로그 파일 설정
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(base_output_dir, f"tuning_log_{ticker}_{timestamp}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # stdout을 파일과 터미널 모두에 출력하도록 설정
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_file)
    
    try:
        print(f"🚀 Starting Random Search Tuning for {ticker}")
        print(f"📅 Fixed Predict Days: {PREDICT_DAYS}")
        print(f"🎲 Max Experiments: {MAX_EVALS}")
        print(f"📝 Log file: {log_file_path}")
        print(f"{'='*80}\n")
        
        # =================================================================
        # 2. 튜닝할 파라미터 그리드 (중요 파라미터 위주)
        # =================================================================
        # 규칙: "AgentName__ParameterName" (Double Underscore 구분)
        
        # Case A: 공통 파라미터
        common_grid = {
            "common__fine_tune_lr": [1e-4, 5e-4, 1e-3],
            # y_scale_factor는 100.0으로 고정 (불필요한 탐색 제외)
        }
        
        # Case B: TechnicalAgent (Window, Model Size, LR)
        tech_grid = {
            "TechnicalAgent__window_size": [10, 20, 30],
            "TechnicalAgent__rnn_units1": [32, 64, 128],
            "TechnicalAgent__learning_rate": [1e-4, 5e-4, 1e-3],
        }
        
        # Case C: SentimentalAgent (Window, Model Size, LR)
        # 주의: window_size 변경 시 데이터셋 재생성 로직이 내부적으로 처리되어야 함
        sent_grid = {
            "SentimentalAgent__window_size": [10, 20, 30],
            "SentimentalAgent__d_model": [32, 64, 128],
            "SentimentalAgent__learning_rate": [1e-4, 5e-4, 1e-3],
        }
        
        # Case D: MacroAgent (Window, LR)
        macro_grid = {
            "MacroAgent__window_size": [10, 20, 30],
            "MacroAgent__learning_rate": [1e-4, 5e-4, 1e-3],
        }
        
        # 모든 그리드 병합 (Cartesian Product)
        full_param_grid = {**common_grid, **tech_grid, **sent_grid, **macro_grid}
        
        keys, values = zip(*full_param_grid.items())
        # 모든 가능한 조합 생성
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"🔍 Total possible combinations: {len(combinations)}")
        
        # =================================================================
        # ★ 그리드 서치 (전체 조합 실행)
        # =================================================================
        # if len(combinations) > MAX_EVALS:
        #     print(f"⚡ Too many combinations. Randomly sampling {MAX_EVALS}...")
        #     random.seed(42)  # 결과 재현성을 위해 시드 고정
        #     combinations = random.sample(combinations, MAX_EVALS)
        
        results = []
        
        # 결과 저장 경로 미리 정의
        save_path = os.path.join(base_output_dir, f"tuning_summary_{ticker}_{timestamp}.csv")
        
        # 원본 설정 백업
        original_agents_info = copy.deepcopy(cfg_agents.agents_info)
        original_common_params = copy.deepcopy(cfg_agents.common_params)
        
        try:
            for idx, params in enumerate(combinations):
                print(f"\n{'='*80}")
                print(f"🧪 Experiment {idx+1}/{len(combinations)}")
                print(f"⚙️ Params: {params}")
                print(f"{'='*80}")
                
                # =================================================================
                # 3. Config 동적 적용
                # =================================================================
                for key, value in params.items():
                    if "__" not in key:
                        continue
                        
                    target, param_name = key.split("__", 1)
                    
                    # 3-1. 공통 파라미터 적용
                    if target == "common":
                        if param_name in cfg_agents.common_params:
                            cfg_agents.common_params[param_name] = value
                            
                    # 3-2. 에이전트별 파라미터 적용
                    elif target in cfg_agents.agents_info:
                        # config에 키가 없더라도 주입 (새로운 파라미터 실험 가능)
                        cfg_agents.agents_info[target][param_name] = value

                # =================================================================
                # 4. 클린업 (모델 및 전처리 데이터 삭제 - 파라미터 변경 반영 위해)
                # =================================================================
                print("🧹 Cleaning up models & processed data (keeping raw news)...")
                models_dir = os.path.join(project_root, "backtest", "models")
                data_dir = os.path.join(project_root, "backtest", "data", "processed")
                
                # 지워야 할 파일 패턴들
                cleanup_patterns = [
                    os.path.join(models_dir, f"{ticker}_*.pt"),
                    os.path.join(models_dir, "scalers", f"{ticker}_*.pkl"),
                    os.path.join(data_dir, f"{ticker}_*.csv"),
                    os.path.join(data_dir, f"{ticker}_*.pkl"),
                ]
                
                for pat in cleanup_patterns:
                    for f in glob.glob(pat):
                        try: os.remove(f)
                        except: pass

                # =================================================================
                # 5. 실행
                # =================================================================
                exp_id = f"exp_{idx+1}"
                exp_output_dir = os.path.join(base_output_dir, exp_id)
                
                try:
                    runner = RollingBacktester(
                        ticker=ticker,
                        start_date=None,
                        predict_days=PREDICT_DAYS,
                        rounds=params.get("rounds", 3), # 기본값 3
                        output_dir=exp_output_dir,
                        auto_analyze=True
                    )
                    
                    runner.prepare_data()
                    runner.run_loop()
                    
                    # 1차 시도: 내장 analyze 함수
                    metrics = None
                    try:
                        metrics = runner.analyze()
                        if metrics:
                            print(f"✅ runner.analyze() succeeded")
                    except Exception as e:
                        print(f"⚠️ runner.analyze() crashed: {e}")
                        metrics = None
                    
                    # 2차 시도: 백업 로직 (CSV에서 직접 계산)
                    if not metrics:
                        print(f"⚠️ Trying manual calculation from CSV for Exp {idx+1}...")
                        exp_output_dir_abs = os.path.abspath(exp_output_dir)
                        res_files = glob.glob(os.path.join(exp_output_dir_abs, "rolling_*.csv"))
                        
                        if res_files:
                            csv_path = sorted(res_files)[-1]  # 가장 최근 파일
                            print(f"[DEBUG] Using CSV: {os.path.basename(csv_path)}")
                            metrics = calculate_metrics_from_csv(csv_path)
                            
                            if metrics:
                                print(f"✅ Manual calculation successful")
                            else:
                                print("❌ Manual calculation returned None")
                        else:
                            print("❌ No CSV files found")

                    # ★ [DEBUG] metrics 상태 확인
                    print(f"[DEBUG] Exp {idx+1} - metrics type: {type(metrics)}, value: {metrics}")
                    
                    # ★ [수정] metrics가 있으면 저장, 없어도 일단 기록
                    if metrics:
                        # 결과 저장
                        row = {
                            "experiment_id": idx + 1,
                            **params,
                            "strategy_return": metrics.get("strategy_return", 0.0),
                            "buy_hold_return": metrics.get("buy_hold_return", 0.0),
                            "direction_acc": metrics.get("direction_acc", 0.0),
                            "mse": metrics.get("mse", 0.0),
                            "mae": metrics.get("mae", 0.0)
                        }
                        # ★ [추가] 개별 에이전트 점수도 포함
                        for k, v in metrics.items():
                            if k.endswith("_acc"):
                                row[k] = v
                        
                        results.append(row)
                        print(f"[DEBUG] Exp {idx+1} - Added to results. Total results count: {len(results)}")
                        print(f"✅ Exp {idx+1} Done! Return: {row['strategy_return']:.2f}%, DirAcc: {row['direction_acc']:.2f}%")
                    else:
                        # metrics가 없어도 일단 기록 (디버깅용)
                        row = {
                            "experiment_id": idx + 1,
                            **params,
                            "error": "No metrics available"
                        }
                        results.append(row)
                        print(f"[DEBUG] Exp {idx+1} - Added error row. Total results count: {len(results)}")
                        print(f"⚠️ Exp {idx+1} saved with error flag (no metrics).")
                    
                    # 실험마다 즉시 저장 (중간 결과 보존)
                    print(f"[DEBUG] Exp {idx+1} - Attempting to save summary. Results count: {len(results)}")
                    print(f"[DEBUG] Exp {idx+1} - Save path: {save_path}")
                    print(f"[DEBUG] Exp {idx+1} - Save path exists (dir): {os.path.exists(os.path.dirname(save_path))}")
                    
                    if len(results) > 0:
                        try:
                            print(f"[DEBUG] Exp {idx+1} - Creating DataFrame from {len(results)} results...")
                            current_df = pd.DataFrame(results)
                            print(f"[DEBUG] Exp {idx+1} - DataFrame created. Shape: {current_df.shape}, Columns: {list(current_df.columns)[:5]}...")
                            
                            if "strategy_return" in current_df.columns:
                                current_df = current_df.sort_values(by=["strategy_return", "direction_acc"], ascending=[False, False])
                                print(f"[DEBUG] Exp {idx+1} - DataFrame sorted by strategy_return")
                            
                            print(f"[DEBUG] Exp {idx+1} - Writing to CSV: {save_path}")
                            current_df.to_csv(save_path, index=False)
                            
                            # 저장 확인
                            if os.path.exists(save_path):
                                file_size = os.path.getsize(save_path)
                                print(f"[DEBUG] Exp {idx+1} - File saved successfully! Size: {file_size} bytes")
                            else:
                                print(f"[DEBUG] Exp {idx+1} - ⚠️ WARNING: File was not created!")
                            
                            print(f"💾 Summary updated: {save_path} ({len(results)} experiments)")
                        except Exception as e:
                            print(f"[DEBUG] Exp {idx+1} - ❌ Exception during save: {type(e).__name__}: {e}")
                            traceback.print_exc()
                            print(f"❌ Failed to save summary: {e}")
                    else:
                        print(f"[DEBUG] Exp {idx+1} - ⚠️ No results to save (results list is empty)")
                        
                except Exception as e:
                    print(f"❌ Exp {idx+1} Failed: {e}")
                    traceback.print_exc()
        
        finally:
            # 설정 복구
            cfg_agents.agents_info = original_agents_info
            cfg_agents.common_params = original_common_params
            
            print(f"\n{'='*80}")
            print("🏁 Tuning Completed")
            print(f"{'='*80}")
            
            # 최종 요약 출력 (이미 저장된 파일이 있으므로 중복 저장하지 않음)
            if results:
                df = pd.DataFrame(results)
                if "strategy_return" in df.columns:
                    df = df.sort_values(by=["strategy_return", "direction_acc"], ascending=[False, False])
                
                print(f"\n📊 Final Summary: {len(results)} experiments completed")
                print(f"📄 Results file: {save_path}")
                print("\n🏆 Top 3 Configurations:")
                
                # 동적으로 출력할 컬럼 선택
                cols_to_show = ["experiment_id"] + list(full_param_grid.keys()) + ["strategy_return", "direction_acc"]
                cols_to_show = [c for c in cols_to_show if c in df.columns]
                
                print(df[cols_to_show].head(3).to_string())
                
                # 1위 결과 반환
                best_row = df.iloc[0].to_dict()
                best_row["ticker"] = ticker
                return best_row
            else:
                print("❌ No results collected.")
                return None
    
    finally:
        # stdout 복구 및 로그 파일 닫기
        sys.stdout = original_stdout
        if not log_file.closed:
            log_file.close()
        print(f"\n📝 Log saved to: {log_file_path}")

if __name__ == "__main__":
    # 튜닝할 티커 리스트 (우선 MSFT 하나만 테스트하거나 필요시 추가)
    target_tickers = ["MSFT", "AAPL", "NVDA"]
    
    # 전체 종목 베스트 설정 수집용
    all_best_configs = []
    
    for ticker in target_tickers:
        print(f"\n{'#'*60}")
        print(f"🚀 Launching Tuning Sequence for: {ticker}")
        print(f"{'#'*60}\n")
        
        try:
            # 티커별로 결과 폴더 분리
            ticker_output_dir = os.path.join("backtest", "tuning_results", ticker)
            
            best_result = run_hyperparameter_tuning(ticker=ticker, base_output_dir=ticker_output_dir)
            
            if best_result:
                all_best_configs.append(best_result)
            
        except Exception as e:
            print(f"❌ Critical Error during {ticker} tuning: {e}")
            traceback.print_exc()
            continue

    # 모든 튜닝 완료 후 종합 리포트 출력
    if all_best_configs:
        print(f"\n\n{'='*80}")
        print("🌟 FINAL SUMMARY: Best Configuration per Ticker")
        print(f"{'='*80}")
        
        summary_df = pd.DataFrame(all_best_configs)
        
        base_cols = ["ticker", "strategy_return", "direction_acc", "experiment_id"]
        param_cols = [c for c in summary_df.columns if c not in base_cols]
        final_cols = base_cols + param_cols
        
        print(summary_df[final_cols].to_string(index=False))
        
        total_summary_path = os.path.join("backtest", "tuning_results", f"total_best_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary_df[final_cols].to_csv(total_summary_path, index=False)
        print(f"\n📄 Total summary saved to: {total_summary_path}")
    else:
        print("\n❌ No successful tuning results collected.")
