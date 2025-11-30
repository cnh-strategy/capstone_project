import os
import sys
import itertools
import pandas as pd
import copy
import glob
from datetime import datetime
import traceback

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import config.agents as cfg_agents
from backtest import RollingBacktester

def flatten_dict(d, parent_key='', sep='_'):
    """중첩된 딕셔너리를 평탄화합니다 (결과 저장용)."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def run_hyperparameter_tuning(
    ticker="NVDL",
    base_output_dir="backtest/tuning_results"
):
    # 1. 검증 기간 고정
    PREDICT_DAYS = 10
    
    print(f"🚀 Starting Extended Hyperparameter Tuning for {ticker}")
    print(f"📅 Fixed Predict Days: {PREDICT_DAYS}")
    
    # =================================================================
    # 2. 튜닝할 파라미터 그리드 정의 (확장됨)
    # =================================================================
    # 공통 파라미터와 각 에이전트별 파라미터를 분리하여 정의
    
    # Case A: 공통 파라미터 (전체 영향)
    common_grid = {
        "rounds": [3],                 # 토론 라운드 (보통 3이면 충분)
        "common_n_samples": [30],      # MC Dropout 샘플 수
    }
    
    # Case B: TechnicalAgent 전용
    tech_grid = {
        "tech_window_size": [20, 40],
        "tech_dropout": [0.2, 0.3],
    }
    
    # Case C: SentimentalAgent 전용
    # (주의: window_size 변경 시 데이터셋 재생성 필요)
    sent_grid = {
        "sent_window_size": [20, 40],
        # "sent_dropout": [0.2] # 고정하고 싶으면 이렇게 하나만
    }
    
    # 모든 그리드 병합 (Cartesian Product)
    # 딕셔너리들을 하나로 합쳐서 itertools.product에 넘깁니다.
    full_param_grid = {**common_grid, **tech_grid, **sent_grid}
    
    keys, values = zip(*full_param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔍 Total combinations to test: {len(combinations)}")
    
    results = []
    
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
            # 3. Config 동적 적용 (정교화)
            # =================================================================
            
            # 3-1. 공통 파라미터 적용
            if "common_n_samples" in params:
                cfg_agents.common_params["n_samples"] = params["common_n_samples"]
                
            # 3-2. TechnicalAgent 적용
            tech_cfg = cfg_agents.agents_info["TechnicalAgent"]
            if "tech_window_size" in params:
                tech_cfg["window_size"] = params["tech_window_size"]
            if "tech_dropout" in params:
                tech_cfg["dropout"] = params["tech_dropout"]
                
            # 3-3. SentimentalAgent 적용
            sent_cfg = cfg_agents.agents_info["SentimentalAgent"]
            if "sent_window_size" in params:
                sent_cfg["window_size"] = params["sent_window_size"]
            if "sent_dropout" in params:
                sent_cfg["dropout"] = params["sent_dropout"]

            # =================================================================
            # 4. 클린업 (뉴스 데이터 보존)
            # =================================================================
            print("🧹 Cleaning up models & processed data (keeping raw news)...")
            models_dir = os.path.join(project_root, "backtest", "models")
            data_dir = os.path.join(project_root, "backtest", "data", "processed")
            
            # 지워야 할 파일 패턴들
            cleanup_patterns = [
                os.path.join(models_dir, f"{ticker}_*.pt"),
                os.path.join(models_dir, "scalers", f"{ticker}_*.pkl"),
                os.path.join(data_dir, f"{ticker}_*.csv"),  # 전처리된 데이터 (window_size 변경 반영 위해 삭제)
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
                    start_date=None,  # 자동 계산을 위해 None 전달
                    predict_days=PREDICT_DAYS,
                    rounds=params.get("rounds", 3),
                    output_dir=exp_output_dir,
                    auto_analyze=True
                )
                
                runner.prepare_data()
                runner.run_loop()
                metrics = runner.analyze()
                
                if metrics:
                    # 결과 저장 (파라미터 + 메트릭)
                    # 방향 정확도(direction_acc)가 핵심 지표 중 하나
                    row = {
                        "experiment_id": idx + 1,
                        **params,  # 실험 파라미터
                        "strategy_return": metrics.get("strategy_return", 0.0),
                        "buy_hold_return": metrics.get("buy_hold_return", 0.0),
                        "direction_acc": metrics.get("direction_acc", 0.0),
                        "mse": metrics.get("mse", 0.0),
                        "mae": metrics.get("mae", 0.0)
                    }
                    results.append(row)
                    print(f"✅ Exp {idx+1} Done! Return: {row['strategy_return']:.2f}%, DirAcc: {row['direction_acc']:.2f}%")
                else:
                    print(f"⚠️ Exp {idx+1} Finished but no metrics returned.")
                    
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
        
        if results:
            df = pd.DataFrame(results)
            # 정렬 기준: 수익률 우선, 그 다음 방향 정확도
            df = df.sort_values(by=["strategy_return", "direction_acc"], ascending=[False, False])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(base_output_dir, f"tuning_summary_{ticker}_{timestamp}.csv")
            os.makedirs(base_output_dir, exist_ok=True)
            df.to_csv(save_path, index=False)
            
            print(f"📄 Results saved to: {save_path}")
            print("\n🏆 Top 3 Configurations:")
            # 주요 컬럼만 출력
            cols_to_show = ["experiment_id", "strategy_return", "direction_acc"] + list(full_param_grid.keys())
            print(df[cols_to_show].head(3).to_string())
        else:
            print("❌ No results.")

if __name__ == "__main__":
    # NVDL 종목에 대해 수행
    run_hyperparameter_tuning(ticker="NVDL")
