# macro_tuning.py
# MacroAgent 전용 파라미터 튜닝 스크립트
# predict_days만큼 롤링 예측을 진행하고 방향 예측도를 최적화합니다.

import os
import sys
import itertools
import pandas as pd
import copy
import glob
import random
from datetime import datetime, timedelta
import traceback
import numpy as np
import torch
import yfinance as yf

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import config.agents as cfg_agents
from config.agents import common_params
from agents.macro_agent import MacroAgent
from core.data_set import build_dataset


class TeeOutput:
    """터미널 출력과 파일 출력을 동시에 처리하는 클래스"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def _parse_period(period: str) -> int:
    """'2y', '1y', '6mo' 등의 기간 문자열을 일수(int)로 변환 (run_tuning.py와 동일)"""
    period = period.lower()
    if period.endswith("y"):
        return int(period[:-1]) * 365
    elif period.endswith("mo"):
        return int(period[:-2]) * 30
    elif period.endswith("d"):
        return int(period[:-1])
    else:
        print(f"[WARN] Unknown period format '{period}', defaulting to 730 days")
        return 730


def _generate_trading_days(start_dt: datetime, count: int) -> list:
    """주말을 제외한 평일 리스트 생성"""
    days = []
    current = start_dt
    while len(days) < count:
        if current.weekday() < 5:  # 0=Monday, ..., 4=Friday
            days.append(current)
        current += timedelta(days=1)
    return days


def _prepare_filtered_dataset(sim_date: str, ticker: str, raw_dir: str):
    """
    시뮬레이션 날짜 이전 데이터만 포함하는 필터링된 데이터셋 생성
    backtest_temp 디렉토리에 저장 (backtest.py 방식과 동일)
    """
    sim_date_dt = pd.to_datetime(sim_date)
    original_path = os.path.join(raw_dir, f"{ticker}_MacroAgent_raw.csv")
    
    if not os.path.exists(original_path):
        return False
    
    try:
        df = pd.read_csv(original_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        else:
            return False
        
        df = df.sort_values("Date").reset_index(drop=True)
        
        # 미래 데이터 필터링
        df_filtered = df[df["Date"] < sim_date_dt].copy()
        
        if len(df_filtered) > 0:
            # backtest_temp 디렉토리에 저장 (backtest.py 방식)
            temp_dir = os.path.join(raw_dir, "backtest_temp")
            os.makedirs(temp_dir, exist_ok=True)
            date_str = sim_date.replace("-", "")
            temp_path = os.path.join(temp_dir, f"{ticker}_MacroAgent_raw_{date_str}.csv")
            df_filtered.to_csv(temp_path, index=False)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  [WARN] 데이터셋 필터링 실패: {e}")
        return False


def calculate_direction_accuracy_rolling(
    agent: MacroAgent,
    ticker: str,
    data_dir: str,
    raw_dir: str,
    predict_days: int,
    start_date: str = None,
    output_dir: str = None
):
    """
    predict_days만큼 롤링 예측을 수행하고 방향 예측도를 계산합니다.
    각 예측마다 모델을 재학습합니다.
    
    실행 흐름:
    1. predict_days만큼 거래일 리스트 생성
    2. 각 날짜마다:
       a. 백테스팅 모드 설정 (test_mode, simulation_date)
       b. 해당 날짜 이전 데이터만 필터링하여 backtest_temp에 저장
       c. 기존 모델/스케일러 삭제
       d. 필터링된 데이터로 데이터셋 재생성
       e. 모델 재학습 (pretrain)
       f. 예측 수행 (searcher -> predict)
       g. 실제 종가와 비교하여 방향 예측도 계산
    3. 전체 결과의 방향 예측도 평균 계산
    
    Args:
        agent: MacroAgent 인스턴스
        ticker: 종목 코드
        data_dir: 데이터 디렉토리
        raw_dir: 원본 데이터 디렉토리
        predict_days: 예측할 거래일 수
        start_date: 시작 날짜 (YYYY-MM-DD, None이면 자동 계산)
        output_dir: 출력 디렉토리
    
    Returns:
        dict: 방향 예측도 및 기타 지표
    """
    try:
        # 예측 날짜 리스트 생성
        if start_date is None:
            # 오늘로부터 predict_days 거래일 전으로 자동 설정
            today = datetime.today()
            lookback = max(30, predict_days * 2)
            temp_start = today - timedelta(days=lookback)
            candidates = _generate_trading_days(temp_start, lookback)
            candidates = [d for d in candidates if d < today]
            
            if len(candidates) < predict_days:
                start_dt = today - timedelta(days=predict_days + 2)
            else:
                start_dt = candidates[-predict_days]
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        predict_dates = _generate_trading_days(start_dt, predict_days)
        
        if len(predict_dates) == 0:
            print(f"  ❌ 예측할 거래일을 찾을 수 없습니다.")
            return None
        
        print(f"  📅 예측 기간: {predict_dates[0].strftime('%Y-%m-%d')} ~ {predict_dates[-1].strftime('%Y-%m-%d')} ({len(predict_dates)}일)")
        
        results = []
        
        # 각 날짜마다 롤링 예측 수행
        for day_idx, sim_dt in enumerate(predict_dates, start=1):
            sim_date = sim_dt.strftime("%Y-%m-%d")
            print(f"  [{day_idx}/{len(predict_dates)}] {sim_date} 예측 중...")
            
            try:
                # ============================================================
                # 각 날짜별 예측 프로세스 (run_tuning.py의 _run_backtest_without_llm과 동일한 순서)
                # ============================================================
                # 1. 시점별 데이터셋 필터링 (Data Leakage 방지) - run_tuning.py와 동일한 순서
                _prepare_filtered_dataset(sim_date, ticker, raw_dir)
                
                # 2. 백테스팅 모드 설정 (run_tuning.py와 동일한 순서)
                if hasattr(agent, 'set_test_mode'):
                    agent.set_test_mode(True)
                if hasattr(agent, 'set_simulation_date'):
                    agent.set_simulation_date(sim_date)
                # train_start는 period 설정 기반으로 계산 (run_tuning.py와 동일)
                period_str = common_params.get("period", "2y")
                train_days = _parse_period(period_str)
                train_start_dt = sim_dt - timedelta(days=train_days)
                train_start = train_start_dt.strftime("%Y-%m-%d")
                if hasattr(agent, 'set_training_window'):
                    agent.set_training_window(train_start)
                
                # 3. 데이터 준비 및 학습 (run_tuning.py의 _run_backtest_without_llm과 동일한 순서)
                # 3-1. searcher 먼저 실행 (run_tuning.py line 127와 동일)
                X_tensor = agent.searcher(ticker, rebuild=False)
                if X_tensor is None or X_tensor.shape[0] == 0:
                    print(f"    ⚠️ 데이터 없음, 스킵")
                    continue
                
                # 3-2. pretrain 실행 (run_tuning.py line 131와 동일, force_pretrain=True이므로 항상 실행)
                # 모델 삭제는 예측 후에 수행 (run_tuning.py의 _cleanup_backtest_models와 동일)
                agent.pretrain()
                
                # 4. 예측 수행 (run_tuning.py line 142와 동일)
                target = agent.predict(X_tensor)
                pred_close = target.next_close
                
                # 현재 종가 가져오기 (stockdata에서 가져오거나 yfinance 사용)
                current_close = None
                if hasattr(agent, 'stockdata') and agent.stockdata and hasattr(agent.stockdata, 'last_price'):
                    current_close = agent.stockdata.last_price
                
                if current_close is None:
                    try:
                        df_price = yf.download(ticker, start=sim_date, end=(sim_dt + timedelta(days=2)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
                        if not df_price.empty:
                            val = df_price["Close"].iloc[0]
                            current_close = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
                    except Exception as e:
                        print(f"    ⚠️ 종가 조회 실패: {e}")
                        continue
                
                if current_close is None:
                    print(f"    ⚠️ 종가를 가져올 수 없음")
                    continue
                
                # 실제 종가 가져오기 (backtest.py와 동일한 방식)
                # sim_date 당일의 실제 종가를 가져옴
                actual_close = np.nan
                try:
                    # backtest.py와 동일한 방식: date 당일의 종가를 가져옴
                    next_day = (sim_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                    df_price = yf.download(ticker, start=sim_date, end=next_day, progress=False, auto_adjust=False)
                    if not df_price.empty:
                        val = df_price["Close"].iloc[0]
                        actual_close = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
                except Exception as e:
                    print(f"    ⚠️ 실제 종가 조회 실패: {e}")
                    pass
                
                if np.isnan(actual_close):
                    print(f"    ⚠️ 실제 종가를 가져올 수 없음 (주말/휴장일 가능)")
                    # 주말이거나 휴장일일 수 있으므로 스킵
                    continue
                
                results.append({
                    "date": sim_date,
                    "pred_close": pred_close,
                    "actual_close": actual_close,
                    "current_close": current_close
                })
                
                print(f"    ✅ 예측: {pred_close:.2f}, 실제: {actual_close:.2f}")
                
                # 5. 리소스 정리 (run_tuning.py의 _cleanup_backtest_models와 동일한 순서)
                # 5-1. 모델 및 스케일러 삭제 (예측 후에 수행, run_tuning.py line 313와 동일)
                model_path = os.path.join(agent.model_dir, f"{ticker}_{agent.agent_id}.pt")
                if os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                    except:
                        pass
                
                scaler_patterns = [
                    os.path.join(agent.model_dir, "scalers", f"{ticker}_{agent.agent_id}_*.pkl"),
                ]
                for pat in scaler_patterns:
                    for f in glob.glob(pat):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                # 5-2. 필터링된 임시 데이터셋 삭제 (run_tuning.py의 _cleanup_filtered_datasets와 동일)
                temp_dir = os.path.join(raw_dir, "backtest_temp")
                date_str = sim_date.replace("-", "")
                temp_path = os.path.join(temp_dir, f"{ticker}_MacroAgent_raw_{date_str}.csv")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
            except Exception as e:
                print(f"    ❌ {sim_date} 예측 실패: {e}")
                import traceback as tb
                tb.print_exc()
                continue
        
        if len(results) == 0:
            return None
        
        # 방향 예측도 계산
        df_results = pd.DataFrame(results)
        df_results = df_results.dropna()
        
        if len(df_results) == 0:
            return None
        
        # 롤링 예측 결과 CSV 저장 (run_tuning.py와 동일한 방식)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if len(df_results) > 0:
                start_date_str = df_results["date"].iloc[0]
                end_date_str = df_results["date"].iloc[-1]
                csv_filename = f"rolling_{ticker}_{start_date_str}_{end_date_str}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                
                # CSV 형식 맞추기 (run_tuning.py와 동일)
                output_df = pd.DataFrame({
                    "Date": df_results["date"],
                    "Ticker": ticker,
                    "Actual_Close": df_results["actual_close"],
                    "MacroAgent_Pred": df_results["pred_close"],
                })
                output_df.to_csv(csv_path, index=False)
                print(f"  💾 롤링 예측 결과 저장: {csv_path}")
        
        # 이전 종가 대비 방향 계산 (run_tuning.py와 동일한 방식)
        # prev_close = Actual_Close.shift(1)
        prev_close = df_results["actual_close"].shift(1)
        mask = ~prev_close.isna()
        
        if mask.sum() == 0:
            return None
        
        y_true_valid = df_results["actual_close"][mask]
        y_pred_valid = df_results["pred_close"][mask]
        prev_close_valid = prev_close[mask]
        
        # 방향 정확도 계산 (run_tuning.py와 동일)
        dir_match = ((y_true_valid - prev_close_valid) * (y_pred_valid - prev_close_valid)) > 0
        direction_accuracy = float(dir_match.mean() * 100.0)
        
        # 추가 지표 계산을 위한 값들
        pred_close = df_results["pred_close"].values
        actual_close = df_results["actual_close"].values
        
        # 추가 지표
        mae = np.mean(np.abs(pred_close - actual_close))
        rmse = np.sqrt(np.mean((pred_close - actual_close) ** 2))
        
        if np.std(pred_close) == 0 or np.std(actual_close) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(pred_close, actual_close)[0, 1])
        
        return {
            "direction_accuracy": direction_accuracy,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "n_samples": len(results)
        }
        
    except Exception as e:
        print(f"  ❌ 롤링 예측 실패: {e}")
        traceback.print_exc()
        return None


def run_macro_tuning(
    ticker="MSFT",
    start_date=None,
    predict_days=30,
    period="2y",
    base_output_dir="backtest/macro_tuning_results",
    max_evals=50
):
    """
    MacroAgent 파라미터 튜닝 실행
    
    Args:
        ticker: 종목 코드
        start_date: 예측 시작 날짜 (YYYY-MM-DD, None이면 자동 계산)
        predict_days: 예측할 거래일 수
        period: yfinance 데이터 수집 기간
        base_output_dir: 결과 저장 디렉토리
        max_evals: 최대 실험 횟수
    """
    # 로그 파일 설정
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(base_output_dir, f"macro_tuning_log_{ticker}_{timestamp}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # stdout을 파일과 터미널 모두에 출력하도록 설정
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_file)
    
    try:
        print(f"🚀 MacroAgent 파라미터 튜닝 시작: {ticker}")
        print(f"📅 예측 일수: {predict_days}일")
        if start_date:
            print(f"📅 시작 날짜: {start_date}")
        else:
            print(f"📅 시작 날짜: 자동 계산")
        print(f"🎲 최대 실험 횟수: {max_evals}")
        print(f"📝 로그 파일: {log_file_path}")
        print(f"{'='*80}\n")
        
        # =================================================================
        # 1. 디렉토리 설정
        # =================================================================
        data_dir = os.path.join(project_root, "backtest", "data", "processed")
        raw_dir = os.path.join(project_root, "backtest", "data", "raw")
        models_dir = os.path.join(project_root, "backtest", "models")
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, "scalers"), exist_ok=True)
        
        # =================================================================
        # 2. 초기 데이터셋 생성 (전체 기간)
        # =================================================================
        print("📊 초기 데이터셋 생성 중...")
        try:
            build_dataset(
                ticker=ticker,
                save_dir=data_dir,
                agent_id="MacroAgent",
                period=period,
                interval="1d"
            )
            print(f"✅ 데이터셋 생성 완료: {data_dir}\n")
        except Exception as e:
            print(f"❌ 데이터셋 생성 실패: {e}")
            traceback.print_exc()
            return None
        
        # =================================================================
        # 3. 튜닝할 파라미터 그리드 정의
        # =================================================================
        param_grid = {
            "window_size": [20, 30, 40],
            "hidden_dims": [[64, 32, 16], [128, 64, 32], [256, 128, 64]],  # 리스트 형태
            "dropout_rates": [[0.2, 0.2, 0.1], [0.3, 0.3, 0.2], [0.4, 0.4, 0.3]],  # 리스트 형태
            "learning_rate": [1e-4, 5e-4, 1e-3],
            "batch_size": [16, 32, 64],
            "epochs": [20, 30, 40],  # 튜닝 속도 향상을 위해 감소
        }
        
        # 모든 조합 생성
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"🔍 총 가능한 조합: {len(combinations)}")
        
        # 랜덤 샘플링 (너무 많으면)
        if len(combinations) > max_evals:
            print(f"⚡ 조합이 너무 많습니다. {max_evals}개를 랜덤 샘플링합니다...")
            random.seed(42)
            combinations = random.sample(combinations, max_evals)
        
        results = []
        save_path = os.path.join(base_output_dir, f"macro_tuning_summary_{ticker}_{timestamp}.csv")
        
        # 원본 설정 백업
        original_macro_config = copy.deepcopy(cfg_agents.agents_info["MacroAgent"])
        
        try:
            for idx, params in enumerate(combinations):
                print(f"\n{'='*80}")
                print(f"🧪 실험 {idx+1}/{len(combinations)}")
                print(f"⚙️ 파라미터: {params}")
                print(f"{'='*80}")
                
                # =================================================================
                # 4. Config 동적 적용
                # =================================================================
                for key, value in params.items():
                    if key in cfg_agents.agents_info["MacroAgent"]:
                        cfg_agents.agents_info["MacroAgent"][key] = value
                
                # =================================================================
                # 5. 기존 모델 및 스케일러 삭제 (파라미터 변경 반영)
                # =================================================================
                print("🧹 기존 모델 및 스케일러 정리 중...")
                models_dir = os.path.join(project_root, "backtest", "models")
                cleanup_patterns = [
                    os.path.join(models_dir, f"{ticker}_MacroAgent.pt"),
                    os.path.join(models_dir, "scalers", f"{ticker}_MacroAgent_*.pkl"),
                ]
                
                for pat in cleanup_patterns:
                    for f in glob.glob(pat):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                # =================================================================
                # 6. 실험별 출력 디렉토리 생성 (run_tuning.py와 동일)
                # =================================================================
                exp_id = f"exp_{idx+1}"
                exp_output_dir = os.path.join(base_output_dir, exp_id)
                os.makedirs(exp_output_dir, exist_ok=True)
                
                # =================================================================
                # 7. MacroAgent 인스턴스 생성 및 롤링 예측 평가
                # =================================================================
                try:
                    # MacroAgent 인스턴스 생성 (파라미터는 config에서 자동 로드)
                    agent = MacroAgent(
                        agent_id="MacroAgent",
                        ticker=ticker,
                        data_dir=data_dir,
                        model_dir=models_dir
                    )
                    
                    # 롤링 예측 및 방향 예측도 계산
                    print("📈 롤링 예측 및 방향 예측도 계산 중...")
                    metrics = calculate_direction_accuracy_rolling(
                        agent=agent,
                        ticker=ticker,
                        data_dir=data_dir,
                        raw_dir=raw_dir,
                        predict_days=predict_days,
                        start_date=start_date,
                        output_dir=exp_output_dir
                    )
                    
                    if metrics:
                        # 결과 저장 (리스트 파라미터는 문자열로 변환)
                        row = {
                            "experiment_id": idx + 1,
                            "window_size": params.get("window_size"),
                            "hidden_dims": str(params.get("hidden_dims")),
                            "dropout_rates": str(params.get("dropout_rates")),
                            "learning_rate": params.get("learning_rate"),
                            "batch_size": params.get("batch_size"),
                            "epochs": params.get("epochs"),
                            "direction_accuracy": metrics.get("direction_accuracy", 0.0),
                            "mae": metrics.get("mae", 0.0),
                            "rmse": metrics.get("rmse", 0.0),
                            "correlation": metrics.get("correlation", 0.0),
                            "n_samples": metrics.get("n_samples", 0)
                        }
                        results.append(row)
                        print(f"✅ 실험 {idx+1} 완료! 방향 예측도: {row['direction_accuracy']:.2f}%")
                    else:
                        print(f"⚠️ 실험 {idx+1} 실패: 메트릭 계산 불가")
                        row = {
                            "experiment_id": idx + 1,
                            "window_size": params.get("window_size"),
                            "hidden_dims": str(params.get("hidden_dims")),
                            "dropout_rates": str(params.get("dropout_rates")),
                            "learning_rate": params.get("learning_rate"),
                            "batch_size": params.get("batch_size"),
                            "epochs": params.get("epochs"),
                            "error": "Metrics calculation failed"
                        }
                        results.append(row)
                    
                    # 실험마다 즉시 저장
                    if len(results) > 0:
                        try:
                            current_df = pd.DataFrame(results)
                            if "direction_accuracy" in current_df.columns:
                                current_df = current_df.sort_values(
                                    by="direction_accuracy",
                                    ascending=False
                                )
                            else:
                                current_df = current_df.sort_values(
                                    by="experiment_id",
                                    ascending=True
                                )
                            
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            current_df.to_csv(save_path, index=False)
                            file_size = os.path.getsize(save_path) / 1024
                            print(f"💾 요약 저장됨: {save_path} ({len(results)} 실험, {file_size:.2f} KB)")
                        except Exception as e:
                            print(f"❌ 저장 실패: {e}")
                            traceback.print_exc()
                    
                except Exception as e:
                    print(f"❌ 실험 {idx+1} 실패: {e}")
                    traceback.print_exc()
                    row = {
                        "experiment_id": idx + 1,
                        "window_size": params.get("window_size"),
                        "hidden_dims": str(params.get("hidden_dims")),
                        "dropout_rates": str(params.get("dropout_rates")),
                        "learning_rate": params.get("learning_rate"),
                        "batch_size": params.get("batch_size"),
                        "epochs": params.get("epochs"),
                        "error": str(e)
                    }
                    results.append(row)
        
        finally:
            # 설정 복구
            cfg_agents.agents_info["MacroAgent"] = original_macro_config
            
            print(f"\n{'='*80}")
            print("🏁 튜닝 완료")
            print(f"{'='*80}")
            
            # 최종 요약 출력 및 저장
            if results:
                df = pd.DataFrame(results)
                if "direction_accuracy" in df.columns:
                    df = df.sort_values(by="direction_accuracy", ascending=False)
                else:
                    df = df.sort_values(by="experiment_id", ascending=True)
                
                try:
                    df.to_csv(save_path, index=False)
                    print(f"\n💾 최종 결과 저장 완료: {save_path}")
                    print(f"   파일 크기: {os.path.getsize(save_path) / 1024:.2f} KB")
                except Exception as e:
                    print(f"❌ 최종 저장 실패: {e}")
                    traceback.print_exc()
                
                print(f"\n📊 최종 요약: {len(results)}개 실험 완료")
                print(f"📄 결과 파일: {save_path}")
                print("\n🏆 상위 5개 구성:")
                
                cols_to_show = ["experiment_id", "window_size", "hidden_dims", "dropout_rates", 
                               "learning_rate", "batch_size", "epochs", "direction_accuracy", "mae", "rmse", "correlation"]
                cols_to_show = [c for c in cols_to_show if c in df.columns]
                
                print(df[cols_to_show].head(5).to_string())
                
                # 1위 결과 반환
                if "direction_accuracy" in df.columns:
                    best_row = df.iloc[0].to_dict()
                    best_row["ticker"] = ticker
                    return best_row
            else:
                print("⚠️ 저장된 결과가 없습니다.")
            
            return None
    
    finally:
        # stdout 복구 및 로그 파일 닫기
        sys.stdout = original_stdout
        if not log_file.closed:
            log_file.close()
        print(f"\n📝 로그 저장됨: {log_file_path}")


if __name__ == "__main__":
    # 튜닝할 티커 리스트
    target_tickers = ["MSFT", "AAPL", "NVDA"]
    
    # 전체 종목 베스트 설정 수집용
    all_best_configs = []
    
    for ticker in target_tickers:
        print(f"\n{'#'*60}")
        print(f"🚀 MacroAgent 튜닝 시작: {ticker}")
        print(f"{'#'*60}\n")
        
        try:
            # 티커별로 결과 폴더 분리
            ticker_output_dir = os.path.join("backtest", "macro_tuning_results", ticker)
            
            best_result = run_macro_tuning(
                ticker=ticker,
                start_date=None,
                predict_days=30,
                max_evals=30,
                base_output_dir=ticker_output_dir
            )
            
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
        
        base_cols = ["ticker", "direction_accuracy", "experiment_id", "mae", "rmse", "correlation"]
        param_cols = [c for c in summary_df.columns if c not in base_cols and c not in ["error", "n_samples"]]
        final_cols = base_cols + param_cols
        
        final_cols = [c for c in final_cols if c in summary_df.columns]
        
        print(summary_df[final_cols].to_string(index=False))
        
        total_summary_path = os.path.join("backtest", "macro_tuning_results", f"total_best_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary_df[final_cols].to_csv(total_summary_path, index=False)
        print(f"\n📄 Total summary saved to: {total_summary_path}")
    else:
        print("\n❌ No successful tuning results collected.")

