# tech_tuning.py
# TechnicalAgent 전용 파라미터 튜닝 스크립트
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
from agents.technical_agent import TechnicalAgent
from core.technical_classes.technical_data_set import build_dataset, load_dataset as load_dataset_tech


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
    original_path = os.path.join(raw_dir, f"{ticker}_TechnicalAgent_raw.csv")
    
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
            temp_path = os.path.join(temp_dir, f"{ticker}_TechnicalAgent_raw_{date_str}.csv")
            df_filtered.to_csv(temp_path, index=False)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  [WARN] 데이터셋 필터링 실패: {e}")
        return False


def calculate_direction_accuracy_rolling(
    agent: TechnicalAgent,
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
        agent: TechnicalAgent 인스턴스
        ticker: 종목 코드
        data_dir: 데이터 디렉토리
        raw_dir: 원본 데이터 디렉토리
        predict_days: 예측할 거래일 수
        start_date: 시작 날짜 (YYYY-MM-DD, None이면 자동 계산)
    
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
                # 각 날짜별 예측 프로세스
                # ============================================================
                # 1. 백테스팅 모드 설정 (먼저 설정해야 searcher가 필터링된 데이터 사용)
                if hasattr(agent, 'set_test_mode'):
                    agent.set_test_mode(True)
                if hasattr(agent, 'set_simulation_date'):
                    agent.set_simulation_date(sim_date)
                
                # 2. 시점별 데이터셋 필터링 (Data Leakage 방지)
                _prepare_filtered_dataset(sim_date, ticker, raw_dir)
                
                # 3. 모델 재학습을 위해 기존 모델 삭제
                model_path = os.path.join(agent.model_dir, f"{ticker}_{agent.agent_id}.pt")
                if os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                    except:
                        pass
                
                # 스케일러도 삭제 (재학습 시 재생성)
                scaler_patterns = [
                    os.path.join(agent.model_dir, "scalers", f"{ticker}_{agent.agent_id}_*.pkl"),
                ]
                for pat in scaler_patterns:
                    for f in glob.glob(pat):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                # 4. 데이터셋 재생성 (필터링된 raw 데이터 사용)
                # searcher가 백테스팅 모드에서 backtest_temp의 필터링된 파일을 자동으로 사용
                build_dataset(
                    ticker=ticker,
                    save_dir=data_dir,
                    period="2y",  # 충분한 기간
                    interval="1d"
                )
                
                # 5. 데이터 준비 (DebateAgent와 동일한 순서: searcher 먼저)
                X_tensor = agent.searcher(ticker, rebuild=False)
                if X_tensor is None or X_tensor.shape[0] == 0:
                    print(f"    ⚠️ 데이터 없음, 스킵")
                    continue
                
                # 6. 모델 재학습 (DebateAgent와 동일한 순서: searcher 후 pretrain)
                agent.pretrain()
                
                # 7. 예측 수행
                if X_tensor is None or X_tensor.shape[0] == 0:
                    print(f"    ⚠️ 데이터 없음, 스킵")
                    continue
                
                # 스케일링
                agent.scaler.load(ticker)
                X_np = X_tensor.numpy()
                X_scaled, _ = agent.scaler.transform(X_np, np.array([[0.0]]))
                X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                
                # 예측
                agent.eval()
                with torch.no_grad():
                    pred_scaled = agent(X_scaled_tensor).item()
                
                # 스케일 역변환
                y_scale_factor = cfg_agents.common_params.get("y_scale_factor", 100.0)
                pred_return = pred_scaled / y_scale_factor
                
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
                
                pred_close = current_close * (1 + pred_return)
                
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
                
                # 임시 파일 정리
                temp_dir = os.path.join(raw_dir, "backtest_temp")
                date_str = sim_date.replace("-", "")
                temp_path = os.path.join(temp_dir, f"{ticker}_TechnicalAgent_raw_{date_str}.csv")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
            except Exception as e:
                print(f"    ❌ {sim_date} 예측 실패: {e}")
                import traceback as tb
                tb.print_exc()
                # 에러가 발생해도 다음 날짜로 계속 진행
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
            # 날짜 범위 계산
            if len(df_results) > 0:
                start_date_str = df_results["date"].iloc[0]
                end_date_str = df_results["date"].iloc[-1]
                csv_filename = f"rolling_{ticker}_{start_date_str}_{end_date_str}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                
                # CSV 형식 맞추기 (run_tuning.py와 동일하게)
                # Current_Close 제거하고 Actual_Close와 예측값만 저장
                output_df = pd.DataFrame({
                    "Date": df_results["date"],
                    "Ticker": ticker,
                    "Actual_Close": df_results["actual_close"],
                    "TechnicalAgent_Pred": df_results["pred_close"],
                })
                output_df.to_csv(csv_path, index=False)
                print(f"  💾 롤링 예측 결과 저장: {csv_path}")
        
        # 이전 종가 대비 방향 계산 (run_tuning.py와 동일한 방식)
        # prev_close = Actual_Close.shift(1)
        # 주의: df_results는 DataFrame이므로 Series로 변환하여 shift
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


def run_technical_tuning(
    ticker="MSFT",
    start_date=None,
    predict_days=30,
    period="2y",
    base_output_dir="backtest/tech_tuning_results",
    max_evals=200
):
    """
    TechnicalAgent 파라미터 튜닝 실행
    
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
    log_file_path = os.path.join(base_output_dir, f"tech_tuning_log_{ticker}_{timestamp}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # stdout을 파일과 터미널 모두에 출력하도록 설정
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_file)
    
    try:
        print(f"🚀 TechnicalAgent 파라미터 튜닝 시작: {ticker}")
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
            "window_size": [10, 20, 30],
            "rnn_units1": [32, 64, 128],
            "rnn_units2": [16, 32, 64],
            "dropout": [0.1, 0.2, 0.3],
            "learning_rate": [1e-4, 5e-4, 1e-3],
            "batch_size": [32, 64, 128],
            "epochs": [30, 45, 60],
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
        save_path = os.path.join(base_output_dir, f"tech_tuning_summary_{ticker}_{timestamp}.csv")
        
        # 원본 설정 백업
        original_tech_config = copy.deepcopy(cfg_agents.agents_info["TechnicalAgent"])
        
        try:
            for idx, params in enumerate(combinations):
                print(f"\n{'='*80}")
                print(f"🧪 실험 {idx+1}/{len(combinations)}")
                print(f"⚙️ 파라미터: {params}")
                print(f"{'='*80}")
                
                # =================================================================
                # 3. Config 동적 적용
                # =================================================================
                for key, value in params.items():
                    if key in cfg_agents.agents_info["TechnicalAgent"]:
                        cfg_agents.agents_info["TechnicalAgent"][key] = value
                
                # =================================================================
                # 4. 기존 모델 및 스케일러 삭제 (파라미터 변경 반영)
                # =================================================================
                print("🧹 기존 모델 및 스케일러 정리 중...")
                models_dir = os.path.join(project_root, "backtest", "models")
                cleanup_patterns = [
                    os.path.join(models_dir, f"{ticker}_TechnicalAgent.pt"),
                    os.path.join(models_dir, "scalers", f"{ticker}_TechnicalAgent_*.pkl"),
                ]
                
                for pat in cleanup_patterns:
                    for f in glob.glob(pat):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                # =================================================================
                # 5. 실험별 출력 디렉토리 생성 (run_tuning.py와 동일)
                # =================================================================
                exp_id = f"exp_{idx+1}"
                exp_output_dir = os.path.join(base_output_dir, exp_id)
                os.makedirs(exp_output_dir, exist_ok=True)
                
                # =================================================================
                # 6. TechnicalAgent 인스턴스 생성 및 롤링 예측 평가
                # =================================================================
                try:
                    # TechnicalAgent 인스턴스 생성 (파라미터는 config에서 자동 로드)
                    agent = TechnicalAgent(
                        agent_id="TechnicalAgent",
                        ticker=ticker,
                        data_dir=data_dir,
                        model_dir=models_dir,
                        need_training=True
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
                        output_dir=exp_output_dir  # 실험별 출력 디렉토리 전달
                    )
                    
                    if metrics:
                        # 결과 저장
                        row = {
                            "experiment_id": idx + 1,
                            **params,
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
                            **params,
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
                                # direction_accuracy가 없으면 experiment_id로 정렬
                                current_df = current_df.sort_values(
                                    by="experiment_id",
                                    ascending=True
                                )
                            
                            # 디렉토리 확인 및 생성
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            
                            current_df.to_csv(save_path, index=False)
                            file_size = os.path.getsize(save_path) / 1024  # KB
                            print(f"💾 요약 저장됨: {save_path} ({len(results)} 실험, {file_size:.2f} KB)")
                        except Exception as e:
                            print(f"❌ 저장 실패: {e}")
                            traceback.print_exc()
                    
                except Exception as e:
                    print(f"❌ 실험 {idx+1} 실패: {e}")
                    traceback.print_exc()
                    row = {
                        "experiment_id": idx + 1,
                        **params,
                        "error": str(e)
                    }
                    results.append(row)
        
        finally:
            # 설정 복구
            cfg_agents.agents_info["TechnicalAgent"] = original_tech_config
            
            print(f"\n{'='*80}")
            print("🏁 튜닝 완료")
            print(f"{'='*80}")
            
            # 최종 요약 출력 및 저장
            if results:
                df = pd.DataFrame(results)
                if "direction_accuracy" in df.columns:
                    df = df.sort_values(by="direction_accuracy", ascending=False)
                else:
                    # direction_accuracy가 없으면 experiment_id로 정렬
                    df = df.sort_values(by="experiment_id", ascending=True)
                
                # 최종 저장 (다시 한번 저장하여 확실히 보존)
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
                
                cols_to_show = ["experiment_id"] + list(param_grid.keys()) + ["direction_accuracy", "mae", "rmse", "correlation"]
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
        print(f"🚀 TechnicalAgent 튜닝 시작: {ticker}")
        print(f"{'#'*60}\n")
        
        try:
            # 티커별로 결과 폴더 분리
            ticker_output_dir = os.path.join("backtest", "tech_tuning_results", ticker)
            
            best_result = run_technical_tuning(
                ticker=ticker,
                start_date=None,  # None이면 자동 계산
                predict_days=30,  # 30일 예측
                max_evals=200,
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
        
        # 존재하는 컬럼만 선택
        final_cols = [c for c in final_cols if c in summary_df.columns]
        
        print(summary_df[final_cols].to_string(index=False))
        
        total_summary_path = os.path.join("backtest", "tech_tuning_results", f"total_best_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary_df[final_cols].to_csv(total_summary_path, index=False)
        print(f"\n📄 Total summary saved to: {total_summary_path}")
    else:
        print("\n❌ No successful tuning results collected.")

