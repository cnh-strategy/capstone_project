
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 sys.path에 추가
# backtest/scripts에서 실행 시 프로젝트 루트는 ../../ 이 됨
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from agents.debate_agent import DebateAgent
from core.metrics import calculate_metrics, calculate_direction_accuracy, calculate_profitability
from config.agents import agents_info, dir_info, common_params
from core.macro_classes.macro_llm import Opinion

# Backtest 전용 디렉토리 설정
BACKTEST_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # backtest 폴더
BACKTEST_DATA_DIR = os.path.join(BACKTEST_ROOT, "data", "processed")
BACKTEST_RAW_DIR = os.path.join(BACKTEST_ROOT, "data", "raw")
BACKTEST_MODEL_DIR = os.path.join(BACKTEST_ROOT, "models")
BACKTEST_SCALER_DIR = os.path.join(BACKTEST_MODEL_DIR, "scalers")
BACKTEST_OUTPUT_DIR = os.path.join(BACKTEST_ROOT, "data", "backtests")

class RollingBacktester:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        predict_days: int,
        rounds: int = 3,
        output_dir: str = None,
        auto_analyze: bool = True,
    ):
        self.ticker = ticker.upper()
        self.predict_days = predict_days
        self.rounds = rounds
        self.output_dir = output_dir or BACKTEST_OUTPUT_DIR
        self.auto_analyze = auto_analyze
        self.results: List[Dict[str, Any]] = []
        self.csv_path: Optional[str] = None

        # config에서 period 가져오기 (예: "2y")
        self.period_str = common_params.get("period", "2y")
        self.train_days = self._parse_period(self.period_str)
        
        print(f"[INFO] Config Period: {self.period_str} -> Train Days: {self.train_days}")

        # start_date가 없으면 자동 계산
        if start_date is None:
            # 넉넉하게 2배 기간 전부터 오늘까지 평일을 구함
            today = datetime.today()
            lookback = max(30, predict_days * 2)
            temp_start = today - timedelta(days=lookback)
            
            # 평일(월~금)만 추출
            candidates = self._generate_trading_days(temp_start, lookback)
            
            # 오늘보다 과거인 날짜만 필터링
            candidates = [d for d in candidates if d < today]
            
            if len(candidates) < predict_days:
                # 데이터가 너무 부족하면 그냥 predict_days 전으로 강제 설정
                self.start_dt = today - timedelta(days=predict_days + 2)
                print(f"[WARN] 거래일 계산 부족으로 단순 계산된 시작일 사용: {self.start_dt.strftime('%Y-%m-%d')}")
            else:
                # 뒤에서부터 predict_days 만큼 가져오기
                # 예: predict_days=5이면, candidates[-5]가 시작일
                self.start_dt = candidates[-predict_days]
                print(f"[INFO] 자동 계산된 시작일(Start Date): {self.start_dt.strftime('%Y-%m-%d')} (오늘로부터 {predict_days} 거래일 전)")
            
            self.start_date = self.start_dt.strftime("%Y-%m-%d")
        else:
            self.start_date = start_date
            self.start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        self.predict_dates = self._generate_trading_days(self.start_dt, self.predict_days)
        if not self.predict_dates:
            raise ValueError("예측할 거래일을 찾을 수 없습니다. start_date를 확인하세요.")
        self.end_date = self.predict_dates[-1].strftime("%Y-%m-%d")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(BACKTEST_DATA_DIR, exist_ok=True)
        os.makedirs(BACKTEST_RAW_DIR, exist_ok=True)
        os.makedirs(BACKTEST_MODEL_DIR, exist_ok=True)
        os.makedirs(BACKTEST_SCALER_DIR, exist_ok=True)

    def _run_backtest_without_llm(self, agent: DebateAgent, force_pretrain: bool = False) -> Dict[str, Any]:
        """
        백테스트 전용: LLM 호출 없이 예측만 수행
        - reviewer_draft 대신 직접 Opinion 생성 (reason은 빈 문자열)
        - get_rebuttal 스킵 (LLM 호출이므로)
        - get_revise 진행 (fine-tuning과 재예측은 수행, LLM 호출만 스킵)
        - get_ensemble 호출
        """
        try:
            if not hasattr(agent, "opinions"):
                agent.opinions = {}
            
            ticker = agent.ticker
            
            # Round 0: 초기 Opinion 수집 (LLM 없이)
            print(f"\n{'='*80}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Round 0: 초기 Opinion 수집 시작 (LLM 없이, force_pretrain={force_pretrain})")
            print(f"{'='*80}")
            
            opinions = {}
            
            for agent_id, ag in agent.agents.items():
                # === 1단계: 모델 준비 확인 ===
                is_ready = agent._check_agent_ready(agent_id, ticker)
                needs_pretrain = force_pretrain or (not is_ready)
                
                # === 2단계: 에이전트별 데이터 수집 및 학습 ===
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] searcher 실행 (데이터셋 준비)")
                X = ag.searcher(ticker, rebuild=False)
                
                if needs_pretrain:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] pretrain 실행 (모델/스케일러 생성)")
                    ag.pretrain()
                else:
                    model_path = os.path.join(ag.model_dir, f"{ticker}_{agent_id}.pt")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] 기존 모델 사용: {model_path}")
                
                # 예측 (LLM 호출 없이)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] predict 실행 (LLM 없이)")
                if agent_id == "SentimentalAgent":
                    n_samples = common_params.get("n_samples", 30)
                    target = ag.predict(ag.stockdata, n_samples=n_samples)
                else:
                    target = ag.predict(X)
                
                # Opinion 직접 생성 (reason은 빈 문자열, LLM 호출 없음)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] Opinion 생성 (LLM 스킵)")
                opinion = Opinion(agent_id=agent_id, target=target, reason="(백테스트 모드: LLM 호출 없음)")
                ag.opinions.append(opinion)
                opinions[agent_id] = opinion
                
                try:
                    print(f"  - {agent_id}: next_close={opinion.target.next_close:.4f}")
                except Exception:
                    pass
            
            agent.opinions[0] = opinions
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Round 0 의견 수집 완료 ({len(opinions)} agents, LLM 없이)")
            
            # Round 1~N: Rebuttal 스킵, Revise는 진행 (LLM 호출만 스킵)
            # get_revise()는 fine-tuning과 재예측을 수행하므로 진행해야 함
            for round_num in range(1, agent.rounds + 1):
                print(f"\n{'='*80}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round_num} 시작 (Rebuttal 스킵, Revise 진행)")
                print(f"{'='*80}")
                
                # Rebuttal은 스킵 (LLM 호출이므로)
                # 빈 rebuttals 리스트로 설정
                if not hasattr(agent, "rebuttals"):
                    agent.rebuttals = {}
                agent.rebuttals[round_num] = []
                
                # Revise 진행 (LLM 호출만 스킵)
                # 각 agent의 _ask_with_fallback을 임시로 모킹
                original_ask_methods = {}
                for agent_id, ag in agent.agents.items():
                    # 원본 메서드 백업
                    original_ask_methods[agent_id] = ag._ask_with_fallback
                    
                    # LLM 호출 없이 기본 reason만 반환하는 함수로 교체
                    def make_no_llm_ask(agent_id_inner):
                        def no_llm_ask(msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
                            # schema_obj에서 reason 필드 추출
                            if schema_obj and isinstance(schema_obj, dict):
                                props = schema_obj.get("properties", {})
                                if "reason" in props:
                                    return {"reason": f"(백테스트 모드: {agent_id_inner} revise, LLM 호출 없음)"}
                            # 기본 반환
                            return {"reason": f"(백테스트 모드: {agent_id_inner} revise, LLM 호출 없음)"}
                        return no_llm_ask
                    
                    ag._ask_with_fallback = make_no_llm_ask(agent_id)
                
                try:
                    # get_revise 호출 (fine-tuning과 재예측은 수행, LLM만 스킵)
                    agent.get_revise(round_num)
                finally:
                    # 원본 메서드 복원
                    for agent_id, ag in agent.agents.items():
                        ag._ask_with_fallback = original_ask_methods[agent_id]
            
            # 최종 Ensemble 예측 (LLM 사용 안 함)
            print(f"\n{'='*80}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 최종 Ensemble 예측")
            print(f"{'='*80}")
            
            ensemble_result = agent.get_ensemble()
            print(ensemble_result)
            
            return ensemble_result
        except Exception as e:
            print(f"❌ Error in _run_backtest_without_llm: {e}")
            raise

    def _parse_period(self, period: str) -> int:
        """
        '2y', '1y', '6mo' 등의 문자열을 일수(int)로 변환
        """
        period = period.lower()
        if period.endswith("y"):
            return int(period[:-1]) * 365
        elif period.endswith("mo"):
            return int(period[:-2]) * 30
        elif period.endswith("d"):
            return int(period[:-1])
        else:
            # 기본값 2년
            print(f"[WARN] Unknown period format '{period}', defaulting to 730 days")
            return 730

    @staticmethod
    def _generate_trading_days(start_dt: datetime, count: int) -> List[datetime]:
        days: List[datetime] = []
        current = start_dt
        while len(days) < count:
            if current.weekday() < 5:  # 0=Monday, ..., 4=Friday
                days.append(current)
            current += timedelta(days=1)
        return days

    def prepare_data(self):
        """
        Searcher를 사용하여 데이터 준비 (agent 코드 수정 없이)
        각 agent의 searcher() 메서드를 호출하여 raw CSV를 생성합니다.
        """
        total_days = self.train_days + self.predict_days + 60  # 여유분 60일
        total_years = max(int(total_days / 365) + 1, 1)

        print(
            f"\n{'='*60}"
        )
        print(f"📊 Preparing data for {self.ticker}")
        print(f"    Train days: {self.train_days} (from config)")
        print(f"    Predict days: {self.predict_days}")
        print(f"    Total years needed: ~{total_years}y")
        print(f"    Backtest data dir: {BACKTEST_DATA_DIR}")
        print(f"    Backtest raw dir: {BACKTEST_RAW_DIR}")
        print(f"{'='*60}\n")

        # DebateAgent를 임시로 생성하여 각 agent의 searcher 호출
        # test용 dir을 전달하여 초기화 시점에 경로 설정
        temp_agent = DebateAgent(
            ticker=self.ticker,
            rounds=1,
            data_dir=BACKTEST_DATA_DIR,
            model_dir=BACKTEST_MODEL_DIR
        )
        
        # 각 agent의 searcher 호출 (rebuild=True로 강제 재생성)
        print(f"[INFO] 각 Agent의 searcher()를 호출하여 데이터 준비 중...\n")
        for agent_id, agent in temp_agent.agents.items():
            try:
                print(f"[{agent_id}] searcher 실행 중...")
                agent.searcher(ticker=self.ticker, rebuild=True)
                print(f"✅ [{agent_id}] 데이터 준비 완료\n")
            except Exception as e:
                print(f"❌ [{agent_id}] 데이터 준비 실패: {e}\n")
                raise
        
        print("✅ Data preparation complete using searcher methods.")

    def run_loop(self):
        """
        Predict Period를 거래일 단위로 순회하며 시뮬레이션 수행
        
        각 시점별로:
        1. 전체 데이터셋에서 해당 시점 이전 데이터만 필터링
        2. 필터링된 데이터셋으로 모델 학습
        3. 학습된 모델로 예측 수행
        """
        for day_idx, sim_dt in enumerate(self.predict_dates, start=1):
            sim_date = sim_dt.strftime("%Y-%m-%d")
            train_start_dt = sim_dt - timedelta(days=self.train_days)
            train_start = train_start_dt.strftime("%Y-%m-%d")

            print(f"\n{'='*60}")
            print(f"🚀 [{day_idx}/{len(self.predict_dates)}] Running Backtest for {sim_date}")
            print(f"    Train window: {train_start} ~ {sim_date}")
            print(f"{'='*60}")

            # 각 시점별로 필터링된 데이터셋 생성
            self._prepare_filtered_datasets(sim_date)

            # DebateAgent 생성 시 test용 dir 전달
            agent = DebateAgent(
                ticker=self.ticker,
                rounds=self.rounds,
                data_dir=BACKTEST_DATA_DIR,
                model_dir=BACKTEST_MODEL_DIR
            )
            
            # test_mode 설정 (LLM 호출 스킵)
            for name, ag in agent.agents.items():
                if hasattr(ag, "set_test_mode"):
                    ag.set_test_mode(True)
                if hasattr(ag, "set_simulation_date"):
                    ag.set_simulation_date(sim_date)
                if hasattr(ag, "set_training_window"):
                    ag.set_training_window(train_start)
                print(f"[{name}] Simulation={sim_date}, TrainStart={train_start}")

            try:
                # 백테스트 전용: LLM 호출 없이 예측만 수행
                result = self._run_backtest_without_llm(agent, force_pretrain=True)
                result["simulation_date"] = sim_date

                self._collect_result(sim_date, result, agent)
                self.save_results()
                
                # 백테스팅 모델 파일 삭제 (다음 날짜에서 깨끗한 상태로 재학습)
                self._cleanup_backtest_models(sim_date)
                
                # 필터링된 임시 데이터셋 삭제
                self._cleanup_filtered_datasets(sim_date)

            except Exception as e:
                print(f"❌ Error on {sim_date}: {e}")
                # 에러 발생 시에도 정리
                try:
                    self._cleanup_backtest_models(sim_date)
                    self._cleanup_filtered_datasets(sim_date)
                except:
                    pass
                continue

        if self.auto_analyze:
            print(f"\n{'='*60}")
            print("📊 Starting automatic analysis...")
            print(f"{'='*60}")
            self.analyze()

    def _prepare_filtered_datasets(self, sim_date: str):
        """
        각 시점별로 전체 데이터셋에서 해당 시점 이전 데이터만 필터링하여 임시 데이터셋 생성
        
        Args:
            sim_date: 시뮬레이션 날짜 (YYYY-MM-DD)
        """
        sim_date_dt = pd.to_datetime(sim_date)
        temp_dir = os.path.join(BACKTEST_RAW_DIR, "backtest_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        ticker = self.ticker
        agents = ["TechnicalAgent", "MacroAgent", "SentimentalAgent"]
        
        for agent_id in agents:
            # 원본 raw CSV 경로 (backtest 전용)
            original_path = os.path.join(BACKTEST_RAW_DIR, f"{ticker}_{agent_id}_raw.csv")
            if not os.path.exists(original_path):
                print(f"[WARN] 원본 데이터 파일이 없음: {original_path}")
                continue
            
            # 임시 필터링된 CSV 경로
            temp_path = os.path.join(temp_dir, f"{ticker}_{agent_id}_raw_{sim_date.replace('-', '')}.csv")
            
            try:
                # 원본 데이터 로드
                df = pd.read_csv(original_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
                
                # simulation_date 이전 데이터만 필터링
                # ⚠️ 주의: pretrain()에서 타겟 생성 시 마지막 타겟이 sim_date 당일 수익률이 될 수 있음
                # 각 에이전트의 pretrain()에서 마지막 타겟을 제거하도록 수정됨
                df_filtered = df[df["Date"] < sim_date_dt].copy()
                
                if len(df_filtered) > 0:
                    # 필터링된 데이터 저장
                    df_filtered.to_csv(temp_path, index=False)
                    print(f"[INFO] 필터링된 데이터셋 생성: {agent_id} ({len(df_filtered)}행, {sim_date} 이전)")
                else:
                    print(f"[WARN] 필터링된 데이터가 없음: {agent_id} ({sim_date})")
                    
            except Exception as e:
                print(f"[WARN] 데이터셋 필터링 실패 ({agent_id}): {e}")
    
    def _cleanup_filtered_datasets(self, sim_date: str):
        """
        필터링된 임시 데이터셋 삭제
        
        Args:
            sim_date: 시뮬레이션 날짜 (YYYY-MM-DD)
        """
        import glob
        
        temp_dir = os.path.join(BACKTEST_RAW_DIR, "backtest_temp")
        
        if not os.path.exists(temp_dir):
            return
        
        # 해당 날짜의 임시 파일 삭제
        date_str = sim_date.replace("-", "")
        pattern = os.path.join(temp_dir, f"{self.ticker}_*_raw_{date_str}.csv")
        temp_files = glob.glob(pattern)
        
        for file_path in temp_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[WARN] 임시 데이터셋 삭제 실패 ({file_path}): {e}")
    
    def _cleanup_backtest_models(self, sim_date: str):
        """
        백테스팅 모델 파일 삭제 (각 날짜 처리 후 호출)
        다음 날짜에서 깨끗한 상태로 재학습하기 위함
        """
        ticker = self.ticker
        
        # 삭제할 모델 파일 목록
        model_files = [
            os.path.join(BACKTEST_MODEL_DIR, f"{ticker}_TechnicalAgent.pt"),
            os.path.join(BACKTEST_MODEL_DIR, f"{ticker}_MacroAgent.pt"),
            os.path.join(BACKTEST_MODEL_DIR, f"{ticker}_SentimentalAgent.pt"),
            os.path.join(BACKTEST_MODEL_DIR, f"{ticker}_ensemble_lightgbm.pt"),
        ]
        
        # 스케일러 파일
        scaler_files = [
            os.path.join(BACKTEST_SCALER_DIR, f"{ticker}_MacroAgent_xscaler.pkl"),
            os.path.join(BACKTEST_SCALER_DIR, f"{ticker}_MacroAgent_yscaler.pkl"),
        ]
        
        deleted_count = 0
        for file_path in model_files + scaler_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"[WARN] 모델 파일 삭제 실패 ({file_path}): {e}")
        
        if deleted_count > 0:
            print(f"[INFO] 백테스팅 모델 파일 {deleted_count}개 삭제 완료 ({sim_date})")

    def _collect_result(self, date: str, result: Dict[str, Any], agent: Any = None):
        """
        라운드별 예측 결과를 평탄화하여 저장
        """
        # 실제 종가 조회 (yfinance)
        actual_close = np.nan
        try:
            import yfinance as yf
            # date 다음날까지 조회해야 date 당일 데이터가 나옴 (yfinance 특성)
            next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(self.ticker, start=date, end=next_day, progress=False)
            if not df.empty:
                # MultiIndex 처리
                val = df["Close"].iloc[0]
                if isinstance(val, pd.Series):
                    actual_close = float(val.iloc[0])
                else:
                    actual_close = float(val)
        except Exception as e:
            print(f"[WARN] 실제 종가 조회 실패({date}): {e}")

        row = {
            "Date": date,
            "Ticker": self.ticker,
            "Actual_Close": actual_close,
            "Ensemble_Pred": result.get("ensemble_next_close"),
        }

        # Agent 인스턴스에서 라운드별 예측값 수집
        if agent and hasattr(agent, "opinions"):
            for round_idx, opinions in agent.opinions.items():
                for agent_id, opinion in opinions.items():
                    # {Agent_id}_R{round}_Pred 형식
                    key = f"{agent_id}_R{round_idx}_Pred"
                    if opinion and opinion.target:
                        row[key] = float(opinion.target.next_close)

        self.results.append(row)

    def save_results(self):
        df = pd.DataFrame(self.results)
        filename = f"rolling_{self.ticker}_{self.start_date}_{self.end_date}.csv"
        self.csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(self.csv_path, index=False)
        print(f"💾 Results saved to {self.csv_path}")
    
    def analyze(self, csv_path: str = None, output_dir: str = None):
        """
        백테스팅 결과 분석 및 시각화
        """
        if csv_path is None:
            csv_path = self.csv_path
        
        if csv_path is None or not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return
        
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터 로드
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        print(f"\n📈 Loaded {len(df)} rows from {csv_path}")
        
        # 1. 기본 지표 계산
        df_valid = df.dropna(subset=['Actual_Close', 'Ensemble_Pred'])
        
        if len(df_valid) == 0:
            print("❌ No valid data for analysis")
            return
        
        y_true = df_valid['Actual_Close'].values
        y_pred = df_valid['Ensemble_Pred'].values
        
        metrics = calculate_metrics(y_true, y_pred)
        print("\n[Ensemble Performance]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # 방향 정확도
        prev_close = df_valid['Actual_Close'].shift(1).fillna(method='bfill').values
        dir_acc = calculate_direction_accuracy(y_true, y_pred, prev_close)
        print(f"  Direction Accuracy: {dir_acc:.2f}%")
        
        # 2. 수익률 분석
        dates = df_valid['Date'].dt.strftime('%Y-%m-%d').tolist()
        prof_res = calculate_profitability(dates, y_true, y_pred)
        print("\n[Profitability]")
        print(f"  Strategy Return: {prof_res['Strategy_Return']:.2f}%")
        print(f"  Buy & Hold Return: {prof_res['BuyHold_Return']:.2f}%")
        
        # 3. 시각화
        base_name = os.path.basename(csv_path).replace(".csv", "")
        
        # A. Price Chart
        plt.figure(figsize=(12, 6))
        plt.plot(df_valid['Date'], df_valid['Actual_Close'], label='Actual Close', color='black', linewidth=2)
        plt.plot(df_valid['Date'], df_valid['Ensemble_Pred'], label='Ensemble Pred', color='blue', linestyle='--', linewidth=1.5)
        
        # Agent별 예측 추가 (있는 경우)
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        agent_pred_cols = [c for c in df.columns if c.endswith('_Pred') and c != 'Ensemble_Pred']
        for i, col in enumerate(agent_pred_cols):
            if col in df_valid.columns:
                plt.plot(df_valid['Date'], df_valid[col], label=col.replace('_Pred', ''), alpha=0.5, linestyle=':', color=colors[i%len(colors)])
        
        plt.title(f"Price Prediction: {base_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        price_chart_path = os.path.join(output_dir, f"{base_name}_price.png")
        plt.savefig(price_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved price chart: {price_chart_path}")
        
        # B. Cumulative Return
        df_valid['Daily_Ret'] = df_valid['Actual_Close'].pct_change().fillna(0)
        df_valid['Prev_Close'] = df_valid['Actual_Close'].shift(1)
        df_valid['Signal'] = np.where(df_valid['Ensemble_Pred'] > df_valid['Prev_Close'], 1, 0)
        df_valid['Strat_Daily_Ret'] = df_valid['Signal'] * df_valid['Daily_Ret']
        
        df_valid['Cum_BH'] = (1 + df_valid['Daily_Ret']).cumprod()
        df_valid['Cum_Strat'] = (1 + df_valid['Strat_Daily_Ret']).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_valid['Date'], df_valid['Cum_BH'], label='Buy & Hold', color='gray', linewidth=2)
        plt.plot(df_valid['Date'], df_valid['Cum_Strat'], label='Strategy', color='red', linewidth=2)
        plt.title(f"Cumulative Return: {base_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return_chart_path = os.path.join(output_dir, f"{base_name}_return.png")
        plt.savefig(return_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved return chart: {return_chart_path}")
        
        # C. Error Histogram
        errors = (df_valid['Ensemble_Pred'] - df_valid['Actual_Close']) / df_valid['Actual_Close'] * 100
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
        plt.title(f"Error Distribution (%) : {base_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Error %", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        error_chart_path = os.path.join(output_dir, f"{base_name}_error.png")
        plt.savefig(error_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved error histogram: {error_chart_path}")
        
        print(f"\n✅ Analysis complete! Charts saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Backtest Runner with Auto Analysis")
    parser.add_argument("--ticker", type=str, required=True, help="Target Ticker (e.g. AAPL)")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="첫 번째 예측일 (YYYY-MM-DD). 지정하지 않으면 '오늘 - predict_days' 거래일 전으로 자동 설정됨",
    )
    parser.add_argument(
        "--predict-days",
        type=int,
        default=5,
        help="예측할 거래일 수 (기본 5일)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="디베이트 라운드 수 (기본 3회)",
    )
    parser.add_argument("--no-analyze", action="store_true", help="Skip automatic analysis after backtest")

    args = parser.parse_args()

    runner = RollingBacktester(
        ticker=args.ticker,
        start_date=args.start,
        predict_days=args.predict_days,
        rounds=args.rounds,
        auto_analyze=not args.no_analyze,
    )
    runner.prepare_data()
    runner.run_loop()

