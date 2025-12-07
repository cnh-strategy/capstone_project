import os
import pandas as pd
from config.agents import dir_info
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.debate_agent import DebateAgent
from core.data_set import build_dataset
from core.metrics import calculate_metrics, calculate_direction_accuracy, calculate_profitability
from config.agents import agents_info, dir_info, common_params

class RollingBacktester:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        predict_days: int,
        rounds: int = 3,
        output_dir: str = "data/backtests",
        auto_analyze: bool = True,
    ):
        self.ticker = ticker.upper()
        self.predict_days = predict_days
        self.rounds = rounds
        self.output_dir = output_dir
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
        Lookback(train_days) + Predict(predict_days) 기간을 한 번에 수집합니다.
        """
        total_days = self.train_days + self.predict_days + 60  # 여유분 60일
        total_years = max(int(total_days / 365) + 1, 1)

        print(
            f"Preparing data for {self.ticker}: train_days={self.train_days} (from config), "
            f"predict_days={self.predict_days}, total_years≈{total_years}"
        )

        build_dataset(
            ticker=self.ticker,
            save_dir=dir_info["data_dir"],
            period=f"{total_years}y"
        )
        print("Data preparation complete.")

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

            # 각 시점별로 필터링된 데이터셋 생성  >> ?기본데이터셋은 어디서 생성되나
            self._prepare_filtered_datasets(sim_date)

            agent = DebateAgent(ticker=self.ticker, rounds=self.rounds)

            # ?set_test_mode 가 뭐지. 어디서 설정하는거지
            for name, ag in agent.agents.items():
                if hasattr(ag, "set_test_mode"):
                    ag.set_test_mode(True)
                if hasattr(ag, "set_simulation_date"):
                    ag.set_simulation_date(sim_date)
                if hasattr(ag, "set_training_window"):
                    ag.set_training_window(train_start)
                print(f"[{name}] Simulation={sim_date}, TrainStart={train_start}")

            try:
                result = agent.run(force_pretrain=True)
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
        raw_dir = os.path.join(os.path.dirname(dir_info["data_dir"]), "raw")
        temp_dir = os.path.join(raw_dir, "backtest_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        ticker = self.ticker
        agents = ["TechnicalAgent", "MacroAgent"]
        
        for agent_id in agents:
            # 원본 raw CSV 경로
            original_path = os.path.join(raw_dir, f"{ticker}_{agent_id}_raw.csv")
            if not os.path.exists(original_path):
                continue
            
            # 임시 필터링된 CSV 경로
            temp_path = os.path.join(temp_dir, f"{ticker}_{agent_id}_raw_{sim_date.replace('-', '')}.csv")
            
            try:
                # 원본 데이터 로드
                df = pd.read_csv(original_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
                
                # simulation_date 이전 데이터만 필터링
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

        raw_dir = os.path.join(os.path.dirname(dir_info["data_dir"]), "raw")
        temp_dir = os.path.join(raw_dir, "backtest_temp")
        
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
        
        model_dir = dir_info["model_dir"]
        ticker = self.ticker
        
        # 삭제할 모델 파일 목록
        model_files = [
            os.path.join(model_dir, f"{ticker}_TechnicalAgent.pt"),
            os.path.join(model_dir, f"{ticker}_MacroAgent.pt"),
            os.path.join(model_dir, f"{ticker}_SentimentalAgent.pt"),
            os.path.join(model_dir, f"{ticker}_ensemble_lightgbm.pt"),
        ]
        
        # MacroAgent 스케일러 파일
        scaler_dir = os.path.join(model_dir, "scalers")
        scaler_files = [
            os.path.join(scaler_dir, f"{ticker}_MacroAgent_xscaler.pkl"),
            os.path.join(scaler_dir, f"{ticker}_MacroAgent_yscaler.pkl"),
        ]
        
        # TechnicalAgent, SentimentalAgent 스케일러 (BaseAgent 사용)
        # 스케일러는 ticker별로 저장되므로 삭제하지 않음 (재사용 가능)
        # 필요시 추가 가능
        
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

        # D. Combo Chart: Price Trend & Daily Return 
        # Strategy_Return (%) 계산: 기존 Strat_Daily_Ret(소수점)을 퍼센트로 변환
        df_valid['Strategy_Return_Pct'] = df_valid['Strat_Daily_Ret'] * 100 
        
        plt.style.use('ggplot')

        # 두 개의 서브플롯을 생성 (3:1 비율)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})

        # --- 상단 패널: 주가 및 예측 가격 추이 (ax1) ---

        # 실제 종가
        ax1.plot(df_valid['Date'], df_valid['Actual_Close'], label='실제 종가', color='#0077b6', linewidth=2)

        # 앙상블 예측 종가 (기존 Ensemble_Pred 컬럼 사용)
        ax1.plot(df_valid['Date'], df_valid['Ensemble_Pred'], label='앙상블 예측 종가', color='#ff6f00', linestyle='--', linewidth=1.5)

        ax1.set_ylabel('가격 (Price)', fontsize=12)
        ax1.set_title(f'주가 예측 및 전략 일일 수익률: {base_name}', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # --- 하단 패널: 전략 일일 수익률 (ax2) ---

        # 전략 일일 수익률 (0% 기준 막대 차트, 양/음수 색상 분리)
        ax2.bar(df_valid['Date'], df_valid['Strategy_Return_Pct'], label='전략 일일 수익률 (%)', 
                color=np.where(df_valid['Strategy_Return_Pct'] >= 0, '#2a9d8f', '#e76f51'),
                width=1.0) # width=1.0을 추가하여 날짜 간격에 맞춥니다.

        # 0% 기준선
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)

        ax2.set_xlabel('날짜 (Date)', fontsize=12)
        ax2.set_ylabel('일일 수익률 (%)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', linestyle=':')

        # 레이아웃 조정 및 저장
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_combo_return.png"))
        print(f"Saved combo return chart to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Backtest Runner with Auto Analysis")
    parser.add_argument("--ticker", type=str, default='AAPL',
                        required=False, help="Target Ticker (e.g. AAPL)")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="첫 번째 예측일 (YYYY-MM-DD). 지정하지 않으면 '오늘 - predict_days' 거래일 전으로 자동 설정됨",
    )
    # train-days 인자 제거 (config 사용)
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



