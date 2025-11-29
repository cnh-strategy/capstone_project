
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.debate_agent import DebateAgent
from core.data_set import build_dataset
from core.metrics import calculate_metrics, calculate_direction_accuracy, calculate_profitability
from config.agents import agents_info, dir_info

class RollingBacktester:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        train_days: int,
        predict_days: int,
        rounds: int = 3,
        output_dir: str = "data/backtests",
        auto_analyze: bool = True,
    ):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        self.train_days = train_days
        self.predict_days = predict_days
        self.rounds = rounds
        self.output_dir = output_dir
        self.auto_analyze = auto_analyze
        self.results: List[Dict[str, Any]] = []
        self.csv_path: Optional[str] = None

        self.predict_dates = self._generate_trading_days(self.start_dt, self.predict_days)
        if not self.predict_dates:
            raise ValueError("예측할 거래일을 찾을 수 없습니다. start_date를 확인하세요.")
        self.end_date = self.predict_dates[-1].strftime("%Y-%m-%d")

        os.makedirs(self.output_dir, exist_ok=True)

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
        total_days = self.train_days + self.predict_days + 30  # 여유분
        total_years = max(int(total_days / 365) + 1, 1)

        print(
            f"Preparing data for {self.ticker}: train_days={self.train_days}, "
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
        """
        for day_idx, sim_dt in enumerate(self.predict_dates, start=1):
            sim_date = sim_dt.strftime("%Y-%m-%d")
            train_start_dt = sim_dt - timedelta(days=self.train_days)
            train_start = train_start_dt.strftime("%Y-%m-%d")

            print(f"\n{'='*60}")
            print(f"🚀 [{day_idx}/{len(self.predict_dates)}] Running Backtest for {sim_date}")
            print(f"    Train window: {train_start} ~ {sim_date}")
            print(f"{'='*60}")

            agent = DebateAgent(ticker=self.ticker, rounds=self.rounds)

            # 질문: set_test_mode 가 뭔지??
            for name, ag in agent.agents.items():
                if hasattr(ag, "set_test_mode"):
                    ag.set_test_mode(True)
                if hasattr(ag, "set_simulation_date"):
                    ag.set_simulation_date(sim_date)
                if hasattr(ag, "set_training_window"):
                    ag.set_training_window(train_start)
                print(f"[{name}] Simulation={sim_date}, TrainStart={train_start}")

            try:
                # 디베이트 시작
                result = agent.run(force_pretrain=True)
                result["simulation_date"] = sim_date

                # 라운드 별 내용을 list화
                self._collect_result(sim_date, result)
                # 해당 내용을 csv로 저장
                self.save_results()

            except Exception as e:
                print(f"❌ Error on {sim_date}: {e}")
                continue

        # 클래스 정의 떄 분석 여부 인자 받음
        if self.auto_analyze:
            print(f"\n{'='*60}")
            print("📊 Starting automatic analysis...")
            print(f"{'='*60}")
            self.analyze()

    def _collect_result(self, date: str, result: Dict[str, Any]):
        """
        라운드별 예측 결과를 평탄화하여 저장
        """
        row = {
            "Date": date,
            "Ticker": self.ticker,
            "Actual_Close": result.get("last_price"),
            "Ensemble_Pred": result.get("ensemble_next_close"),
            "Mean_Pred": result.get("mean_next_close"),
        }

        agents_data = result.get("agents", {})
        for k, v in agents_data.items():
            simple_key = k.replace("_next_close", "")
            row[f"{simple_key}_R{self.rounds}_Pred"] = v

        history: List[Dict[str, Any]] = result.get("round_history", [])
        for entry in history:
            round_idx = entry.get("round")
            if round_idx is None:
                continue
            for agent_id in ["TechnicalAgent", "MacroAgent", "SentimentalAgent"]:
                key = entry.get(f"{agent_id}_next_close")
                if key is not None:
                    row[f"{agent_id}_R{round_idx}_Pred"] = key

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
    parser.add_argument("--ticker", type=str, default="AAPL",  # 기본 티커 원하는 값으로 설정
                        required=False, help="Target Ticker (e.g. AAPL)")
    parser.add_argument(
        "--start",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="첫 번째 예측일 (YYYY-MM-DD). 기본값은 오늘",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=365 * 3,
        help="학습에 사용할 Lookback 일수 (기본 3년)",
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
        train_days=args.train_days,
        predict_days=args.predict_days,
        rounds=args.rounds,
        auto_analyze=not args.no_analyze,
    )
    runner.prepare_data()
    runner.run_loop()



