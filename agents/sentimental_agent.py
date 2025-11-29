# agents/sentimental_agent.py

from __future__ import annotations

import os
import json
from typing import Optional, Tuple, Dict, Any, List, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset

# BaseAgent
from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal

# 뉴스 병합 / 뉴스 기반 데이터셋
from core.sentimental_classes.news import merge_price_with_news_features
from core.sentimental_classes.pretrain_dataset_builder import build_pretrain_dataset

# LSTM 모델
from core.sentimental_classes.lstm_model import SentimentalLSTM

# dataset loader
from core.data_set import load_dataset, build_dataset

# 프롬프트
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS

from config.agents import agents_info, dir_info, common_params

CFG_S = agents_info["SentimentalAgent"]

FEATURE_COLS = [
    "return_1d",
    "hl_range",
    "Volume",
    "news_count_1d",
    "news_count_7d",
    "sentiment_mean_1d",
    "sentiment_mean_7d",
    "sentiment_vol_7d",
]

# config 값과 동기화
WINDOW_SIZE = CFG_S["window_size"]
HIDDEN_DIM = CFG_S.get("d_model", 64)   # d_model을 LSTM hidden_dim으로 재활용
NUM_LAYERS = CFG_S["num_layers"]
DROPOUT = CFG_S["dropout"]
# =============================================================================


class SentimentalAgent(BaseAgent):

    def __init__(self, ticker, agent_id="SentimentalAgent", news_dir=None, **kwargs):
        super().__init__(ticker=ticker, agent_id=agent_id, **kwargs)
        
        # news_dir 설정 (없으면 기본값: data_dir의 부모/raw/news)
        if news_dir is None:
            # data_dir이 "backtest/data/processed"면 "backtest/data/raw/news"
            # data_dir이 "data/processed"면 "data/raw/news"
            base_dir = os.path.dirname(self.data_dir)  # "backtest/data" or "data"
            news_dir = os.path.join(base_dir, "raw", "news")
        self.news_dir = news_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = agents_info[self.agent_id]

        # BaseAgent에서도 window_size를 세팅하지만, 여기서도 명시적으로 맞춰둠
        self.window_size = cfg["window_size"]

        # LSTM 구조 관련 하이퍼파라미터
        self.hidden_dim = HIDDEN_DIM          # = cfg.get("d_model", 64)
        self.num_layers = NUM_LAYERS          # = cfg["num_layers"]
        self.dropout = DROPOUT                # = cfg["dropout"]

        # 피처 목록 (실제는 FEATURE_COLS 기준)
        self.feature_cols = list(FEATURE_COLS)

        self.model = None
        self.model_loaded = False

        if not getattr(self, "ticker", None):
            self.ticker = ticker
        if not self.ticker:
            raise ValueError("SentimentalAgent: ticker is None/empty")
        self.ticker = str(self.ticker).upper()
        setattr(self, "symbol", self.ticker)


    # -------------------------------------------------------
    # PRETRAIN
    # -------------------------------------------------------
    def pretrain(self):
        """SentimentalAgent 사전학습 루틴 - data/raw CSV에서 직접 로드하여 scaling/window 처리"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]
        
        if not self.ticker:
            raise ValueError("SentimentalAgent.pretrain: ticker가 설정되지 않았습니다.")
        
        ticker = self.ticker
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")
        
        # 1) data/raw CSV 로드 (백테스팅 모드면 필터링된 임시 파일 우선 사용)
        raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
        raw_csv_path = os.path.join(raw_dir, f"{ticker}_{self.agent_id}_raw.csv")
        
        # 백테스팅 모드: 필터링된 임시 데이터셋 우선 사용
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            temp_dir = os.path.join(raw_dir, "backtest_temp")
            date_str = self.simulation_date.replace("-", "")
            temp_path = os.path.join(temp_dir, f"{ticker}_{self.agent_id}_raw_{date_str}.csv")
            if os.path.exists(temp_path):
                raw_csv_path = temp_path
                print(f"[INFO] 백테스팅 모드: 필터링된 데이터셋 사용 ({self.simulation_date} 이전)")
        
        if not os.path.exists(raw_csv_path):
            print(f"[{self.agent_id}] Raw CSV 파일이 없어 searcher() 실행 중...")
            _ = self.searcher(ticker, rebuild=True)
            raw_csv_path = os.path.join(raw_dir, f"{ticker}_{self.agent_id}_raw.csv")
            if not os.path.exists(raw_csv_path):
                raise FileNotFoundError(f"Raw CSV not found after searcher: {raw_csv_path}")
        
        # raw CSV 읽기 (Date 첫 컬럼, Close 마지막 컬럼)
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        # 2) 피처 컬럼 추출 (Date, Close 제외)
        feature_cols = list(FEATURE_COLS)
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        # 3) 타겟 생성 (다음날 수익률)
        close_prices = df_raw["Close"].values
        y_all = (close_prices[1:] / close_prices[:-1] - 1.0).reshape(-1, 1).astype(np.float32)
        X_all = X_all[:-1]  # 마지막 행 제외 (타겟이 없음)
        
        # 백테스팅 모드: 데이터 누수 방지 - sim_date 당일 수익률이 타겟에 포함되지 않도록
        # 마지막 타겟 제거 (sim_date-1 → sim_date 수익률이므로)
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            if len(y_all) > 0:
                # 마지막 타겟 제거 (sim_date 당일 수익률)
                y_all = y_all[:-1]
                X_all = X_all[:-1]
                print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용 중, 마지막 타겟 제거 (데이터 누수 방지)")
            else:
                print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용 중 (타겟 없음)")
        
        # 4) Window 처리 (시퀀스 생성)
        window_size = self.window_size
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        def _create_sequences(X, y, win: int):
            Xs, ys = [], []
            for i in range(len(X) - win):
                Xs.append(X[i : i + win])
                ys.append(y[i + win])
            return np.array(Xs), np.array(ys)
        
        X_seq, y_seq = _create_sequences(X_all, y_all, window_size)
        print(f"[INFO] 시퀀스 생성 완료: {X_seq.shape}, {y_seq.shape}")
        
        # 5) 타깃 스케일 조정
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_seq = y_seq * y_scale_factor
        
        # 6) Scaling (BaseAgent의 통합 스케일러 사용)
        self.scaler.fit_scalers(X_seq, y_seq)
        self.scaler.save(ticker)
        
        X_train, y_train = map(torch.tensor, self.scaler.transform(X_seq, y_seq))
        X_train, y_train = X_train.float(), y_train.float()
        
        # 7) 모델 생성 및 초기화
        if getattr(self, "model", None) is None:
            input_dim = X_seq.shape[-1]
            self.model = SentimentalLSTM(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        
        model = self.model
        model.train()
        
        # 8) 학습 준비
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loss 함수: config에서 가져오기
        cfg = agents_info.get(self.agent_id, {})
        loss_fn_name = cfg.get("loss_fn", "HuberLoss")
        if loss_fn_name == "HuberLoss":
            huber_delta = common_params.get("huber_loss_delta", 1.0)
            loss_fn = torch.nn.HuberLoss(delta=huber_delta)
        elif loss_fn_name == "L1Loss":
            loss_fn = torch.nn.L1Loss()
        elif loss_fn_name == "MSELoss":
            loss_fn = torch.nn.MSELoss()
        else:
            print(f"[WARN] 알 수 없는 loss_fn: {loss_fn_name}, HuberLoss 사용")
            huber_delta = common_params.get("huber_loss_delta", 1.0)
            loss_fn = torch.nn.HuberLoss(delta=huber_delta)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train.view(-1, 1)),
                                  batch_size=batch_size, shuffle=True)
        
        # 9) 학습 루프
        # 에포크 출력 주기 (config에서 가져오기)
        log_interval = common_params.get("pretrain_log_interval", 5)
        
        final_loss = None
        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            final_loss = avg_loss
            
            if (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
                print(f"  Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.6f}")
        
        # 10) 모델 저장
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        
        # model_loaded 플래그 설정
        self.model_loaded = True
        
        # 완료 메시지 출력
        final_loss_str = f" (Final Loss: {final_loss:.6f})" if final_loss is not None else ""
        print(f"✅ {self.agent_id} 모델 학습 및 저장 완료: {model_path}{final_loss_str}")
        
        # 11) 전처리된 데이터 저장 (config에서 설정)
        if common_params.get("pretrain_save_dataset", True):
            dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_id}_dataset.csv")
            flattened_data = []
            dates_list = df_raw["Date"].values[:-1]  # 마지막 제외
            
            # 스케일된 데이터는 X_train, y_train에서 가져오기
            X_scaled_np = X_train.cpu().numpy()
            y_scaled_np = y_train.cpu().numpy()
            
            for sample_idx in range(len(X_seq)):
                for time_idx in range(window_size):
                    date_idx = sample_idx + time_idx
                    row = {
                        'sample_id': sample_idx,
                        'time_step': time_idx,
                        'date': str(dates_list[date_idx]) if date_idx < len(dates_list) else None,
                        'target': float(y_scaled_np[sample_idx]) if time_idx == window_size - 1 else np.nan,
                    }
                    for feat_idx, feat_name in enumerate(feature_cols):
                        row[feat_name] = float(X_scaled_np[sample_idx, time_idx, feat_idx])
                    flattened_data.append(row)
            
            dataset_df = pd.DataFrame(flattened_data)
            os.makedirs(self.data_dir, exist_ok=True)
            dataset_df.to_csv(dataset_path, index=False)
            print(f"✅ 전처리된 데이터 저장 완료: {dataset_path}")

    # -------------------------------------------------------
    # _BUILD_MODEL
    # -------------------------------------------------------
    def _build_model(self) -> nn.Module:
        """BaseAgent.pretrain에서 사용할 LSTM 모델 생성"""

        # dataset 로드 또는 생성
        try:
            X, y, cols = load_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )
        except Exception:
            build_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )
            X, y, cols = load_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )

        # feature_cols 자동 업데이트
        self.feature_cols = list(cols)
        input_dim = X.shape[-1]

        model = SentimentalLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        return model

    # -------------------------------------------------------
    # RUN_DATASET
    # -------------------------------------------------------
    def run_dataset(self, days: int = None) -> StockData:
        """
        최근 days일치 가격 + 뉴스 피처를 기반으로
        FEATURE_COLS 입력(1, T, F)을 만들고 StockData를 생성
        """
        # common_params에서 period 값을 가져와서 일수로 변환
        if days is None:
            from config.agents import common_params
            # 백테스팅 모드면 period_test, 일반 모드면 period 사용
            if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
                period_str = common_params.get("period_test", "2y")
            else:
                period_str = common_params.get("period", "2y")
            
            # period 문자열을 일수로 변환
            if period_str.endswith("y"):
                years = int(period_str[:-1])
                days = years * 365
            elif period_str.endswith("m"):
                months = int(period_str[:-1])
                days = months * 30
            elif period_str.endswith("d"):
                days = int(period_str[:-1])
            else:
                # 기본값: 2년 (730일)
                days = 2 * 365
        
        # 0) 날짜 범위
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=days)

        # 1) 가격 데이터 (yfinance)
        df_price = yf.download(self.ticker, start=start, end=end)
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0].lower() for c in df_price.columns]
        else:
            df_price.columns = [c.lower() for c in df_price.columns]

        df_price = df_price.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

        df_price["date"] = df_price.index
        df_price = df_price.reset_index(drop=True)

        # 2) 뉴스 + 가격 병합
        df_merged = merge_price_with_news_features( 
            df_price=df_price,
            ticker=self.ticker,
            asof_kst=end.date(),
            base_dir=self.news_dir,
        )
        if isinstance(df_merged, tuple):
            df_feat = df_merged[0]
        else:
            df_feat = df_merged

        df_feat = df_feat.sort_values("date").reset_index(drop=True)

        # ---------------------------------------
        # FEATURE_COLS 검증 (merge_price_with_news_features에서 이미 생성됨)
        # ---------------------------------------
        required = list(FEATURE_COLS)
        missing = [c for c in required if c not in df_feat.columns]
        
        # 뉴스 피처가 없는 경우 기본값 설정
        for col in ["news_count_1d", "sentiment_mean_1d"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        
        # 최종 검증
        missing_after = [c for c in required if c not in df_feat.columns]
        if missing_after:
            raise ValueError(
                f"[SentimentalAgent.run_dataset] FEATURE_COLS 부족: {missing_after}"
            )

        # ---------------------------------------
        # 입력 행렬 생성
        # ---------------------------------------
        feat_values = df_feat[required].values.astype("float32")

        if len(feat_values) < self.window_size:
            raise ValueError(
                f"데이터 길이({len(feat_values)}) < 윈도우({self.window_size})"
            )

        X_last = feat_values[-self.window_size:]
        X_last = X_last[None, :, :]  # (1, T, F)
        self._last_input = X_last

        # ---------------------------------------
        # StockData 생성
        # ---------------------------------------
        last_row = df_feat.iloc[-1]
        last_price = float(last_row["close"])

        sd = StockData()
        sd.ticker = self.ticker
        sd.last_price = last_price
        sd.currency = "USD"
        sd.feature_cols = required
        sd.window_size = self.window_size
        sd.raw_df = df_feat

        sd.news_feats = {
            "news_count_7d": float(last_row.get("news_count_7d", 0)),
            "sentiment_mean_7d": float(last_row.get("sentiment_mean_7d", 0)),
            "sentiment_vol_7d": float(last_row.get("sentiment_vol_7d", 0)),
        }

        sd.snapshot = {
            "agent_id": self.agent_id,
            "feature_cols": sd.feature_cols,
            "window_size": sd.window_size,
            "news_feats": sd.news_feats,
            "raw_df": sd.raw_df,
        }

        sd.X_seq = X_last
        sd.SentimentalAgent = {
            "X_seq": X_last,
            "last_price": last_price,
        }

        self.stockdata = sd
        return sd

    # -------------------------------------------------------
    # 내부 헬퍼: _ensure_sentimental_csv
    #   - CSV 기반 캐싱 패턴 통일을 위한 데이터 수집 메서드
    # -------------------------------------------------------
    def _ensure_sentimental_csv(self, ticker: str, rebuild: bool = False) -> None:
        """
        SentimentalAgent 전용 CSV 생성 메서드
        - 주가 데이터 + 뉴스 데이터 수집 및 병합
        - CSV 저장 (Date 첫 컬럼, Close 마지막 컬럼)
        """
        raw_csv_path = os.path.join(os.path.dirname(self.data_dir), "raw", f"{ticker}_{self.agent_id}_raw.csv")
        
        if not rebuild and os.path.exists(raw_csv_path):
            return
        
        if not os.path.exists(raw_csv_path):
            print(f"[{self.agent_id}] Raw CSV 파일이 없어 생성 중...")
        else:
            print(f"[{self.agent_id}] Rebuild 요청됨. Raw CSV 재생성 중...")
        
        # common_params에서 period 가져오기
        period_str = common_params.get("period", "2y")
        # period 문자열을 일수로 변환
        if period_str.endswith("y"):
            years = int(period_str[:-1])
            days = years * 365
        elif period_str.endswith("m"):
            months = int(period_str[:-1])
            days = months * 30
        elif period_str.endswith("d"):
            days = int(period_str[:-1])
        else:
            days = 2 * 365  # 기본값
        
        # 1) 날짜 범위
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=days)

        # 2) 가격 데이터 (yfinance)
        df_price = yf.download(ticker, start=start, end=end)
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0].lower() for c in df_price.columns]
        else:
            df_price.columns = [c.lower() for c in df_price.columns]

        df_price = df_price.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

        df_price["date"] = df_price.index
        df_price = df_price.reset_index(drop=True)

        # 3) 뉴스 + 가격 병합
        from core.sentimental_classes.news import update_news_db
        
        # df_price의 시작 날짜를 확인하여 뉴스 수집 기간에 반영
        if not df_price.empty:
            if 'date' in df_price.columns:
                price_start_date = pd.to_datetime(df_price['date']).min()
            else:
                price_start_date = df_price.index.min()
        else:
            price_start_date = None

        # 뉴스 DB 가져오기 (없으면 생성)
        df_news = update_news_db(ticker, base_dir=self.news_dir, target_start_date=price_start_date)
        
        if df_news.empty:
            print(f"[WARN] {ticker} 뉴스 데이터가 없습니다. 감성 피처를 0으로 채웁니다.")
            df_merged = df_price.copy()
            df_merged["news_count_7d"] = 0
            df_merged["sentiment_mean_7d"] = 0.0
            df_merged["sentiment_vol_7d"] = 0.0
            df_merged["news_count_1d"] = 0
            df_merged["sentiment_mean_1d"] = 0.0
        else:
            # 일별 집계
            daily_stats = df_news.groupby('date').agg(
                count=('sentiment_score', 'count'),
                mean=('sentiment_score', 'mean')
            )
            
            # 날짜 인덱스 채우기 (뉴스가 없는 날은 0)
            idx = pd.date_range(daily_stats.index.min(), daily_stats.index.max())
            daily_stats = daily_stats.reindex(idx, fill_value=0)
            daily_stats.index.name = 'date'
            
            daily_stats['news_count_1d'] = daily_stats['count']
            daily_stats['sentiment_mean_1d'] = daily_stats['mean']
            
            # Rolling 7d
            daily_stats['news_count_7d'] = daily_stats['count'].rolling(7).sum().fillna(0)
            daily_stats['sentiment_mean_7d'] = daily_stats['mean'].rolling(7).mean().fillna(0)
            daily_stats['sentiment_vol_7d'] = daily_stats['mean'].rolling(7).std().fillna(0)
            
            daily_stats = daily_stats.reset_index()  # date 컬럼 복원
            
            # 주가 데이터와 병합
            df_price_copy = df_price.copy()
            if 'date' not in df_price_copy.columns:
                df_price_copy['date'] = df_price_copy.index
            
            # df_price의 date는 datetime일 수도 있고 string일 수도 있음. 통일 및 TZ 제거
            df_price_copy['date'] = pd.to_datetime(df_price_copy['date'])
            if df_price_copy['date'].dt.tz is not None:
                df_price_copy['date'] = df_price_copy['date'].dt.tz_localize(None)
                
            # daily_stats의 date도 TZ 제거
            if daily_stats['date'].dt.tz is not None:
                daily_stats['date'] = daily_stats['date'].dt.tz_localize(None)
            
            # Left Join
            df_merged = pd.merge(df_price_copy, daily_stats, on='date', how='left')
            
            # 결측치 처리 (뉴스가 없었던 날)
            fill_cols = ['news_count_1d', 'sentiment_mean_1d', 'news_count_7d', 'sentiment_mean_7d', 'sentiment_vol_7d']
            for col in fill_cols:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col].fillna(0)

        df_feat = df_merged.sort_values("date").reset_index(drop=True)

        # 4) FEATURE_COLS 자동 보정
        required = list(FEATURE_COLS)
        print("[SentimentalAgent._ensure_sentimental_csv] missing(before):",
              [c for c in required if c not in df_feat.columns])

        # return_1d
        if "return_1d" not in df_feat.columns:
            df_feat["return_1d"] = df_feat["close"].pct_change().fillna(0)

        # hl_range
        if "hl_range" not in df_feat.columns:
            df_feat["hl_range"] = ((df_feat["high"] - df_feat["low"]) /
                                   df_feat["close"].replace(0, np.nan)).fillna(0)

        # Volume (대문자)
        if "Volume" not in df_feat.columns:
            df_feat["Volume"] = df_feat["volume"].fillna(0)

        # 뉴스 1일 feature (없으면 0)
        for col in ["news_count_1d", "sentiment_mean_1d"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        # 마지막 검증
        missing_after = [c for c in required if c not in df_feat.columns]
        if missing_after:
            raise ValueError(
                f"[SentimentalAgent._ensure_sentimental_csv] FEATURE_COLS 부족: {missing_after}"
            )

        print("[SentimentalAgent._ensure_sentimental_csv] all FEATURE_COLS present.")

        # 5) Raw CSV 저장 (Date 첫 컬럼, Close 마지막 컬럼)
        try:
            os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
            df_raw = df_feat.copy()
            if "date" in df_raw.columns:
                df_raw = df_raw.rename(columns={"date": "Date"})
            if "close" in df_raw.columns:
                df_raw = df_raw.rename(columns={"close": "Close"})

            # 저장할 피처 구성: Date + FEATURE_COLS + Close
            cols_to_save = []
            if "Date" in df_raw.columns:
                cols_to_save.append("Date")

            # FEATURE_COLS 순서를 유지하면서 존재하는 것만 추가
            for col in FEATURE_COLS:
                if col in df_raw.columns and col not in ("Date", "Close"):
                    cols_to_save.append(col)

            # 마지막에 Close 컬럼 추가
            if "Close" in df_raw.columns:
                cols_to_save.append("Close")

            # period에 맞춰 데이터 필터링 (다른 에이전트와 시작일자 통일)
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            end_date = pd.Timestamp.today().normalize()
            # period_str을 일수로 변환 (이미 위에서 계산됨)
            start_date = end - pd.Timedelta(days=days)  # 위에서 계산한 start_date 사용
            
            # period 기간에 맞춰 필터링
            df_raw = df_raw[df_raw["Date"] >= start_date].copy()
            df_raw = df_raw.sort_values("Date").reset_index(drop=True)
            df_raw["Date"] = df_raw["Date"].dt.strftime("%Y-%m-%d")

            df_raw[cols_to_save].to_csv(raw_csv_path, index=False)
            print(f"✅ [{self.agent_id}] Raw CSV 저장 완료: {raw_csv_path} ({len(df_raw):,} rows, period: {period_str})")
        except Exception as e:
            print(f"❌ [{self.agent_id}] Raw CSV 저장 실패: {e}")

    # -------------------------------------------------------
    # searcher (통일된 CSV 기반 캐싱 패턴)
    # -------------------------------------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """SentimentalAgent 전용 searcher - CSV 기반 캐싱 패턴 (다른 에이전트와 통일)"""
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError(f"{agent_id}: ticker가 지정되지 않았습니다.")
        
        self.ticker = str(ticker).upper()
        
        raw_csv_path = os.path.join(os.path.dirname(self.data_dir), "raw", f"{ticker}_{agent_id}_raw.csv")
        cfg = agents_info.get(agent_id, {})
        
        # 1) Raw CSV 보장 (데이터 수집/전처리)
        self._ensure_sentimental_csv(ticker, rebuild=rebuild)
        
        # 2) Raw CSV에서 최신 window_size만큼 직접 추출
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
        
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        feature_cols = list(FEATURE_COLS)
        window_size = cfg.get("window_size", self.window_size)
        
        # 피처 추출 (Date, Close 제외)
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        # 최신 window_size만큼 추출
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        X_latest = X_all[-window_size:].reshape(1, window_size, -1)  # (1, T, F)
        
        print(f"✅ [{agent_id}] Searcher 완료: 윈도우 shape {X_latest.shape}")
        
        # StockData 구성
        self.stockdata = StockData(ticker=ticker)
        self.stockdata.feature_cols = feature_cols
        self.stockdata.window_size = window_size
        
        # last_price (CSV의 마지막 Close 값 사용)
        try:
            self.stockdata.last_price = float(df_raw["Close"].iloc[-1])
        except Exception:
            self.stockdata.last_price = None

        # 통화코드
        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception:
            self.stockdata.currency = "USD"

        # X_seq 설정 (predict()에서 필요)
        self.stockdata.X_seq = X_latest  # (1, T, F) 형태
        
        # feature_dict (마지막 윈도우)
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)
        
        # 뉴스 피처 정보 저장 (선택적)
        if len(df_raw) > 0:
            last_row = df_raw.iloc[-1]
            self.stockdata.news_feats = {
                "news_count_7d": float(last_row.get("news_count_7d", 0)),
                "sentiment_mean_7d": float(last_row.get("sentiment_mean_7d", 0)),
                "sentiment_vol_7d": float(last_row.get("sentiment_vol_7d", 0)),
            }
            self.stockdata.raw_df = df_raw

        return torch.tensor(X_latest, dtype=torch.float32)

    # -------------------------------------------------------
    # predict
    #   - Monte Carlo Dropout + DataScaler + y*100 스케일 고려
    # -------------------------------------------------------
    def predict(self, X, n_samples: Optional[int] = None, current_price: Optional[float] = None):
        """
        SentimentalAgent 전용 Monte Carlo Dropout 예측 함수

        - 입력: StockData 또는 (T, F) / (1, T, F) numpy/tensor
        - self.model(SentimentalLSTM) + self.scaler 로드
        - MC Dropout으로 예측 분포 샘플링
        - "수익률 * 100" → 실제 가격(next_close)로 변환
        - Target(next_close, uncertainty, confidence) 반환
        """
        # n_samples 설정 (config에서 가져오기)
        if n_samples is None:
            n_samples = common_params.get("n_samples", 30)
        
        # -----------------------------
        # 0) 입력 정리 (StockData 래핑) - 통일된 처리
        # -----------------------------
        if isinstance(X, StockData):
            sd = X
            X_in = getattr(sd, "X_seq", None)
            if X_in is None:
                # StockData에 X_seq가 없으면 agent_id로 찾기 (다른 에이전트와 통일)
                X_in = getattr(sd, self.agent_id, None)
                if isinstance(X_in, dict):
                    # dict 형태면 DataFrame으로 변환
                    df = pd.DataFrame(X_in)
                    X_in = df.values
            if X_in is None:
                raise ValueError(f"StockData에 {self.agent_id} 데이터가 없습니다. searcher()를 먼저 호출하세요.")
            if current_price is None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
        else:
            sd = None
            X_in = X

        if X_in is None:
            raise ValueError("predict()에 전달된 입력 X가 None 입니다.")

        # numpy / tensor 로 통일
        if isinstance(X_in, np.ndarray):
            X_raw_np = X_in.copy()
        elif isinstance(X_in, torch.Tensor):
            X_raw_np = X_in.detach().cpu().numpy().copy()
        else:
            raise TypeError(f"Unsupported input type for predict: {type(X_in)}")

        # -----------------------------
        # 1) 모델 준비 (BaseAgent.load_model() 사용)
        # -----------------------------
        if not self.ticker:
            raise ValueError("ticker가 설정되지 않았습니다. 먼저 searcher(ticker)를 호출하세요.")
        
        # 모델 파일 확인
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        if not os.path.exists(model_path):
            print(f"[{self.agent_id}] 모델이 없어 pretrain()을 실행합니다...")
            self.pretrain()
        else:
            # 모델이 있으면 로드 (이미 로드되었는지 확인)
            if not hasattr(self, "model_loaded") or not self.model_loaded:
                # 모델이 없으면 생성
                if getattr(self, "model", None) is None:
                    self.model = SentimentalLSTM(
                        input_dim=len(FEATURE_COLS),
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                    )
                self.load_model(model_path)
        
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError(f"{self.agent_id} 모델이 초기화되지 않음")

        # -----------------------------
        # 1-1) 스케일러 로드
        # -----------------------------
        if not hasattr(self, "scaler"):
            raise RuntimeError("[SentimentalAgent] self.scaler가 정의되지 않았습니다.")
        self.scaler.load(self.ticker)

        # -----------------------------
        # 2) 입력 형태 정규화 및 스케일링 (통일된 처리)
        # -----------------------------
        # (T, F) → (1, T, F)로 정규화 (스케일링 전에 차원 통일)
        if X_raw_np.ndim == 2:
            X_raw_np = X_raw_np[None, :, :]  # (T, F) → (1, T, F)
        elif X_raw_np.ndim == 3 and X_raw_np.shape[0] != 1:
            raise ValueError(f"예상하지 못한 배치 크기: {X_raw_np.shape[0]}, (1, T, F) 형태만 지원합니다.")
        
        # (1, T, F) 형태로 스케일링
        X_scaled, _ = self.scaler.transform(X_raw_np)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        device = getattr(self, "device", torch.device("cpu"))
        X_tensor = X_tensor.to(device)
        model.to(device)

        # -----------------------------
        # 3) Monte Carlo Dropout 추론
        # -----------------------------
        model.train()  # dropout 활성화 (MC Dropout)
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor)   # 예: (1, seq_len) 또는 (1, 1)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                preds.append(y_pred.detach().cpu().numpy().flatten())

        preds = np.stack(preds)           # (samples, L)
        mean_pred = preds.mean(axis=0)    # (L,)
        std_pred = np.abs(preds.std(axis=0))

        # -----------------------------
        # 4) σ 기반 confidence 계산
        # -----------------------------
        sigma = float(std_pred[-1])
        sigma_min = common_params.get("sigma_min", 1e-6)
        sigma = max(sigma, sigma_min)
        confidence = float(1.0 / (1.0 + np.log1p(sigma)))

        # -----------------------------
        # 5) y_scaler 역변환 + 수익률 → 가격 변환
        # -----------------------------
        # 모델 출력이 "수익률 * 100" 형태라고 가정 (예: 3.5 → +3.5%)
        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        # config에서 스케일 팩터 가져오기
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        predicted_return = float(mean_pred[-1]) / y_scale_factor  # 3.5 → 0.035
        
        # 수익률 클리핑 (agents_info에서 가져오기)
        cfg = agents_info.get(self.agent_id, {})
        return_clip_min = cfg.get("return_clip_min", -0.5)
        return_clip_max = cfg.get("return_clip_max", 0.5)
        predicted_return_raw = predicted_return
        predicted_return = np.clip(predicted_return, return_clip_min, return_clip_max)

        # current_price 추론
        if current_price is None:
            if sd is not None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
            else:
                default_price = common_params.get("default_current_price", 100.0)
                current_price = float(getattr(self, "last_price", default_price))

        predicted_price = float(current_price * (1.0 + predicted_return))

        # -----------------------------
        # 6) Target 생성 및 저장
        # -----------------------------
        target = Target(
            next_close=predicted_price,
            uncertainty=sigma,
            confidence=confidence,
        )
        
        # 통일된 예측 결과 로그 출력 (불필요한 로그 제거)
        # clipped_info = f" (클리핑: {predicted_return_raw:.4f} → {predicted_return:.4f})" if predicted_return_raw != predicted_return else ""
        # print(f"[{self.agent_id}] Predict 완료: next_close={predicted_price:.2f}, return={predicted_return*100:.2f}%{clipped_info}, uncertainty={sigma:.4f}, confidence={confidence:.4f}")

        if hasattr(self, "targets"):
            self.targets.append(target)

        return target

    # -------------------------------------------------------
    # 내부 helper: _predict_next_close
    #   - run_dataset → self.predict(StockData) 조합으로 사용
    # -------------------------------------------------------
    @torch.inference_mode()
    def _predict_next_close(self) -> Tuple[float, float, float, List[str]]:
        """
        run_dataset() 결과 또는 이미 계산된 self.stockdata를 이용해
        다음날 종가 / 불확실성 / 신뢰도를 얻는다.
        """
        sd = getattr(self, "stockdata", None)
        if sd is None or getattr(sd, "X_seq", None) is None:
            cfg = agents_info.get(self.agent_id, {})
            # common_params에서 period 가져오기 (파일 상단에서 이미 import됨)
            period_str = common_params.get("period", "2y")
            # period 문자열을 일수로 변환
            if period_str.endswith("y"):
                years = int(period_str[:-1])
                days = years * 365
            elif period_str.endswith("m"):
                months = int(period_str[:-1])
                days = months * 30
            elif period_str.endswith("d"):
                days = int(period_str[:-1])
            else:
                days = 2 * 365  # 기본값
            sd = self.run_dataset(days=days)

        # config에서 n_samples 가져오기 (파일 상단에서 이미 import됨)
        n_samples = common_params.get("n_samples", 30)
        target = self.predict(sd, n_samples=n_samples)
        cols = list(getattr(sd, "feature_cols", self.feature_cols))
        return float(target.next_close), float(target.uncertainty or 0.0), float(target.confidence or 0.0), cols

    # -------------------------------------------------------
    # ctx 구성 (run_dataset의 news_feats 사용)
    # -------------------------------------------------------
    def build_ctx(self, asof_date_kst: Optional[str] = None) -> Dict[str, Any]:
        # 0) StockData 확보
        stockdata: StockData | None = getattr(self, "stockdata", None)
        if stockdata is None or getattr(stockdata, "X_seq", None) is None:
            stockdata = self.run_dataset()

        # 1) 기준 날짜(asof_date_kst)
        if asof_date_kst is None:
            asof_date_kst = datetime.now().strftime("%Y-%m-%d")

        # 2) 예측 값
        pred_close, uncertainty_std, confidence, cols = self._predict_next_close()

        # 3) 가격 스냅샷 (raw_df 마지막 행 기준)
        price_snapshot: Dict[str, Optional[float]] = {}
        df = getattr(stockdata, "raw_df", None)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            last = df.iloc[-1]
            price_snapshot["Close"] = float(last.get("close", np.nan))
            price_snapshot["Open"] = float(last.get("open", np.nan))
            price_snapshot["High"] = float(last.get("high", np.nan))
            price_snapshot["Low"] = float(last.get("low", np.nan))
            price_snapshot["Volume"] = float(last.get("volume", np.nan))
        else:
            price_snapshot = {
                "Close": getattr(stockdata, "last_price", np.nan),
                "Open": None,
                "High": None,
                "Low": None,
                "Volume": None,
            }

        # 4) 뉴스/감성 피처: run_dataset()에서 저장한 news_feats 사용
        nf = getattr(stockdata, "news_feats", {}) or {}
        news_count_7d = float(nf.get("news_count_7d", 0.0))
        sentiment_mean_7d = float(nf.get("sentiment_mean_7d", 0.0))
        sentiment_vol_7d = float(nf.get("sentiment_vol_7d", 0.0))

        sentiment_summary = {
            "mean_7d": sentiment_mean_7d,
            "mean_30d": 0.0,          # 아직은 30일 피처 없으니 0으로
            "pos_ratio_7d": 0.0,
            "neg_ratio_7d": 0.0,
        }
        sentiment_vol = {"vol_7d": sentiment_vol_7d}
        news_count = {"count_7d": int(news_count_7d)}
        trend_7d = 0.0
        has_news = bool(news_count_7d > 0)

        # 5) snapshot / prediction 구성
        last_price = price_snapshot.get("Close", np.nan)
        if last_price and last_price == last_price:
            pred_return = float(pred_close / last_price - 1.0)
        else:
            pred_return = None

        snapshot = {
            "asof_date": asof_date_kst,
            "last_price": last_price,
            "currency": getattr(stockdata, "currency", "USD"),
            "window_size": self.window_size,
            "feature_cols_preview": [c for c in (cols or [])[:8]],
        }

        feature_importance = {
            "sentiment_score": sentiment_summary.get("mean_7d", 0.0),
            "sentiment_summary": sentiment_summary,
            "sentiment_volatility": sentiment_vol,
            "trend_7d": trend_7d,
            "news_count": news_count,
            "has_news": has_news,
            "price_snapshot": {
                "Close": price_snapshot.get("Close"),
                "Open": price_snapshot.get("Open"),
                "High": price_snapshot.get("High"),
                "Low": price_snapshot.get("Low"),
                "Volume": price_snapshot.get("Volume"),
                "ret_1d": None,
                "ret_5d": None,
                "ret_20d": None,
                "zscore_20d": None,
                "vol_change_5d": None,
            },
        }

        ctx = {
            "agent_id": self.agent_id,
            "ticker": self.ticker,
            "snapshot": snapshot,
            "prediction": {
                "pred_close": pred_close,
                "pred_return": pred_return,
                "uncertainty": {
                    "std": uncertainty_std,
                    "ci95": float(1.96 * uncertainty_std),
                },
                "confidence": confidence,
                "pred_next_close": pred_close,
            },
            "feature_importance": feature_importance,
        }
        return ctx

    def reviewer_rebuttal(
        self,
        my_opinion: Opinion,
        other_opinion: Opinion,
        round_index: int,
    ) -> Rebuttal:

        return self.reviewer_rebut(
            my_opinion=my_opinion,
            other_opinion=other_opinion,
            round=round_index,
        )

    # -------------------------------------------------------
    # Opinion / Rebuttal / Revision 프롬프트
    # -------------------------------------------------------
    def _build_messages_opinion(
        self,
        stock_data: StockData,
        target: Target,
    ) -> Tuple[str, str]:
        if stock_data is None:
            stock_data = self.stockdata

        # 공통 ctx 사용
        ctx = self.build_ctx()
        
        # [New] 최근 뉴스 헤드라인 조회 및 추가 (XAI용)
        news_summary = []
        try:
            db_path = os.path.join(self.news_dir, f"{self.ticker}_news_db.csv")
            if os.path.exists(db_path):
                df_news = pd.read_csv(db_path)
                df_news['date'] = pd.to_datetime(df_news['date'])
                # 타임존 정보가 있을 경우 제거하여 비교 오류 방지
                if df_news['date'].dt.tz is not None:
                    df_news['date'] = df_news['date'].dt.tz_localize(None)
                
                # 최근 7일 뉴스 필터링
                asof_date_str = ctx['snapshot']['asof_date']
                last_date = pd.to_datetime(asof_date_str)
                start_date = last_date - pd.Timedelta(days=7)
                
                recent_news = df_news[(df_news['date'] >= start_date) & (df_news['date'] <= last_date)]
                
                if not recent_news.empty:
                    # 감성 점수가 극단적인 뉴스 위주로 5개 선정
                    recent_news = recent_news.copy()
                    recent_news['abs_score'] = recent_news['sentiment_score'].abs()
                    top_news = recent_news.sort_values('abs_score', ascending=False).head(5)
                    
                    for _, row in top_news.iterrows():
                        date_str = row['date'].strftime('%Y-%m-%d')
                        title = str(row['title'])
                        label = str(row['sentiment_label'])
                        score = float(row['sentiment_score'])
                        news_summary.append(f"- {date_str}: {title} ({label}, {score:.2f})")
        except Exception as e:
            print(f"[WARN] 뉴스 요약 생성 실패: {e}")
            
        ctx['recent_news_headlines'] = news_summary if news_summary else ["(최근 7일간 주요 뉴스 없음)"]

        # DebateAgent에서 target이 업데이트됐을 수 있으므로 반영
        ctx["prediction"]["pred_next_close"] = float(getattr(target, "next_close", 0.0))
        ctx["prediction"]["pred_close"] = ctx["prediction"]["pred_next_close"]

        last_close = ctx["snapshot"].get("last_price")
        if isinstance(last_close, (int, float)) and last_close not in (0, None):
            try:
                chg = ctx["prediction"]["pred_next_close"] / float(last_close) - 1.0
            except ZeroDivisionError:
                chg = None
        else:
            chg = None
        ctx["prediction"]["pred_return"] = chg

        ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
        prompts = OPINION_PROMPTS["SentimentalAgent"]
        system_text = prompts["system"]
        user_tmpl = prompts["user"]

        try:
            user_text = user_tmpl.format(context=ctx_json)
        except KeyError:
            user_text = user_tmpl.replace("{context}", ctx_json)

        return system_text, user_text

    def _build_messages_rebuttal(
        self,
        my_opinion: Opinion,
        target_opinion: Opinion,
        stock_data: StockData,
    ) -> Tuple[str, str]:
        opp_agent = getattr(target_opinion, "agent_id", "UnknownAgent")
        opp_reason = getattr(target_opinion, "reason", "")

        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        pred_close = float(my_opinion.target.next_close)
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        system_tmpl = None
        user_tmpl = None
        if REBUTTAL_PROMPTS and "SentimentalAgent" in REBUTTAL_PROMPTS:
            pp = REBUTTAL_PROMPTS["SentimentalAgent"]
            system_tmpl = pp.get("system")
            user_tmpl = pp.get("user")

        if not system_tmpl:
            system_tmpl = (
                "당신은 감성 기반 단기 주가 분석가로서 상대 의견의 논리적/수치적 허점을 분석해 반박합니다. "
                "감성지표(평균, 추세, 변동성)와 뉴스 개수, 예측의 불확실성을 근거로 삼되, "
                "합리적 포인트는 인정하고 핵심 쟁점 위주로 간결히 반박하세요."
            )

        if not user_tmpl:
            user_tmpl = (
                "티커: {ticker}\n"
                "상대 에이전트: {opp_agent}\n"
                "상대 의견:\n{opp_reason}\n\n"
                "우리 예측:\n- next_close: {pred_close}\n- 예상 변화율(현재가 대비): {chg}\n"
                "감성 근거:\n- mean7={mean7}, mean30={mean30}, pos7={pos7}, neg7={neg7}\n"
                "- vol7={vol7}, trend7={trend7}, news7={news7}\n\n"
                "요청: 위 정보를 바탕으로 상대 의견의 약점 2~4개를 조목조목 반박하세요. "
                "특히 감성 추세/변동성, 뉴스 수의 맥락, 예측 불확실성(높/낮음)이 "
                "상대 주장과 어떻게 상충/보완되는지 구체적으로 지적하세요."
            )

        user_text = user_tmpl.format(
            ticker=self.ticker,
            opp_agent=opp_agent,
            opp_reason=opp_reason if opp_reason else "(상대 의견 내용 없음)",
            pred_close=f"{pred_close:.4f}",
            chg=("NA" if change_ratio is None else f"{change_ratio*100:.2f}%"),
            mean7=f"{sent.get('mean_7d', 0.0):.4f}",
            mean30=f"{sent.get('mean_30d', 0.0):.4f}",
            pos7=f"{sent.get('pos_ratio_7d', 0.0):.4f}",
            neg7=f"{sent.get('neg_ratio_7d', 0.0):.4f}",
            vol7=("NA" if vol7 is None else f"{vol7:.4f}"),
            trend7=("NA" if trend7 is None else f"{trend7:.4f}"),
            news7=("NA" if news7 is None else f"{news7}"),
        )
        return system_tmpl, user_text

    def _build_messages_revision(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: Optional[List[Rebuttal]] = None,
        stock_data: StockData = None,
    ) -> Tuple[str, str]:
        if stock_data is None:
            stock_data = self.stockdata

        def _op_text(x: Union[Opinion, Dict[str, Any], str, None, Any]) -> str:
            if isinstance(x, Opinion):
                return getattr(x, "reason", "")
            if isinstance(x, dict):
                return x.get("reason", "") or x.get("message", "")
            if hasattr(x, "message"):
                return getattr(x, "message", "")
            if hasattr(x, "reason"):
                return getattr(x, "reason", "")
            return str(x) if x else ""

        prev_reason = _op_text(my_opinion)

        reb_texts: List[str] = []
        if isinstance(rebuttals, list):
            for r in rebuttals:
                reb_texts.append(_op_text(r))
        elif rebuttals is not None:
            reb_texts.append(_op_text(rebuttals))

        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        pred_info = ctx.get("prediction", {}) or {}
        unc_dict = pred_info.get("uncertainty", {}) or {}
        unc_std = unc_dict.get("std", None)
        confidence = pred_info.get("confidence", None)

        pred_close = float(my_opinion.target.next_close)
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        context_parts: List[str] = []
        if last_price is not None:
            if change_ratio is not None:
                context_parts.append(
                    f"현재 주가는 {last_price:.2f}이고, 모델은 다음 거래일 종가를 {pred_close:.2f}로 예측했습니다 "
                    f"(변화율 약 {change_ratio*100:.2f}%)."
                )
            else:
                context_parts.append(
                    f"현재 주가는 {last_price:.2f}이며, 다음 거래일 종가 예측값은 {pred_close:.2f}입니다."
                )
        else:
            context_parts.append(
                f"다음 거래일 종가 예측값은 {pred_close:.2f}입니다."
            )

        mean7_val = sent.get('mean_7d', None)
        mean30_val = sent.get('mean_30d', None)
        pos7_val = sent.get('pos_ratio_7d', None)
        neg7_val = sent.get('neg_ratio_7d', None)

        if mean7_val is not None and mean30_val is not None:
            try:
                mean7_val = float(mean7_val)
                mean30_val = float(mean30_val)
                context_parts.append(
                    f"최근 7일 평균 감성 점수는 {mean7_val:.3f}, 최근 30일 평균은 {mean30_val:.3f}입니다."
                )
            except (ValueError, TypeError):
                pass
        if pos7_val is not None and neg7_val is not None:
            try:
                pos7_val = float(pos7_val)
                neg7_val = float(neg7_val)
                context_parts.append(
                    f"최근 7일 기준 긍정 기사 비율은 {pos7_val:.2%}, 부정 기사 비율은 {neg7_val:.2%}입니다."
                )
            except (ValueError, TypeError):
                pass
        if vol7 is not None:
            context_parts.append(
                f"최근 7일 감성 점수의 변동성(표준편차)은 {vol7:.3f}입니다."
            )
        if trend7 is not None:
            context_parts.append(
                f"최근 7일 감성 추세(회귀 기울기)는 {trend7:.4f}입니다."
            )
        if news7 is not None:
            context_parts.append(
                f"최근 7일 동안 수집된 뉴스 개수는 {news7}건입니다."
            )

        if unc_std is not None and confidence is not None:
            context_parts.append(
                f"예측 표준편차는 {unc_std:.4f}, 신뢰도는 {confidence:.3f}입니다."
            )

        context_str = " ".join(context_parts) if context_parts else (
            "최근 뉴스 감성 점수, 변동성, 긍·부정 비율, 뉴스 수, 예측 불확실성 등을 종합해 단기 주가를 해석합니다."
        )

        system_tmpl = None
        user_tmpl = None
        if REVISION_PROMPTS and "SentimentalAgent" in REVISION_PROMPTS:
            pp = REVISION_PROMPTS["SentimentalAgent"]
            system_tmpl = pp.get("system")
            user_tmpl = pp.get("user")

        if not system_tmpl:
            system_tmpl = (
                "당신은 감성 기반 단기 주가 분석가입니다. "
                "초안 의견과 반박들을 검토해 핵심만 남기고, 데이터에 근거해 결론을 더 명확히 다듬습니다. "
                "불확실성/신뢰도 해석을 포함하여 한 단계 더 견고한 최종 의견으로 수정하세요."
            )

        if not user_tmpl:
            user_tmpl = (
                "티커: {ticker}\n"
                "초안 의견:\n{prev}\n\n"
                "수신한 반박 요약:\n{rebuts}\n\n"
                "업데이트된 수치:\n- next_close: {pred_close}\n- 예상 변화율: {chg}\n"
                "감성 근거 스냅샷:\n- mean7={mean7}, mean30={mean30}, pos7={pos7}, neg7={neg7}\n"
                "- vol7={vol7}, trend7={trend7}, news7={news7}\n\n"
                "추가 컨텍스트:\n{context}\n\n"
                "요청: 초안의 과장/중복/약한 근거를 정리하고, 강한 근거(감성 추세, 변동성, 뉴스 수 변화)를 중심으로 "
                "최종 의견을 3~5문장으로 재작성하세요. 불확실성/신뢰도 해석을 포함하세요."
            )

        rebuts_joined = "- " + "\n- ".join(
            [s for s in reb_texts if s]
        ) if reb_texts else "(반박 없음)"

        user_text = user_tmpl.format(
            ticker=self.ticker,
            prev=prev_reason if prev_reason else "(초안 없음)",
            rebuts=rebuts_joined,
            pred_close=f"{pred_close:.4f}",
            chg=("NA" if change_ratio is None else f"{change_ratio*100:.2f}%"),
            mean7=("NA" if mean7_val is None else f"{float(mean7_val):.4f}"),
            mean30=("NA" if mean30_val is None else f"{float(mean30_val):.4f}"),
            pos7=("NA" if pos7_val is None else f"{float(pos7_val):.4f}"),
            neg7=("NA" if neg7_val is None else f"{float(neg7_val):.4f}"),
            vol7=("NA" if vol7 is None else f"{float(vol7):.4f}"),
            trend7=("NA" if trend7 is None else f"{float(trend7):.4f}"),
            news7=("NA" if news7 is None else f"{news7}"),
            context=context_str,
        )
        return system_tmpl, user_text
        
    def reviewer_revise(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: Optional[List[Rebuttal]] = None,
        stock_data: StockData = None,
    ) -> Opinion:
        # SentimentalAgent는 BaseAgent의 revise 로직을 사용하되,
        # revise된 예측값을 유지하도록 수정 (기존에는 원래 값으로 되돌렸음)
        revised = super().reviewer_revise(
            my_opinion=my_opinion,
            others=others,
            rebuttals=rebuttals,
            stock_data=stock_data,
        )

        # revise된 값이 유효한 경우 유지 (None이거나 0에 가까운 값이 아닌 경우)
        try:
            if revised is not None and hasattr(revised, "target") and revised.target is not None:
                revised_close = revised.target.next_close
                # revise된 값이 비정상적으로 작거나 None인 경우에만 원래 값으로 복원
                if revised_close is None or revised_close < 10.0:
                    print(f"[SentimentalAgent] revise된 값({revised_close})이 비정상적이어서 원래 값({my_opinion.target.next_close})으로 복원")
                    revised.target.next_close = my_opinion.target.next_close
                # 그 외에는 revise된 값을 유지하여 다양성 확보
        except Exception as e:
            print(f"[SentimentalAgent] reviewer_revise post-fix 실패: {e}")

        return revised

    # -------------------------------------------------------
    # 레거시 get_opinion (단독 테스트용)
    # -------------------------------------------------------
    def get_opinion(self, idx: int = 0, ticker: Optional[str] = None) -> Opinion:
        if ticker and ticker != self.ticker:
            self.ticker = str(ticker).upper()

        pred_close, uncertainty_std, confidence, _ = self._predict_next_close()
        target = Target(
            next_close=float(pred_close),
            uncertainty=float(uncertainty_std),
            confidence=float(confidence),
        )

        try:
            if hasattr(self, "reviewer_draft"):
                op = self.reviewer_draft(getattr(self, "stockdata", None), target)
                return op
        except Exception as e:
            print("[SentimentalAgent] reviewer_draft 사용 실패:", e)

        ctx = self.build_ctx()
        fi = ctx["feature_importance"]
        sent = fi["sentiment_summary"]

        reason = (
            f"{self.ticker}의 최근 7일 감성 평균은 {sent['mean_7d']:.3f}이며 "
            f"뉴스 개수(7d)는 {fi['news_count']['count_7d']}건입니다. "
            f"감성 변동성(vol_7d)={fi['sentiment_volatility']['vol_7d']:.3f}, "
            f"감성 추세(trend_7d)={fi['trend_7d']:.3f}입니다."
        )

        return Opinion(
            agent_id=self.agent_id,
            target=target,
            reason=reason,
        )
