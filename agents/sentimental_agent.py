# agents/sentimental_agent.py

from __future__ import annotations

import os
import json
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import yfinance as yf

from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal
from config.agents import agents_info, dir_info
from core.utils_datetime import today_kst
from core.data_set import load_dataset, build_dataset, get_latest_close_price
from core.data_set import StockData, Target
from core.sentimental_classes.lstm_model import SentimentalLSTM
from core.sentimental_classes.news import merge_price_with_news_features
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from core.sentimental_classes.train_sentimental import LSTMModel 
from sklearn.preprocessing import StandardScaler
from core.sentimental_classes.train_sentimental import (
    LSTMModel,
    FEATURE_COLS,
    WINDOW_SIZE,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
)

# FinBERT / 뉴스 유틸
FinBertScorer: Any | None = None
load_or_fetch_news: Any | None = None
score_news_items: Any | None = None
attach_scores_to_items: Any | None = None
compute_finbert_features: Any | None = None
from core.sentimental_classes.train_sentimental import (
    LSTMModel,
    FEATURE_COLS,
    WINDOW_SIZE,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
)

try:
    from core.sentimental_classes.finbert_utils import (
        FinBertScorer,
        load_or_fetch_news,
        score_news_items,
        attach_scores_to_items,
        compute_finbert_features,
    )
except Exception as e:
    print("[warn] FinBERT 유틸 로드 실패:", repr(e))
    FinBertScorer = None
    load_or_fetch_news = None
    score_news_items = None
    attach_scores_to_items = None
    compute_finbert_features = None

USE_FINBERT = all(
    x is not None
    for x in [
        FinBertScorer,
        score_news_items,
        attach_scores_to_items,
        compute_finbert_features,
    ]
)

FEATURE_COLS_8 = [
    "return_1d",
    "hl_range",
    "Volume",
    "news_count_1d",
    "news_count_7d",
    "sentiment_mean_1d",
    "sentiment_mean_7d",
    "sentiment_vol_7d",
]

class DataScaler:
    """
    SentimentalAgent 전용 스케일러
    - X: (N, T, F) 시퀀스를 받아 feature 단위로 StandardScaler 적용
    - BaseAgent.pretrain에서 기대하는 인터페이스(예상):
      * fit_scalers(X, y)
      * transform_dataset(X, y) -> (X_scaled, y_scaled)
      * save_meta(ticker, agent_id)
      * load_meta(ticker, agent_id)
    """

    def __init__(self, feature_cols: List[str], window_size: int = 40):
        self.feature_cols = list(feature_cols)
        self.window_size = int(window_size)
        self.scaler = StandardScaler()

    def fit_scalers(self, X: np.ndarray, y: np.ndarray | None = None):
        """(N, T, F) 전체에 대해 feature 단위 평균/표준편차 학습"""
        if X.ndim != 3:
            raise ValueError(f"DataScaler.fit_scalers expects 3D array, got shape={X.shape}")
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        self.scaler.fit(X_flat)
        return self

    def transform_dataset(self, X: np.ndarray, y: np.ndarray):
        """학습된 스케일러로 X만 변환, y는 그대로 반환"""
        if X.ndim != 3:
            raise ValueError(f"DataScaler.transform_dataset expects 3D array, got shape={X.shape}")
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(n, t, f)
        return X_scaled.astype(np.float32), y.astype(np.float32)

    def save_meta(self, ticker: str, agent_id: str = "SentimentalAgent"):
        """스케일러와 메타 정보를 models/scalers 아래에 저장"""
        model_dir = Path("models/scalers")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{ticker}_{agent_id}.pkl"
        joblib.dump(self, path)
        print(f"[DataScaler.save_meta] saved to {path}")

    @classmethod
    def load_meta(cls, ticker: str, agent_id: str = "SentimentalAgent"):
        """저장된 스케일러 메타가 있으면 로드, 없으면 None"""
        path = Path("models/scalers") / f"{ticker}_{agent_id}.pkl"
        if not path.exists():
            print(f"[DataScaler.load_meta] no scaler file: {path}")
            return None
        scaler = joblib.load(path)
        print(f"[DataScaler.load_meta] loaded from {path}")
        return scaler

class SentimentalAgent(BaseAgent):
    def __init__(self, ticker: str, agent_id: str = "SentimentalAgent", **kwargs):
        # 0) BaseAgent 초기화 (한 번만!)
        super().__init__(ticker=ticker, agent_id=agent_id, **kwargs)

        # 1) 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) train_sentimental.py 의 기본 하이퍼파라미터 반영
        #    (이미 상단에서 HIDDEN_DIM, NUM_LAYERS, DROPOUT, WINDOW_SIZE, FEATURE_COLS import 되어 있다고 가정)
        self.window_size: int = int(WINDOW_SIZE)
        self.hidden_dim: int = int(HIDDEN_DIM)
        self.num_layers: int = int(NUM_LAYERS)
        self.dropout: float = float(DROPOUT)

        # 3) feature 목록 (훈련 때 사용한 전체 리스트)
        self.feature_cols: List[str] = list(FEATURE_COLS)
        input_dim = len(self.feature_cols)

        # 4) agents_info 설정으로 override (있으면)
        cfg = (agents_info or {}).get(agent_id, {})
        if not cfg:
            print("[WARN] agents_info['SentimentalAgent'] 없음 → 기본값 사용")
            cfg = {
                "window_size": self.window_size,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "epochs": 30,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "gamma": 0.3,
                "delta_limit": 0.05,
            }

        # config 기반으로 덮어쓰기
        self.window_size = cfg.get("window_size", self.window_size)
        self.hidden_dim = cfg.get("hidden_dim", self.hidden_dim)
        self.dropout = cfg.get("dropout", self.dropout)
        # num_layers 도 config에 있으면 반영
        self.num_layers = cfg.get("num_layers", self.num_layers)

        # 5) LSTM 모델 구성 (한 번만!)
        self.model: nn.Module = SentimentalLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.model_loaded: bool = False

        # 6) 사전 학습 weight 로드
        state_path = f"models/{ticker}_SentimentalAgent.pt"
        try:
            state = torch.load(state_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_loaded = True
            print(f"[SentimentalAgent] 사전 학습 모델 로드: {state_path}")
        except Exception as e:
            print(f"[SentimentalAgent] 모델 로드 실패: {e}")

        # 7) 기타 상태 변수
        self.last_price: Optional[float] = None
        self.currency: str = "USD"
        self._last_input: Optional[np.ndarray] = None  # (1, T, F)
        self.stockdata: Optional[StockData] = None
        self.window_size = cfg.get("window_size", self.window_size)
        self.hidden_dim = cfg.get("hidden_dim", self.hidden_dim)
        self.dropout = cfg.get("dropout", self.dropout)
        self.epochs = cfg.get("epochs", 30)
        self.learning_rate = cfg.get("learning_rate", 1e-3)
        self.batch_size = cfg.get("batch_size", 64)
        self.gamma = cfg.get("gamma", 0.3)
        self.delta_limit = cfg.get("delta_limit", 0.05)

        # --- LSTM 가중치 로드 ---
        try:
            self._load_model_only()
        except Exception as e:
            print(f"[SentimentalAgent] 초기 모델 로드 중 예외 (무시): {e}")

    def _load_model_only(self) -> None:
        """스케일러는 DataScaler가 관리하니까, 여기선 LSTM 가중치만 로드"""
        base_name = f"{self.ticker}_SentimentalAgent"
        model_path = Path("models") / f"{base_name}.pt"

        if model_path.exists():
            try:
                state = torch.load(model_path, map_location=self.device)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                print(f"[SentimentalAgent] 사전 학습 모델 로드: {model_path}")
                if missing or unexpected:
                    print("[SentimentalAgent] state_dict mismatch:",
                          "missing:", missing, "/ unexpected:", unexpected)
                else:
                    self.model_loaded = True
            except Exception as e:
                print(f"[SentimentalAgent] 모델 로드 실패: {e}")
                self.model_loaded = False
        else:
            print(f"[SentimentalAgent] 사전 학습 모델 파일 없음: {model_path}")
            self.model_loaded = False

    def _convert_uncertainty_to_confidence(self, sigma: float) -> float:
        """
        표준편차 sigma가 작을수록 confidence(0~1)가 커지도록 변환.
        """
        import numpy as np
        sigma = float(abs(sigma) or 1e-6)
        return float(1.0 / (1.0 + np.log1p(sigma)))

    def run_dataset(self, days: int = 365) -> StockData:
        """
        최근 days일치 가격 + 뉴스 피처를 기반으로
        1) FEATURE_COLS 입력 행렬 생성
        2) LSTM 입력 윈도우(1, T, F) 생성
        3) StockData 초기 스냅샷 생성
        """

        import pandas as pd
        import numpy as np
        from core.sentimental_classes.news import merge_price_with_news_features
        from core.sentimental_classes.train_sentimental import FEATURE_COLS, WINDOW_SIZE

        # ------------------------------------------------------------
        # 0) 날짜 범위 설정
        # ------------------------------------------------------------
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=days)

        # ------------------------------------------------------------
        # 1) 가격 데이터 (yfinance)
        # ------------------------------------------------------------
        df_price = yf.download(self.ticker, start=start, end=end)

        # yfinance MultiIndex column 대응 + 소문자 통일
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0].lower() for c in df_price.columns]
        else:
            df_price.columns = [c.lower() for c in df_price.columns]

        df_price = df_price.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )

        df_price["date"] = df_price.index
        df_price = df_price.reset_index(drop=True)

        # ------------------------------------------------------------
        # 2) 뉴스 + 가격 병합 (7일 집계 피처 포함)
        # ------------------------------------------------------------
        df_merged = merge_price_with_news_features(
            df_price=df_price,
            ticker=self.ticker,
            asof_kst=end.date(),
            base_dir=os.path.join("data", "raw", "news"),
        )

        # merge_price_with_news_features 가 (df, meta) 튜플을 반환하는 경우 방어
        if isinstance(df_merged, tuple):
            df_merged_df = df_merged[0]
        else:
            df_merged_df = df_merged

        # ------------------------------------------------------------
        # 3) FEATURE_COLS 누락 자동 보정
        # ------------------------------------------------------------
        df_feat = df_merged_df.sort_values("date").reset_index(drop=True)

        required_cols = list(FEATURE_COLS)
        missing = [c for c in required_cols if c not in df_feat.columns]
        print("[SentimentalAgent.run_dataset] missing(before):", missing)

        # --- 컬럼 대소문자 통일 (이미 소문자일 가능성이 높지만 방어) ---
        close_col = "close" if "close" in df_feat.columns else None
        high_col = "high" if "high" in df_feat.columns else None
        low_col = "low" if "low" in df_feat.columns else None

        # --- return_1d: 종가 기준 1일 수익률 ---
        if "return_1d" in missing:
            if close_col is not None:
                df_feat["return_1d"] = df_feat[close_col].pct_change().fillna(0.0)
            else:
                print("[SentimentalAgent.run_dataset] WARN: close column not found, return_1d filled with 0.0")
                df_feat["return_1d"] = 0.0

        # --- hl_range: (고가-저가)/종가 ---
        if "hl_range" in missing:
            if high_col is not None and low_col is not None and close_col is not None:
                rng = (df_feat[high_col] - df_feat[low_col]) / df_feat[close_col].replace(0, np.nan)
                df_feat["hl_range"] = rng.fillna(0.0)
            else:
                print("[SentimentalAgent.run_dataset] WARN: high/low/close missing, hl_range filled with 0.0")
                df_feat["hl_range"] = 0.0

        # --- Volume: 소문자 volume → 대문자 Volume 맞추기 ---
        if "Volume" not in df_feat.columns:
            if "volume" in df_feat.columns:
                df_feat["Volume"] = df_feat["volume"].fillna(0.0)
            else:
                df_feat["Volume"] = 0.0

        # --- 뉴스 1일 기준 피처(없으면 0으로 채우기) ---
        for col in ["news_count_1d", "sentiment_mean_1d"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        # ------------------------------------------------------------
        # 4) 최종 FEATURE_COLS 검증
        # ------------------------------------------------------------
        missing_after = [c for c in required_cols if c not in df_feat.columns]
        if missing_after:
            raise ValueError(
                f"[SentimentalAgent.run_dataset] FEATURE_COLS 중 아직 없는 컬럼: {missing_after}\n"
                f"현재 df_feat.columns = {df_feat.columns.tolist()}"
            )

        print("[SentimentalAgent.run_dataset] all FEATURE_COLS present.")

        # ------------------------------------------------------------
        # 5) 시계열 특성 행렬 생성
        # ------------------------------------------------------------
        feat_values = df_feat[required_cols].values.astype("float32")

        if len(feat_values) < WINDOW_SIZE:
            raise ValueError(
                f"윈도우 크기({WINDOW_SIZE})보다 데이터 길이({len(feat_values)})가 짧습니다."
            )

        X_last = feat_values[-WINDOW_SIZE:]      # (T, F)
        X_last = X_last[None, :, :]              # (1, T, F)
        self._last_input = X_last                # predict() 에서 사용

        # ------------------------------------------------------------
        # 6) StockData 생성 + 메타 정보 부착
        # ------------------------------------------------------------
        last_row = df_feat.iloc[-1]
        last_price = float(last_row.get("close", np.nan))
        self.last_price = last_price  # predict 에서 current_price 기본값으로 사용
        currency = "USD"

        # StockData 생성자에는 최소 인자만
        sd = StockData(
            ticker=self.ticker,
            last_price=last_price,
            currency=currency,
        )

        # 부가 정보는 속성으로 달기 (BaseAgent / DebateAgent에서 사용 가능)
        sd.feature_cols = FEATURE_COLS
        sd.window_size = WINDOW_SIZE
        sd.news_feats = {
            "news_count_7d": float(last_row.get("news_count_7d", 0)),
            "sentiment_mean_7d": float(last_row.get("sentiment_mean_7d", 0)),
            "sentiment_vol_7d": float(last_row.get("sentiment_vol_7d", 0)),
        }
        sd.raw_df = df_feat
        sd.agent_id = getattr(self, "agent_id", None)

        # 예전 코드와 호환되도록 snapshot 도 만들어 줌
        sd.snapshot = {
            "agent_id": sd.agent_id,
            "feature_cols": sd.feature_cols,
            "window_size": sd.window_size,
            "news_feats": sd.news_feats,
            "raw_df": sd.raw_df,
        }

        # LSTM 입력 시퀀스
        sd.X_seq = X_last  # (1, T, F)

        return sd


    from agents.base_agent import StockData  # 파일 상단 import 쪽에 추가되어 있지 않다면 추가

    def predict(
        self,
        X,
        n_samples: int = 100,
        current_price: float | None = None,
    ):
        """
        X로 StockData 또는 (T, F)/(1, T, F) 넘파이/텐서를 모두 허용한다.
        StockData가 들어오면 내부에서 X_seq와 last_price를 꺼내 쓴다.
        """
        # 1) StockData가 들어온 경우 처리
        if isinstance(X, StockData):
            sd = X

            # run_dataset에서 X_seq를 세팅했는지 확인
            if getattr(sd, "X_seq", None) is None:
                raise ValueError(
                    "StockData에 X_seq가 없습니다. SentimentalAgent.run_dataset()에서 "
                    "sd.X_seq를 설정했는지 확인해 주세요."
                )

            X_in = sd.X_seq

            # current_price가 안 들어왔으면 StockData의 last_price 사용
            if current_price is None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)

            # 디버깅/설명용으로 보관
            self._last_stockdata = sd
            self._last_input = X_in

        else:
            # 이미 넘파이/텐서인 경우 그대로 사용
            X_in = X

        # 2) BaseAgent.predict 호출 (여기는 넘파이/텐서만 받도록 유지)
        target = super().predict(X_in, n_samples=n_samples, current_price=current_price)

        # 3) 타겟에 메타정보 보강
        target.ticker = self.ticker
        target.agent_id = getattr(self, "agent_id", "SentimentalAgent")
        self.target = target

        return target


    # BaseAgent.decode_prediction
    def decode_prediction(self, y_pred_raw, stock_data=None, current_price=None) -> float:
        """
        BaseAgent.predict 내부에서 사용할 디코더.
        모델 출력(수익률 비슷한 값)을 [-20%, +20%]로 제한 후 가격으로 변환.
        """
        y_raw = float(np.asarray(y_pred_raw).reshape(-1)[-1])

        y_scaler = None
        if self.scaler is not None:
            if isinstance(self.scaler, dict):
                y_scaler = self.scaler.get("y_scaler", None)
            else:
                y_scaler = getattr(self.scaler, "y_scaler", None)

        if y_scaler is not None:
            try:
                y_decoded = float(y_scaler.inverse_transform([[y_raw]])[0, 0])
            except Exception:
                y_decoded = y_raw
        else:
            y_decoded = y_raw

        max_abs_return = 0.20
        predicted_return = max(min(y_decoded, max_abs_return), -max_abs_return)

        base_price = None
        if stock_data is not None:
            base_price = getattr(stock_data, "last_price", None)
        if base_price is None:
            base_price = current_price
        if base_price is None:
            base_price = 1.0

        next_close = float(base_price * (1.0 + predicted_return))
        return next_close

    def build_finbert_news_features(
        self,
        ticker: str,
        asof_kst: datetime,
        base_dir: str = "data/raw/news",
        days_list = [7, 30],
    ):
        from core.sentimental_classes.eodhd_client import fetch_news_from_eodhd
        from core.sentimental_classes.finbert_utils import FinBertScorer
        import numpy as np
        import os

        scorer = FinBertScorer()
        feats = {}

        for d in days_list:
            news = fetch_news_from_eodhd(ticker, days=d)

            # 🔥 뉴스 없으면 바로 실패
            if (news is None) or (len(news) == 0):
                raise RuntimeError(
                    f"[NewsError] {ticker} 최근 {d}일 뉴스 없음 (fetch_news_from_eodhd)"
                )

            scores = []
            for item in news:
                title = item.get("title", "") or ""
                score = scorer.score(title) or 0
                scores.append(score)

            scores = np.array(scores)

            feats[f"sentiment_mean_{d}d"] = float(np.mean(scores))
            feats[f"sentiment_vol_{d}d"] = float(np.std(scores))
            feats[f"news_count_{d}d"] = len(scores)

            # Trend: 마지막 25% 평균 - 처음 25% 평균
            if len(scores) >= 4:
                q = len(scores) // 4
                early = np.mean(scores[:q])
                late = np.mean(scores[-q:])
                feats[f"sentiment_trend_{d}d"] = float(late - early)
            else:
                feats[f"sentiment_trend_{d}d"] = 0.0

            # Shock: z-score of last score
            if len(scores) >= 2:
                feats[f"sentiment_shock_z_{d}d"] = float(
                    (scores[-1] - np.mean(scores)) / (np.std(scores) + 1e-6)
                )
            else:
                feats[f"sentiment_shock_z_{d}d"] = 0.0

        return feats


        news_feats = self.build_finbert_news_features(
            ticker=self.ticker,
            asof_kst=asof_date_kst,
            base_dir=os.path.join("data", "raw", "news"),
        )

        if (not news_feats) or (news_feats.get("news_count_7d", 0) == 0 and
                                news_feats.get("news_count_30d", 0) == 0):
            raise RuntimeError(
                f"[SentimentalAgent] 뉴스 수집 실패: {self.ticker} 최근 7/30일 기사 0건"
            )

        # 5) StockData 인스턴스 생성
        sd = StockData()
        sd.ticker = self.ticker
        sd.last_price = last_price
        sd.currency = currency

        # SentimentalAgent 전용 스냅샷 dict에 넣기
        sd.SentimentalAgent = {
            "X_seq": X_latest,
            "feature_cols": feature_cols,
            "asof_date": asof_date_kst,
            "news_features": news_feats,
        }

        sd.X_seq = X_latest
        sd.feature_cols = feature_cols

        # 6) self.stockdata에 보관 후 반환
        self.stockdata = sd
        return sd

    def _load_scaler_and_model(self) -> None:
        # train_sentimental.py에서 저장한 스케일러/모델 로드
        base_name = f"{self.ticker}_SentimentalAgent"
        scaler_path = Path("models") / "scalers" / f"{base_name}.pkl"
        model_path = Path("models") / f"{base_name}.pt"

        # 스케일러/메타
        if scaler_path.exists():
            meta = joblib.load(scaler_path)
            meta_feature_cols = None
            meta_window_size = None

            if isinstance(meta, dict) and ("x_scaler" in meta or "y_scaler" in meta):
                self.scaler = meta
                meta_feature_cols = meta.get("feature_cols", None)
                meta_window_size = meta.get("window_size", None)
            elif isinstance(meta, dict) and "scaler" in meta:
                self.scaler = meta.get("scaler", None)
                meta_feature_cols = meta.get("feature_cols", None)
                meta_window_size = meta.get("window_size", None)
            else:
                self.scaler = meta

            if meta_feature_cols is not None:
                self.feature_cols = list(meta_feature_cols)
            if meta_window_size is not None:
                self.window_size = int(meta_window_size)

            print(f"[SentimentalAgent] 스케일러/메타 로드: {scaler_path}")
        else:
            print(f"[SentimentalAgent] 스케일러 파일 없음: {scaler_path}")
            self.scaler = None

        # LSTM 가중치
        if model_path.exists():
            try:
                state = torch.load(model_path, map_location=self.device)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                print(f"[SentimentalAgent] 사전 학습 모델 로드: {model_path}")
                if missing or unexpected:
                    print("[SentimentalAgent] state_dict mismatch:",
                          "missing:", missing, "/ unexpected:", unexpected)
                else:
                    self.model_loaded = True
            except Exception as e:
                print(f"[SentimentalAgent] 모델 로드 실패: {e}")
                self.model_loaded = False
        else:
            print(f"[SentimentalAgent] 사전 학습 모델 파일 없음: {model_path}")
            self.model_loaded = False

    # BaseAgent.pretrain()용
    def _build_model(self) -> nn.Module:
        """BaseAgent.pretrain에서 사용할 모델 생성."""
        try:
            X, y, cols = _load_dataset_compat(
                self.ticker,
                self.agent_id,
                window_size=self.window_size,
            )
        except Exception:
            _build_dataset_compat(
                self.ticker,
                self.agent_id,
                window_size=self.window_size,
            )
            X, y, cols = _load_dataset_compat(
                self.ticker,
                self.agent_id,
                window_size=self.window_size,
            )

        input_dim = X.shape[-1]
        self.feature_cols = list(cols)

        net = SentimentalLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return net

    def model_path(self) -> str:
        """모델 저장 경로 통일."""
        try:
            model_dir = dir_info["model_dir"]
        except Exception:
            model_dir = "models"

        ticker = getattr(self, "ticker", "UNKNOWN")
        agent_id = getattr(self, "agent_id", "SentimentalAgent")
        return os.path.join(model_dir, f"{ticker}_{agent_id}.pt")

    def _load_model_if_exists(self) -> None:
        """model_path 기준으로 파일이 있으면 BaseAgent.load_model 사용."""
        model_path = self.model_path()
        if not os.path.exists(model_path):
            self.model_loaded = False
            return

        ok = False
        try:
            ok = self.load_model(model_path)
        except Exception as e:
            print(f"[SentimentalAgent] 모델 로드 실패: {e}")
            ok = False

        self.model_loaded = bool(ok)

    # MC Dropout helper (dataset 기반)
    @torch.inference_mode()
    def _mc_dropout_predict(self, x: torch.Tensor, T: int = 30) -> Tuple[float, float]:
        """입력 x에 대해 MC Dropout으로 (mean, std) 수익률 예측."""
        if self.model is None:
            raise RuntimeError("model is None for MC Dropout")

        self.model.train()
        outs = []
        for _ in range(T):
            outs.append(self.model(x).detach())
        self.model.eval()

        y = torch.stack(outs, dim=0).squeeze(-1)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        return float(mean.squeeze().item()), float(std.squeeze().item())

    @torch.inference_mode()
    def _predict_next_close(self) -> Tuple[float, float, float, List[str]]:
        """
        ✅ 새 파이프라인 기반 다음날 종가 예측
        - run_dataset() 으로 최근 40일 X_seq + last_price 생성
        - LSTM 출력 = '다음날 수익률'
        - last_price * (1 + return) = next_close
        """
        if not self.ticker:
            raise ValueError("ticker is None in _predict_next_close")

        # 1) 최신 데이터셋 / 입력 윈도우 만들기
        X = getattr(self, "_last_input", None)
        if X is None:
            # run_dataset() 이 self._last_input, self.last_price 채워줌
            X = self.run_dataset()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # 2) LSTM으로 수익률 예측
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_tensor)

        out = out.reshape(-1)
        mean_ret = float(out[0])          # 예: -0.003 → -0.3%
        std_ret = 0.01                    # 필요하면 MC Dropout로 교체

        # 3) 현재 종가 (run_dataset에서 세팅한 실제 Close)
        last_close = float(getattr(self, "last_price", 0.0) or 0.0)

        if last_close > 0:
            pred_close = float(last_close * (1.0 + mean_ret))
        else:
            # 혹시라도 last_price 못 구하면 그냥 return만 리턴
            pred_close = float(mean_ret)

        uncertainty_std = float(std_ret)
        confidence = float(1.0 / (1.0 + max(1e-6, uncertainty_std)))

        # cols: 특징 이름 리스트 (필요하면 self.feature_cols 사용)
        cols = list(getattr(self, "feature_cols", []))
        return pred_close, uncertainty_std, confidence, cols


    # ctx 구성 (가격 + 뉴스 감성 스냅샷)
    def build_ctx(self, asof_date_kst: Optional[str] = None) -> Dict[str, Any]:
        # 0) StockData 확보
        stockdata: StockData | None = getattr(self, "stockdata", None)
        if stockdata is None:
            raise ValueError(
                "[SentimentalAgent] build_ctx 호출 전에 run_dataset()을 먼저 실행하세요."
            )

        senti_snap = getattr(stockdata, "SentimentalAgent", None) or {}

        # 1) 기준 날짜(asof_date_kst)
        if asof_date_kst is None:
            asof_date_kst = senti_snap.get("asof_date") or datetime.now().strftime("%Y-%m-%d")

        # 2) 예측 값 (기존 로직 유지)
        pred_close, uncertainty_std, confidence, cols = self._predict_next_close()

        # 3) 가격 스냅샷
        price_snapshot: Dict[str, Optional[float]] = {}

        # 우선 run_dataset()에서 저장해둔 X_seq / feature_cols 사용
        x_seq = senti_snap.get("X_seq", None)
        feat_cols = senti_snap.get("feature_cols", None)

        try:
            if isinstance(x_seq, np.ndarray) and feat_cols:
                # X_seq: (1, T, F) 가정 → 마지막 시점
                last = x_seq[-1, -1, :]
                snap_map = {c: float(v) for c, v in zip(feat_cols, last)}
                for k in ("Close", "Open", "High", "Low", "Volume", "returns"):
                    if k in snap_map:
                        price_snapshot[k] = snap_map[k]
            else:
                # fallback: 예전처럼 dataset에서 다시 로드
                X, _, cols2 = _load_dataset_compat(
                    self.ticker, self.agent_id, window_size=self.window_size
                )
                last = X[-1, -1, :]
                snap_map = {c: float(v) for c, v in zip(cols2, last)}
                for k in ("Close", "Open", "High", "Low", "Volume", "returns"):
                    if k in snap_map:
                        price_snapshot[k] = snap_map[k]
        except Exception:
            pass

        # 4) ✅ 뉴스/감성 피처: run_dataset()에서 저장한 것만 사용
        news_feats = senti_snap.get("news_features", None)
        if not news_feats:
            # 없으면 0 피처
            news_feats = _zero_news_feats()

        # 안전하게 꺼내기
        sentiment_summary = news_feats.get("sentiment_summary", {})
        sentiment_vol = news_feats.get("sentiment_volatility", {})
        news_count = news_feats.get("news_count", {})
        trend_7d = news_feats.get("trend_7d", 0.0)
        has_news = bool(news_feats.get("has_news", False))

        # 5) snapshot / prediction 구성 (기존과 거의 동일)
        snapshot = {
            "asof_date": asof_date_kst,
            "last_price": price_snapshot.get("Close", np.nan),
            "currency": stockdata.currency or "USD",
            "window_size": self.window_size,
            "feature_cols_preview": [c for c in (cols or [])[:8]],
        }

        last_price = snapshot["last_price"]
        if last_price and last_price == last_price:
            pred_return = float(pred_close / last_price - 1.0)
        else:
            pred_return = None

        feature_importance = {
            "sentiment_score": sentiment_summary.get("mean_7d", 0.0),
            "sentiment_summary": sentiment_summary,
            "sentiment_volatility": {
                "vol_7d": sentiment_vol.get("vol_7d", 0.0),
            },
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

    # Opinion / Rebuttal / Revision 프롬프트
    def _build_messages_opinion(
        self,
        stock_data: StockData,
        target: Target,
    ) -> Tuple[str, str]:
        if stock_data is None:
            stock_data = self.stockdata

        ctx: Dict[str, Any] = {}

        # 메타
        ctx["ticker"] = getattr(stock_data, "ticker", self.ticker)
        ctx["currency"] = getattr(stock_data, "currency", "USD")

        # 가격
        last_close = getattr(stock_data, "last_price", None)
        ctx["last_close"] = last_close
        ctx["next_close"] = float(getattr(target, "next_close", 0.0))

        change_ratio = None
        if isinstance(last_close, (int, float)) and last_close not in (0, None):
            try:
                change_ratio = ctx["next_close"] / float(last_close) - 1.0
            except ZeroDivisionError:
                change_ratio = None
        ctx["change_ratio"] = change_ratio

        ctx["uncertainty_std"] = getattr(target, "uncertainty", None)
        ctx["confidence"] = getattr(target, "confidence", None)

        # SentimentalAgent 전용 스냅샷 (stock_data.SentimentalAgent dict 가정)
        snap = getattr(stock_data, "SentimentalAgent", None)
        if isinstance(snap, dict):
            for k, v in snap.items():
                if isinstance(v, np.ndarray):
                    if v.ndim == 0:
                        ctx[k] = v.item()
                    elif v.size > 0:
                        flat = v.reshape(-1)
                        last_val = flat[-1]
                        try:
                            ctx[k] = float(last_val)
                        except Exception:
                            ctx[k] = last_val
                    else:
                        ctx[k] = None
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    ctx[k] = v[-1]
                else:
                    ctx[k] = v

        ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)

        prompts = OPINION_PROMPTS["SentimentalAgent"]
        system_text = prompts["system"]
        user_tmpl = prompts["user"]

        try:
            user_text = user_tmpl.format(context=ctx_json)
        except KeyError:
            user_text = user_tmpl.replace("{context}", ctx_json)

        return system_text, user_text

    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        stock_data = args[0] if len(args) > 0 else kwargs.get("stock_data")
        target: Optional[Target] = args[1] if len(args) > 1 else kwargs.get("target")

        opponent = None
        for key in ("opponent", "opponent_opinion", "other_opinion", "other", "opinion"):
            if key in kwargs:
                opponent = kwargs[key]
                break
        if opponent is None and len(args) > 2:
            opponent = args[2]

        if isinstance(opponent, Opinion):
            opp_agent = getattr(opponent, "agent_id", "UnknownAgent")
            opp_reason = getattr(opponent, "reason", "")
        elif isinstance(opponent, dict):
            opp_agent = opponent.get("agent_id", "UnknownAgent")
            opp_reason = opponent.get("reason", "")
        else:
            opp_agent = "UnknownAgent"
            opp_reason = str(opponent) if opponent is not None else ""

        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        pred_close = float(target.next_close) if target else float(
            ctx["prediction"]["pred_next_close"]
        )
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        pp = REBUTTAL_PROMPTS.get("SentimentalAgent", {})
        system_tmpl = pp.get(
            "system",
            "당신은 감성 기반 단기 주가 분석가로서 상대 의견의 허점을 감성 지표와 뉴스 데이터를 바탕으로 반박합니다.",
        )
        user_tmpl = pp.get(
            "user",
            (
                "티커: {ticker}\n"
                "상대 에이전트: {opp_agent}\n"
                "상대 의견:\n{opp_reason}\n\n"
                "우리 예측:\n- next_close: {pred_close}\n- 예상 변화율(현재가 대비): {chg}\n"
                "감성 근거:\n- mean7={mean7}, mean30={mean30}, pos7={pos7}, neg7={neg7}\n"
                "- vol7={vol7}, trend7={trend7}, news7={news7}\n\n"
                "요청: 위 정보를 바탕으로 상대 의견의 약점 2~4개를 조목조목 반박하세요."
            ),
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

    def _build_messages_revision(self, *args, **kwargs) -> Tuple[str, str]:
        stock_data = args[0] if len(args) > 0 else kwargs.get("stock_data")
        target: Optional[Target] = args[1] if len(args) > 1 else kwargs.get("target")

        # 초안
        prev = None
        rebs = None
        for key in ("previous", "previous_opinion", "draft", "opinion"):
            if key in kwargs:
                prev = kwargs[key]
                break
        if prev is None and len(args) > 2:
            prev = args[2]

        # 반박들
        for key in ("rebuttals", "replies", "responses"):
            if key in kwargs:
                rebs = kwargs[key]
                break
        if rebs is None and len(args) > 3:
            rebs = args[3]

        def _op_text(x: Union[Opinion, Dict[str, Any], str, None]) -> str:
            if isinstance(x, Opinion):
                return getattr(x, "reason", "")
            if isinstance(x, dict):
                return x.get("reason", "")
            return x or ""

        prev_reason = _op_text(prev)

        reb_texts: List[str] = []
        if isinstance(rebs, list):
            for r in rebs:
                reb_texts.append(_op_text(r))
        elif rebs is not None:
            reb_texts.append(_op_text(rebs))

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

        pred_close = float(target.next_close) if target else float(
            pred_info.get("pred_next_close")
        )
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        # context 요약
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

        mean7 = sent.get("mean_7d", None)
        mean30 = sent.get("mean_30d", None)
        pos7 = sent.get("pos_ratio_7d", None)
        neg7 = sent.get("neg_ratio_7d", None)

        if mean7 is not None and mean30 is not None:
            context_parts.append(
                f"최근 7일 평균 감성 점수는 {mean7:.3f}, 최근 30일 평균은 {mean30:.3f}입니다."
            )
        if pos7 is not None and neg7 is not None:
            context_parts.append(
                f"최근 7일 기준 긍정 기사 비율은 {pos7:.2%}, 부정 기사 비율은 {neg7:.2%}입니다."
            )
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

        pp = REVISION_PROMPTS.get("SentimentalAgent", {})
        system_tmpl = pp.get(
            "system",
            (
                "당신은 감성 기반 단기 주가 분석가입니다. "
                "초안 의견과 반박들을 검토해 핵심만 남기고, 데이터에 근거해 결론을 다듬습니다."
            ),
        )
        user_tmpl = pp.get(
            "user",
            (
                "티커: {ticker}\n"
                "초안 의견:\n{prev}\n\n"
                "수신한 반박 요약:\n{rebuts}\n\n"
                "업데이트된 수치:\n- next_close: {pred_close}\n- 예상 변화율: {chg}\n"
                "감성 근거 스냅샷:\n- mean7={mean7}, mean30={mean30}, pos7={pos7}, neg7={neg7}\n"
                "- vol7={vol7}, trend7={trend7}, news7={news7}\n\n"
                "추가 컨텍스트:\n{context}\n\n"
                "요청: 초안의 과장/중복/약한 근거를 정리하고, 강한 근거(감성 추세, 변동성, 뉴스 수 변화)를 중심으로 "
                "최종 의견을 3~5문장으로 재작성하세요. 불확실성/신뢰도 해석을 포함하세요."
            ),
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
            mean7=("NA" if mean7 is None else f"{mean7:.4f}"),
            mean30=("NA" if mean30 is None else f"{mean30:.4f}"),
            pos7=("NA" if pos7 is None else f"{pos7:.4f}"),
            neg7=("NA" if neg7 is None else f"{neg7:.4f}"),
            vol7=("NA" if vol7 is None else f"{vol7:.4f}"),
            trend7=("NA" if trend7 is None else f"{trend7:.4f}"),
            news7=("NA" if news7 is None else f"{news7}"),
            context=context_str,
        )
        return system_tmpl, user_text


    # 레거시 Opinion API
    def get_opinion(self, idx: int = 0, ticker: Optional[str] = None) -> Opinion:
        if ticker and ticker != self.ticker:
            self.ticker = str(ticker).upper()

        pred_close, uncertainty_std, confidence, _ = self._predict_next_close()
        target = Target(
            next_close=float(pred_close),
            uncertainty=float(uncertainty_std),
            confidence=float(confidence),
        )

        # BaseAgent.reviewer_draft 사용 시도
        try:
            if hasattr(self, "reviewer_draft"):
                op = self.reviewer_draft(getattr(self, "stockdata", None), target)
                return op
        except Exception as e:
            print("[SentimentalAgent] reviewer_draft 사용 실패:", e)

        # fallback: 단순 텍스트 요약
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
