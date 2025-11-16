# agents/sentimental_agent.py
# ===============================================================
# SentimentalAgent: 감성(뉴스/텍스트) + LSTM 기반 예측 에이전트
#  - LSTM 출력은 "다음날 수익률(return)"을 예측한다고 가정하고,
#    마지막 종가(last_close)에 곱해서 다음 종가(pred_next_close)를 계산한다.
# ===============================================================

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union

from pathlib import Path
from datetime import datetime, timedelta, date

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any
from core.data_set import load_dataset

# ---------------------------
# 프로젝트 의존 모듈 (안전 import)
# ---------------------------
# 이미 있는 타입 힌트용 코드도 활용
try:
    from agents.base_agent import StockData, Target, Opinion, Rebuttal
except Exception:
    StockData = Any
    Target = Any
    Opinion = Any
    Rebuttal = Any

# dir_info 쓰는 경우를 대비해서 (있으면 사용, 없으면 기본값)
try:
    from config.agents import dir_info
except Exception:
    dir_info = {}

try:
    from agents.base_agent import BaseAgent, StockData, Target, Opinion  # type: ignore
except Exception:
    BaseAgent = object  # type: ignore

    @dataclass
    class Target:  # type: ignore
        next_close: float
        uncertainty: float
        confidence: float

    @dataclass
    class Opinion:  # type: ignore
        agent_id: str
        target: Target
        reason: str

try:
    from config.agents import agents_info, dir_info  # type: ignore
except Exception:
    agents_info = {
        "SentimentalAgent": {
            "window_size": 40,
            "hidden_dim": 128,
            "dropout": 0.2,
            "epochs": 30,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "x_scaler": "StandardScaler",
            "y_scaler": "StandardScaler",
            "gamma": 0.3,
            "delta_limit": 0.05,
        }
    }
    dir_info = {
        "data_dir": "data",
        "model_dir": "models",
        "scaler_dir": os.path.join("models", "scalers"),
    }

try:
    from core.data_set import build_dataset, load_dataset  # type: ignore
except Exception:
    build_dataset = None  # type: ignore

    def load_dataset(*args, **kwargs):  # type: ignore
        raise RuntimeError("core.data_set.load_dataset 를 찾을 수 없습니다.")

# 프롬프트 세트 (있는 경우 사용)
try:
    from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS  # type: ignore
except Exception:
    OPINION_PROMPTS = REBUTTAL_PROMPTS = REVISION_PROMPTS = None  # type: ignore


# ---------------------------
# FinBERT / 뉴스 유틸 import
# ---------------------------
from typing import Any  # 파일 맨 위에 이미 있으면 생략

FinBertScorer: Any | None = None
load_or_fetch_news: Any | None = None
score_news_items: Any | None = None
attach_scores_to_items: Any | None = None
compute_finbert_features: Any | None = None

try:
    # ✅ FinBERT 관련 유틸은 전부 이 모듈에서 가져온다
    from core.sentimental_classes.finbert_utils import (
        FinBertScorer,
        load_or_fetch_news,
        score_news_items,
        attach_scores_to_items,
        compute_finbert_features,
    )
except Exception as e:
    print("[warn] core.sentimental_classes.finbert_utils 에서 FinBERT 관련 유틸을 불러오지 못했습니다:", repr(e))
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

try:
    from core.data_set import load_dataset  # 프로젝트 공통 데이터 로더
except Exception:
    load_dataset = None  # 타입 힌트용/안전장치

# ---------------------------------------------------------------
# 모델 정의: LSTM + Dropout (MC Dropout 지원)
#   ⚠️ 출력은 "다음날 수익률(return)" 값으로 가정
#   Note: SentimentalNet은 이제 SentimentalAgent 내부로 통합됨
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# load/build dataset 시그니처 호환 유틸
# ---------------------------------------------------------------
def _load_dataset_compat(ticker: str, agent_id: str, window_size: Optional[int] = None):
    if not ticker:
        raise ValueError("load_dataset_compat: empty ticker")
    try:
        return load_dataset(ticker, agent_id, window_size=window_size)  # type: ignore
    except TypeError:
        pass
    try:
        return load_dataset(ticker, agent_id, seq_len=window_size)  # type: ignore
    except TypeError:
        pass
    return load_dataset(ticker, agent_id)  # type: ignore


def _build_dataset_compat(ticker: str, agent_id: str, window_size: Optional[int] = None):
    if not ticker:
        raise ValueError("build_dataset_compat: empty ticker")
    if build_dataset is None:
        return
    try:
        return build_dataset(ticker, agent_id, window_size=window_size)  # type: ignore
    except TypeError:
        pass
    try:
        return build_dataset(ticker, agent_id, seq_len=window_size)  # type: ignore
    except TypeError:
        pass
    return build_dataset(ticker, agent_id)  # type: ignore


# ---------------------------------------------------------------
# 유틸: 진단 스크립트가 저장한 뉴스 캐시를 읽어 FinBERT 집계 피처 생성
# ---------------------------------------------------------------
def _utc_from_kst_asof(asof_kst: str, lookback_days: int = 40) -> Tuple[str, str, date]:
    kst_dt = datetime.fromisoformat(asof_kst)
    utc_today = (kst_dt - timedelta(hours=9)).date()
    to_utc_date = utc_today - timedelta(days=1)
    from_utc_date = to_utc_date - timedelta(days=lookback_days)
    return from_utc_date.isoformat(), to_utc_date.isoformat(), to_utc_date


def _zero_news_feats() -> Dict[str, Any]:
    """뉴스/FinBERT 피처가 없을 때 기본값."""
    return {
        "sentiment_summary": {
            "mean_7d": 0.0,
            "mean_30d": 0.0,
            "pos_ratio_7d": 0.0,
            "neg_ratio_7d": 0.0,
        },
        "sentiment_volatility": {"vol_7d": 0.0},
        "news_count": {"count_1d": 0, "count_7d": 0},
        "trend_7d": 0.0,
        "has_news": False,
    }


def build_finbert_news_features(
    ticker: str,
    asof_kst: str,
    base_dir: str = "data/raw/news",
    text_fields: Tuple[str, ...] = ("title", "content", "text", "summary"),
) -> Dict[str, Any]:
    """
    SentimentalAgent 가 사용할 FinBERT 입력용 뉴스 피처 생성.
    
    프로세스:
    1. 캐시 파일 확인 (정확한 기간 → 최신 파일 fallback)
    2. 캐시가 없으면 load_or_fetch_news를 통해 EODHD에서 뉴스 수집
    3. FinBERT로 감성 분석 수행
    4. 일별 피처 집계
    
    Args:
        ticker: 종목 코드 (예: "NVDA")
        asof_kst: 기준 날짜 (KST, "YYYY-MM-DD" 형식)
        base_dir: 뉴스 캐시 디렉토리
        text_fields: FinBERT 분석에 사용할 텍스트 필드
        
    Returns:
        Dict[str, Any]: 감성 피처 딕셔너리
    """
    
    # FinBERT 자체가 비활성화된 경우: 안전하게 0 피처 반환
    if FinBertScorer is None:
        print("[FinBERT] FinBertScorer 없음 → 감성 피처를 0으로 대체합니다.")
        return _zero_news_feats()

    fr, to, to_date_utc = _utc_from_kst_asof(asof_kst, lookback_days=40)
    symbol_us = f"{ticker}.US"

    project_root = Path(__file__).resolve().parent.parent
    base = Path(base_dir)
    if not base.is_absolute():
        base = project_root / base
    base.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성

    # 1) 정확한 기간 파일 우선 탐색
    path = base / f"{symbol_us}_{fr}_{to}.json"
    print(f"[FinBERT] 캐시 탐색: {path} (exists={path.exists()})")

    items = None
    
    if path.exists():
        # 캐시 파일이 있으면 로드
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
            print(f"[FinBERT] 캐시 파일 로드 성공: {len(items) if isinstance(items, list) else 0}건")
        except Exception as e:
            print(f"[FinBERT] 캐시 로드 실패: {path} ({e})")
            items = None
    
    if items is None:
        # 2) fallback: 동일 티커의 최신 캐시 파일 사용
        pattern = f"{symbol_us}_*.json"
        candidates = sorted(base.glob(pattern))
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            print(
                f"[FinBERT] 정확한 기간 캐시 없음, 최신 파일 사용: {latest.name}"
            )
            try:
                items = json.loads(latest.read_text(encoding="utf-8"))
                print(f"[FinBERT] 최신 캐시 로드 성공: {len(items) if isinstance(items, list) else 0}건")
            except Exception as e:
                print(f"[FinBERT] 최신 캐시 로드 실패: {latest} ({e})")
                items = None
    
    # 3) 캐시가 없으면 load_or_fetch_news를 통해 실제로 뉴스 수집
    if items is None and load_or_fetch_news is not None:
        print(f"[FinBERT] 캐시 없음, EODHD에서 뉴스 수집 시도...")
        try:
            from datetime import date as date_type
            start_date = date_type.fromisoformat(fr)
            end_date = date_type.fromisoformat(to)
            
            import os
            api_key = os.getenv("EODHD_API_KEY", "")
            
            items = load_or_fetch_news(
                ticker=ticker,
                start=start_date,
                end=end_date,
                api_key=api_key
            )
            
            if items and len(items) > 0:
                print(f"[FinBERT] 뉴스 수집 성공: {len(items)}건")
            else:
                print(f"[FinBERT] 뉴스 수집 결과: 0건 (또는 수집 실패)")
        except Exception as e:
            print(f"[FinBERT] 뉴스 수집 중 오류 발생: {e}")
            items = None
    
    # 4) 여전히 뉴스가 없으면 0 피처 반환
    if not items or not isinstance(items, list) or len(items) == 0:
        print(f"[FinBERT] 뉴스 데이터 없음 → 0 피처 반환")
        return _zero_news_feats()

    # 날짜 필드 정규화
    for it in items:
        for k in ("date", "published_date", "time", "pubDate"):
            if not isinstance(it.get(k), str):
                it[k] = ""

    print(f"[FinBERT] {len(items)}건 뉴스 감성 분석 시작...")
    try:
        scorer = FinBertScorer()
        scores = score_news_items(items, scorer=scorer, text_fields=text_fields)
        items_scored = attach_scores_to_items(items, scores)
        feats = compute_finbert_features(items_scored, asof_utc_date=to_date_utc)
    except Exception as e:
        print(f"[FinBERT] 감성 분석 중 오류 발생 → 0 피처로 대체: {e}")
        import traceback
        traceback.print_exc()
        return _zero_news_feats()

    vol7 = feats.get("sentiment_volatility", {}).get("vol_7d", 0.0)
    feats["sentiment_volatility"] = {"vol_7d": vol7}

    print(
        f"[FinBERT] 7d_mean={feats['sentiment_summary']['mean_7d']:.3f} "
        f"7d_cnt={feats['news_count']['count_7d']}"
    )
    feats["has_news"] = True
    return feats


# ---------------------------------------------------------------
# 본체: SentimentalAgent
# ---------------------------------------------------------------
class SentimentalAgent(BaseAgent, nn.Module):  # type: ignore
    """
    SentimentalAgent: 감성 분석 기반 주가 예측 에이전트
    
    뉴스 및 텍스트 데이터의 감성 분석과 LSTM 모델을 결합하여
    주가 예측을 수행하는 에이전트입니다.
    
    주요 기능:
    - FinBERT를 활용한 뉴스 감성 분석
    - LSTM 기반 시계열 예측
    - Monte Carlo Dropout을 통한 불확실성 추정
    - LLM을 활용한 Opinion, Rebuttal, Revision 생성
    
    Attributes:
        agent_id: 에이전트 식별자 (기본값: "SentimentalAgent")
        window_size: 시계열 윈도우 크기
        hidden_dim: LSTM hidden dimension
        dropout: Dropout 비율
        input_dim: 입력 feature 차원
        lstm: LSTM 레이어
        fc: Fully connected 레이어
    """
    agent_id: str = "SentimentalAgent"

    def __init__(
        self,
        ticker: str | None = None,
        agent_id: str = "SentimentalAgent",
        data_dir: str = None,
        window_size: int = None,
        hidden_dim: int = None,
        dropout: float = None,
        epochs: int = None,
        learning_rate: float = None,
        batch_size: int = None,
        *args,
        **kwargs,
    ):
        # 1) nn.Module 먼저 초기화
        nn.Module.__init__(self)
        
        # 2) BaseAgent 초기화
        agent_id = agent_id or "SentimentalAgent"
        if data_dir is None:
            data_dir = dir_info.get("data_dir", "data")
        super().__init__(agent_id=agent_id, ticker=ticker, data_dir=data_dir, *args, **kwargs)

        # 티커 정리
        if not getattr(self, "ticker", None):
            self.ticker = ticker
        if self.ticker is None or str(self.ticker).strip() == "":
            raise ValueError("SentimentalAgent: ticker is None/empty")
        self.ticker = str(self.ticker).upper()
        setattr(self, "symbol", self.ticker)

        # 설정 로드
        cfg = (agents_info or {}).get(self.agent_id, {})
        if not cfg:
            print("[WARN] agents_info['SentimentalAgent'] 가 없어 기본값 사용")
            cfg = {
                "window_size": 40,
                "hidden_dim": 128,
                "dropout": 0.2,
                "epochs": 30,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "x_scaler": "StandardScaler",
                "y_scaler": "StandardScaler",
                "gamma": 0.3,
                "delta_limit": 0.05,
            }
        self.window_size = int(window_size or cfg.get("window_size", 40))
        self.hidden_dim = int(hidden_dim or cfg.get("hidden_dim", 128))
        self.dropout = float(dropout or cfg.get("dropout", 0.2))
        self.epochs = int(epochs or cfg.get("epochs", 30))
        self.lr = float(learning_rate or cfg.get("learning_rate", 1e-3))
        self.batch_size = int(batch_size or cfg.get("batch_size", 64))

        # 모델 하이퍼파라미터 설정 (Config 기반)
        # input_dim은 기본값 사용, pretrain에서 실제 데이터로 업데이트 가능
        self.input_dim = cfg.get("input_dim", 8)  # 기본값, pretrain에서 실제 값으로 업데이트 가능
        
        # LSTM 레이어 즉시 정의 (TechnicalAgent/MacroAgent 패턴)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.feature_cols: List[str] = []  # ctx에서 써먹을 용도
            
    def run_dataset(self, save_dir: str | None = None) -> "StockData":
        """
        core.data_set.load_dataset()를 호출해서
        (X, y, feature_cols)를 로드하고,
        BaseAgent에서 사용하는 StockData 형태로 감싸서 self.stockdata에 저장.

        반환값: StockData 인스턴스
        """

        # 1) save_dir 기본값 설정
        if save_dir is None:
            # config.agents.dir_info 에서 처리 경로를 관리하고 있다면 우선 사용
            # (키 이름은 프로젝트 상황에 따라 다를 수 있음)
            save_dir = (
                dir_info.get("processed_dir")
                or dir_info.get("data_processed")
                or "data/processed"
            )

        # 2) 공통 데이터셋 로더 호출
        # 시그니처: load_dataset(ticker: str, agent_id: str, save_dir: str)
        X, y, feature_cols = load_dataset(
            ticker=self.ticker,
            agent_id=self.agent_id,
            save_dir=save_dir,
        )

        # 3) StockData 래핑 (positional 인자로만 전달)
        stock_data = StockData(X, y, feature_cols)

        # 4) BaseAgent 쪽에서 사용할 수 있도록 저장
        self.stockdata = stock_data

        return stock_data

    # -----------------------------------------------------------
    # 모델 관련 유틸
    # -----------------------------------------------------------
    def _build_model(self):
        """
        TechnicalAgent 패턴: nn.Module이면 자기 자신 반환
        이미 __init__에서 레이어가 정의되어 있으므로 재생성 로직만 처리
        """
        # input_dim이 실제 데이터와 다를 경우 레이어 재생성
        try:
            X, y, cols = _load_dataset_compat(
                self.ticker,
                self.agent_id,
                window_size=self.window_size,
            )
            actual_input_dim = X.shape[-1]
            
            if actual_input_dim != self.input_dim:
                # input_dim이 다르면 레이어 재생성
                print(f"[INFO] input_dim 불일치 감지: {self.input_dim} -> {actual_input_dim}, 레이어 재생성")
                self.input_dim = actual_input_dim
                self.feature_cols = list(cols)
                
                self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
                self.fc = nn.Linear(self.hidden_dim, 1)
            else:
                self.feature_cols = list(cols)
        except Exception:
            # 데이터셋이 없다면 나중에 pretrain에서 처리
            pass
        
        return self  # 자기 자신 반환 (TechnicalAgent 패턴)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델 forward pass
        입력: x (B, T, F)
        출력: (B, 1) - 다음날 수익률(return)
        """
        # x: [B, T, F]
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])  # 마지막 시점 사용
        out = self.fc(out)  # [B, 1]  ← "예측된 수익률" (return)
        return out

    # -----------------------------------------------------------
    # TechnicalAgent 패턴: searcher, pretrain, predict 구현
    # -----------------------------------------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """SentimentalAgent 전용 searcher - load_dataset 사용"""
        import yfinance as yf
        import pandas as pd
        
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        self.ticker = ticker
        
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")
        cfg = agents_info.get(self.agent_id, {})
        
        need_build = rebuild or (not os.path.exists(dataset_path))
        if need_build:
            print(f"⚙️ {ticker} {agent_id} dataset not found. Building new dataset..." if not os.path.exists(dataset_path) else f"⚙️ {ticker} {agent_id} rebuild requested. Building dataset...")
            from core.data_set import build_dataset
            build_dataset(
                ticker=ticker,
                agent_id=agent_id,
                save_dir=self.data_dir,
            )
        
        # CSV 로드
        X, y, feature_cols = load_dataset(
            ticker=ticker,
            agent_id=agent_id,
            save_dir=self.data_dir,
        )
        
        # 최근 window
        X_latest = X[-1:]
        
        # StockData 구성
        self.stockdata = StockData(ticker=ticker)
        self.stockdata.feature_cols = feature_cols
        
        # last_price 안전 변환
        try:
            data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
            if data is not None and not data.empty:
                last_val = data["Close"].iloc[-1]
                self.stockdata.last_price = float(last_val.item() if hasattr(last_val, "item") else last_val)
            else:
                self.stockdata.last_price = None
        except Exception:
            self.stockdata.last_price = None
        
        # 통화코드
        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception:
            self.stockdata.currency = "USD"
        
        # feature_dict 구성
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)  # (T, F)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)
        
        # StockData 생성 완료 (로그는 DebateAgent에서 처리)
        
        return torch.tensor(X_latest, dtype=torch.float32)
    
    def pretrain(self):
        """Agent별 사전학습 루틴 (모델 생성, 학습, 저장, self.model 연결까지 포함)"""
        from torch.utils.data import DataLoader, TensorDataset
        from datetime import datetime
        
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]
        
        # 데이터 로드
        X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 타깃 스케일 조정 - 상승/하락율을 100배로 스케일링
        y_train *= 100.0
        y_val *= 100.0
        
        self.scaler.fit_scalers(X_train, y_train)
        self.scaler.save(self.ticker)
        
        X_train, y_train = map(torch.tensor, self.scaler.transform(X_train, y_train))
        X_train, y_train = X_train.float(), y_train.float()
        
        # 모델 준비 - 이미 __init__에서 생성되었지만, input_dim이 다를 경우 재생성
        model = self
        self._modules.pop("model", None)
        
        # input_dim이 실제 데이터와 다를 경우 레이어 재생성
        actual_input_dim = X_train.shape[-1]
        if actual_input_dim != self.input_dim:
            print(f"[INFO] input_dim 불일치 감지: {self.input_dim} -> {actual_input_dim}, 레이어 재생성")
            self.input_dim = actual_input_dim
            self.feature_cols = list(cols)
            
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, 1)
        else:
            self.feature_cols = list(cols)
        
        # 학습
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.HuberLoss(delta=1.0)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train.view(-1, 1)),
                                  batch_size=batch_size, shuffle=True)
        
        # 학습 루프
        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")
        
        # 모델 저장 및 연결
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        
        # nn.Module 자기 자신이면 self.model에 등록하지 않음
        if model is not self:
            self.model = model
        
        print(f" {self.agent_id} 모델 학습 및 저장 완료: {model_path}")

    def predict(self, X, n_samples: int = 30, current_price: float = None, X_last: np.ndarray = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)
        TechnicalAgent 패턴
        """
        # 모델 준비 및 스케일러 로드
        # 과거 자기참조(child) 정리 - RecursionError 방지
        if isinstance(self, nn.Module):
            for name, child in list(getattr(self, "_modules", {}).items()):
                if child is self:
                    del self._modules[name]
            if getattr(self, "model", None) is self:
                self.model = None

        # 이 에이전트가 nn.Module이면 그 자체 사용
        if isinstance(self, nn.Module) and hasattr(self, "forward"):
            model = self
        else:
            if self.model is None or not hasattr(self.model, "parameters"):
                model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
                if os.path.exists(model_path):
                    self.load_model(model_path)
                else:
                    self.pretrain()
            if self.model is None:
                raise RuntimeError(f"{self.agent_id} 모델이 초기화되지 않음")
            model = self.model

        self.scaler.load(self.ticker)

        # 입력 변환
        if isinstance(X, np.ndarray):
            X_raw_np = X.copy()
            X_scaled, _ = self.scaler.transform(X_raw_np)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_raw_np = X.detach().cpu().numpy().copy()
            X_scaled, _ = self.scaler.transform(X_raw_np)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        device = next(model.parameters()).device
        X_tensor = X_tensor.to(device)

        # Monte Carlo Dropout 추론
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor).cpu().numpy().flatten()
                preds.append(y_pred)

        preds = np.stack(preds)  # (samples, seq)
        mean_pred = preds.mean(axis=0)
        std_pred = np.abs(preds.std(axis=0))  # 항상 양수

        # σ 기반 confidence 계산
        sigma = float(std_pred[-1])
        sigma = max(sigma, 1e-6)

        # 신뢰도: 불확실성 작을수록 1에 가까움
        confidence = 1 / (1 + np.log1p(sigma))

        # 역변환 및 가격 계산
        if hasattr(self.scaler, 'y_scaler') and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        if current_price is None:
            current_price = getattr(self.stockdata, 'last_price', 100.0)

        # 현재 모델은 "다음날 수익률(return)"을 예측하므로, 종가로 변환 시 (1 + return)
        predicted_return = float(mean_pred[-1]) / 100.0  # 예측된 상승률 (%)
        predicted_price = current_price * (1 + predicted_return)

        # Target 생성 및 반환
        target = Target(
            next_close=float(predicted_price),
            uncertainty=sigma,
            confidence=float(confidence),
        )

        return target

    # -----------------------------------------------------------
    # MC Dropout 기반 helper (기존 기능 유지 - 하위 호환성)
    # -----------------------------------------------------------
    @torch.inference_mode()
    def _mc_dropout_predict(self, x: torch.Tensor, T: int = 30) -> Tuple[float, float]:
        """
        x: [1, T, F]
        반환: (mean_return, std_return)
        """
        model = self if isinstance(self, nn.Module) else self.model
        if model is None:
            raise RuntimeError("model is None for MC Dropout")

        model.train()
        outs = []
        with torch.no_grad():
            for _ in range(T):
                outs.append(model(x).detach())
        model.eval()

        y = torch.stack(outs, dim=0).squeeze(-1)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        return float(mean.squeeze().item()), float(std.squeeze().item())

    @torch.inference_mode()
    def _predict_next_close(self) -> Tuple[float, float, float, List[str]]:
        """
        LSTM 출력은 "다음날 수익률(return)"로 가정하고,
        마지막 종가(last_close)에 곱해 다음 종가(pred_close)를 계산한다.
        반환:
            pred_close      : 예측된 다음 종가 (price)
            uncertainty_std : 예측 수익률(return)의 표준편차
            confidence      : 1 / (1 + uncertainty_std)
            cols            : feature 컬럼 리스트
        """
        if not self.ticker:
            raise ValueError("ticker is None in _predict_next_close")

        try:
            X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
        except Exception:
            _build_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
            X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)

        # 마지막 종가 추출
        last_close_idx = cols.index("Close") if "Close" in cols else -1
        last_close = None
        if last_close_idx >= 0:
            last_close = float(X[-1, -1, last_close_idx])

        # last_close가 없으면 fallback 값 (이 경우엔 네트워크 출력을 그대로 price처럼 취급)
        if last_close is None or not (last_close == last_close):
            last_close = None

        # 모델이 없으면: 마지막 종가 그대로 사용 (수익률 0 가정)
        if self.model is None:
            pred_close = float(last_close) if last_close is not None else float("nan")
            uncertainty_std = 0.10  # return 기준 대략 10% 정도로 가정
            confidence = 1.0 / (1.0 + uncertainty_std)
            return pred_close, uncertainty_std, confidence, cols

        # 모델이 있으면: 수익률(return) 예측 후 price로 변환
        # 스케일러 적용
        self.scaler.load(self.ticker)
        X_scaled, _ = self.scaler.transform(X[-1:])
        x_last = torch.tensor(X_scaled, dtype=torch.float32)
        mean_ret, std_ret = self._mc_dropout_predict(x_last, T=30)

        if last_close is not None:
            pred_close = float(last_close * (1.0 + mean_ret))
        else:
            # 어쩔 수 없이 return 값을 그대로 price처럼 사용 (이 경우는 거의 없을 것)
            pred_close = float(mean_ret)

        uncertainty_std = float(std_ret)  # "수익률"의 표준편차
        confidence = float(1.0 / (1.0 + max(1e-6, uncertainty_std)))

        # 디버깅용 로그 (필요하면 주석 해제)
        # print(f"[SentimentalAgent] last_close={last_close}, mean_ret={mean_ret}, pred_close={pred_close}")

        return pred_close, uncertainty_std, confidence, cols

    # -----------------------------------------------------------
    # ctx 생성 (FinBERT + 가격 스냅샷)
    # -----------------------------------------------------------
    def build_ctx(self, asof_date_kst: Optional[str] = None) -> Dict[str, Any]:
        if asof_date_kst is None:
            asof_date_kst = datetime.now().strftime("%Y-%m-%d")

        pred_close, uncertainty_std, confidence, cols = self._predict_next_close()

        price_snapshot: Dict[str, Optional[float]] = {}
        try:
            X, _, cols2 = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
            last = X[-1, -1, :]
            snap_map = {c: float(v) for c, v in zip(cols2, last)}
            for k in ("Close", "Open", "High", "Low", "Volume", "returns"):
                if k in snap_map:
                    price_snapshot[k] = snap_map[k]
        except Exception:
            pass

        news_feats = build_finbert_news_features(
            self.ticker, asof_date_kst, base_dir=os.path.join("data", "raw", "news")
        )

        snapshot = {
            "asof_date": asof_date_kst,
            "last_price": price_snapshot.get("Close", np.nan),
            "currency": "USD",
            "window_size": self.window_size,
            "feature_cols_preview": [c for c in (cols or [])[:8]],
        }

        last_price = snapshot["last_price"]
        pred_return = float(pred_close / last_price - 1.0) if (last_price and last_price == last_price) else None

        feature_importance = {
            "sentiment_score": news_feats["sentiment_summary"]["mean_7d"],
            "sentiment_summary": news_feats["sentiment_summary"],
            "sentiment_volatility": {"vol_7d": news_feats["sentiment_volatility"].get("vol_7d", 0.0)},
            "trend_7d": news_feats["trend_7d"],
            "news_count": news_feats["news_count"],
            "has_news": news_feats.get("has_news", False),
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
                "uncertainty": {"std": uncertainty_std, "ci95": float(1.96 * uncertainty_std)},
                "confidence": confidence,
                "pred_next_close": pred_close,
            },
            "feature_importance": feature_importance,
        }
        return ctx

    # -----------------------------------------------------------
    # TechnicalAgent 패턴: reviewer_draft, reviewer_rebut, reviewer_revise 구현
    # -----------------------------------------------------------
    def reviewer_draft(self, stock_data: StockData = None, target: Target = None) -> Opinion:
        """(1) searcher → (2) predicter → (3) LLM(JSON Schema)로 reason 생성 → Opinion 반환"""
        
        # 1) 데이터 수집
        if stock_data is None:
            stock_data = self.stockdata
        
        # 2) 예측값 생성
        if target is None:
            X_input = self.searcher(self.ticker)  # (1,T,F)
            target = self.predict(X_input)
        
        # 3) LLM 호출(reason 생성) - 전달받은 stock_data 사용
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)
        
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}
        )
        
        reason = parsed.get("reason", "(사유 생성 실패)")
        
        # 4) Opinion 기록/반환 (항상 최신 값 append)
        self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason))
        
        # 최신 오피니언 반환
        return self.opinions[-1]
    
    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""
        
        # 메시지 생성 (context 구성은 별도 헬퍼에서)
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )
        
        # LLM 호출
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"}
                },
                "required": ["stance", "message"],
                "additionalProperties": False
            }
        )
        
        # 결과 정리 및 기록
        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )
        
        # 저장
        self.rebuttals[round].append(result)
        
        # 디버깅 로그
        if self.verbose:
            print(
                f"[{self.agent_id}] rebuttal 생성 → {result.stance} "
                f"({my_opinion.agent_id} → {other_opinion.agent_id})"
            )
        
        return result
    
    # DebateAgent.get_rebuttal() 호환용 래퍼
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
    
    def reviewer_revise(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: List[Rebuttal],
        stock_data: StockData,
        fine_tune: bool = True,
        lr: float = 1e-4,
        epochs: int = 20,
    ):
        """
        Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - fine-tuning (수익률 단위)
        - reasoning 생성
        """
        gamma = getattr(self, "gamma", 0.3)  # 수렴율 (0~1)
        delta_limit = getattr(self, "delta_limit", 0.05)  # fine-tuning 보정 한계
        
        try:
            # β 계산 (불확실성 작을수록 신뢰 높음)
            my_price = my_opinion.target.next_close
            my_sigma = abs(my_opinion.target.uncertainty or 1e-6)
            
            other_prices = np.array([o.target.next_close for o in others])
            other_sigmas = np.array([abs(o.target.uncertainty or 1e-6) for o in others])
            
            all_sigmas = np.concatenate([[my_sigma], other_sigmas])
            all_prices = np.concatenate([[my_price], other_prices])
            
            inv_sigmas = 1 / (all_sigmas + 1e-6)
            betas = inv_sigmas / inv_sigmas.sum()
            
            # 논문식 수렴 업데이트
            # y_i_rev = y_i + γ Σ β_j (y_j - y_i)
            delta = np.sum(betas[1:] * (other_prices - my_price))
            revised_price = my_price + gamma * delta
            
        except Exception as e:
            print(f"[{self.agent_id}] revised_target 계산 실패: {e}")
            revised_price = my_opinion.target.next_close
        
        # fine-tuning (선택적)
        loss_value = 0.0
        if fine_tune and isinstance(self, nn.Module):
            try:
                # revised_return 계산
                current_price = getattr(stock_data, 'last_price', my_price)
                revised_return = (revised_price / current_price - 1.0) if current_price else 0.0
                revised_return_scaled = revised_return * 100.0  # 스케일링
                
                # 입력 준비
                X_seq = self.searcher(self.ticker)
                if isinstance(X_seq, torch.Tensor):
                    X_seq = X_seq.to(next(self.parameters()).device)
                else:
                    X_seq = torch.tensor(X_seq, dtype=torch.float32).to(next(self.parameters()).device)
                
                y_true = torch.FloatTensor([[revised_return_scaled]]).to(next(self.parameters()).device)
                
                # Fine-tuning
                self.train()
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)
                criterion = nn.MSELoss()
                
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    pred = self(X_seq)
                    loss = criterion(pred, y_true)
                    loss.backward()
                    optimizer.step()
                    loss_value = loss.item()
                
                self.eval()
                
            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")
        
        # 새로운 Target 생성
        new_target = Target(
            next_close=float(revised_price),
            uncertainty=my_opinion.target.uncertainty,
            confidence=my_opinion.target.confidence,
        )
        
        # LLM으로 수정된 reason 생성
        sys_text, user_text = self._build_messages_revision(
            my_opinion=my_opinion,
            others=others,
            rebuttals=rebuttals,
            stock_data=stock_data
        )
        
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False
            },
        )
        
        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=new_target,
            reason=revised_reason,
        )
        
        self.opinions.append(revised_opinion)
        print(f"[{self.agent_id}] revise 완료 → new_close={new_target.next_close:.2f}, loss={loss_value}")
        return self.opinions[-1]

    # --------------------------
    # BaseAgent 규약 충족: Opinion 프롬프트
    # --------------------------
    def _build_messages_opinion(
        self,
        stock_data: "StockData",
        target: "Target",
    ):
        from prompts import OPINION_PROMPTS  # 상단에 이미 있으면 생략 가능
        import json
        from typing import Dict, Any
        import numpy as np

        if stock_data is None:
            stock_data = self.stockdata

        ctx: Dict[str, Any] = {}

        # 기본 메타 정보
        ctx["ticker"] = getattr(stock_data, "ticker", self.ticker)
        ctx["currency"] = getattr(stock_data, "currency", "USD")

        # 현재가 / 다음날 예측 종가
        last_close = getattr(stock_data, "last_price", None)
        ctx["last_close"] = last_close
        ctx["next_close"] = float(getattr(target, "next_close", None) or 0.0)

        # 예상 수익률 (비율)
        change_ratio = None
        if isinstance(last_close, (int, float)) and last_close not in (0, None):
            try:
                change_ratio = ctx["next_close"] / float(last_close) - 1.0
            except ZeroDivisionError:
                change_ratio = None
        ctx["change_ratio"] = change_ratio

        # 예측 불확실성 / 신뢰도
        ctx["uncertainty_std"] = getattr(target, "uncertainty", None)
        ctx["confidence"] = getattr(target, "confidence", None)

        # --- SentimentalAgent 전용 스냅샷 주입 ---
        # 이전 코드: snap = getattr(stock_data, "SentimentalAgent", {}) or {}
        # → numpy 배열일 때 truth value 에러 발생하므로 수정
        snap = getattr(stock_data, "SentimentalAgent", None)

        if isinstance(snap, dict):
            for k, v in snap.items():
                # numpy 배열이면 마지막 값 위주로 사용
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
                # 리스트/튜플이면 마지막 값 사용
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    ctx[k] = v[-1]
                else:
                    ctx[k] = v
        # dict 가 아니면(예: np.array 직접 할당) 그냥 무시

        # JSON 직렬화
        ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)

        # 프롬프트 로딩
        prompts = OPINION_PROMPTS.get("SentimentalAgent", {}) if OPINION_PROMPTS else {}
        system_text = prompts.get("system", "너는 감성/뉴스 중심의 단기 주가 분석가다.")
        user_tmpl = prompts.get(
            "user",
            "ctx(JSON):\n{context}\n\n위 ctx를 바탕으로 reason을 생성하라.",
        )

        # 템플릿 포맷 안전 처리
        try:
            user_text = user_tmpl.format(context=ctx_json)
        except KeyError:
            try:
                user_text = user_tmpl.format(ctx_json)
            except Exception:
                user_text = user_tmpl.replace("{context}", ctx_json)

        return system_text, user_text

    # --------------------------
    # BaseAgent 규약 충족: Rebuttal 프롬프트
    # --------------------------
    def _build_messages_rebuttal(
        self,
        my_opinion: Opinion,
        target_opinion: Opinion,
        stock_data: StockData
    ) -> Tuple[str, str]:
        """TechnicalAgent 패턴: 시그니처 통일"""
        
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

    # --------------------------
    # BaseAgent 규약 충족: Revision 프롬프트
    # --------------------------
    def _build_messages_revision(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: Optional[List[Rebuttal]] = None,
        stock_data: StockData = None,
    ) -> Tuple[str, str]:
        """TechnicalAgent 패턴: 시그니처 통일"""
        
        if stock_data is None:
            stock_data = self.stockdata
        
        # Opinion/Dict/str/Rebuttal → text
        def _op_text(x: Union[Opinion, Dict[str, Any], str, None, Any]) -> str:
            if isinstance(x, Opinion):
                return getattr(x, "reason", "")
            if isinstance(x, dict):
                return x.get("reason", "") or x.get("message", "")
            # Rebuttal 객체 처리
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

        # 공통 ctx 로딩
        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        # 불확실성/신뢰도 (prediction 블록에서 가져옴)
        pred_info = ctx.get("prediction", {}) or {}
        unc_dict = pred_info.get("uncertainty", {}) or {}
        unc_std = unc_dict.get("std", None)
        confidence = pred_info.get("confidence", None)

        pred_close = float(my_opinion.target.next_close)
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        # === 감성/예측 요약 context 문자열 생성 ===
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
                f"Monte Carlo Dropout 기반 예측 표준편차는 {unc_std:.4f}, 신뢰도는 {confidence:.3f}입니다."
            )

        context_str = " ".join(context_parts) if context_parts else (
            "최근 뉴스 감성 점수, 변동성, 긍·부정 비율, 뉴스 수, 예측 불확실성 등을 종합해 단기 주가를 해석합니다."
        )

        # ==========================
        # 프롬프트 템플릿 선택
        # ==========================
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
            # 기본 템플릿은 context 없이 동작 (기존 형식 유지)
            user_tmpl = (
                "티커: {ticker}\n"
                "초안 의견:\n{prev}\n\n"
                "수신한 반박 요약:\n{rebuts}\n\n"
                "업데이트된 수치:\n- next_close: {pred_close}\n- 예상 변화율: {chg}\n"
                "감성 근거 스냅샷:\n- mean7={mean7}, mean30={mean30}, pos7={pos7}, neg7={neg7}\n"
                "- vol7={vol7}, trend7={trend7}, news7={news7}\n\n"
                "요청: 초안에서 과장/중복/약한 근거를 정리하고, 강한 근거(감성 추세, 변동성 안정/확대, 뉴스 수 변화)를 중심으로 "
                "최종 의견을 3~5문장으로 재작성하세요. 불확실성(표준편차)/신뢰도 해석을 함께 제시하세요."
            )

        rebuts_joined = "- " + "\n- ".join([s for s in reb_texts if s]) if reb_texts else "(반박 없음)"

        # 🔴 핵심: context를 항상 함께 넘겨줌
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
            context=context_str,  # ✅ REVISION_PROMPTS에서 {context}를 써도 안전하게
        )
        return system_tmpl, user_text

    # Opinion 생성 (legacy 경로)
    # --------------------------
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
