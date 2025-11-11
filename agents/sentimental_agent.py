# agents/sentimental_agent.py
# ===============================================================
# SentimentalAgent: 감성(뉴스/텍스트) + LSTM 기반 예측 에이전트
#  - BaseAgent 호환 (reviewer_* 로직은 BaseAgent 구현 사용)
#  - 데이터는 core/data_set.py에서 만든 CSV를 로드
#  - FinBERT 기반 뉴스 감성 피처를 ctx.feature_importance에 주입
#  - 반영사항:
#      1) super().__init__(agent_id, ticker) 순서 고정 + ticker 가드/정규화
#      2) load_dataset/build_dataset 시그니처 호환 유틸 추가
#      3) finbert_utils 단일 경로(core.finbert_utils)로 고정
#      4) 뉴스 캐시 파일 폴백(glob 최신 파일 선택) 추가
#      5) 모델 state_dict 언랩(model_state_dict) + strict=False 로드
#      6) Target 생성 시 3개 인자만 전달
#      7) MC Dropout로 uncertainty/confidence 실제 계산
#      8) sentiment_vol_30 노출 제거 (vol_7d만 제공)
# ===============================================================

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from pathlib import Path
from datetime import datetime, timedelta, date

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------
# 프로젝트 의존 모듈 (안전 import)
# ---------------------------
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

# FinBERT 유틸 (단일 경로로 고정)
from core.finbert_utils import (
    FinBertScorer,
    score_news_items,
    attach_scores_to_items,
    compute_finbert_features,
)

# ---------------------------------------------------------------
# 모델 정의: LSTM + Dropout (MC Dropout 지원)
# ---------------------------------------------------------------
class SentimentalNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        out = self.fc(out)  # [B, 1]
        return out


# ---------------------------------------------------------------
# load/build dataset 시그니처 호환 유틸
# ---------------------------------------------------------------
def _load_dataset_compat(ticker: str, agent_id: str, window_size: Optional[int] = None):
    """
    load_dataset 시그니처가 팀별로 다른 경우를 호환.
    시도 순서:
      1) load_dataset(ticker, agent_id, window_size=..)
      2) load_dataset(ticker, agent_id, seq_len=..)
      3) load_dataset(ticker, agent_id)
    """
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
    """
    build_dataset 시그니처 호환: window_size/seq_len/무인자 순으로 시도.
    """
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
    """
    asof_kst(YYYY-MM-DD) 기준으로 UTC 날짜 범위를 생성
      - 미래 컷오프 방지 위해 to_utc = (kst-9h).date() - 1day
      - from_utc = to_utc - lookback_days
    반환: (from_utc_str, to_utc_str, to_utc_date)
    """
    kst_dt = datetime.fromisoformat(asof_kst)
    utc_today = (kst_dt - timedelta(hours=9)).date()
    to_utc_date = utc_today - timedelta(days=1)
    from_utc_date = to_utc_date - timedelta(days=lookback_days)
    return from_utc_date.isoformat(), to_utc_date.isoformat(), to_utc_date


def build_finbert_news_features(
    ticker: str,
    asof_kst: str,
    base_dir: str = "data/raw/news",
    text_fields: Tuple[str, ...] = ("title", "content", "text", "summary"),
) -> Dict[str, Any]:
    """
    저장된 뉴스 JSON을 읽고 FinBERT로 감성 점수 계산 후 요약 피처 반환
    - diagnostics_news.py가 만들어둔 파일명 포맷을 그대로 사용
    - 캐시 미존재 시 해당 티커의 최신 파일로 폴백
    """
    fr, to, to_date_utc = _utc_from_kst_asof(asof_kst, lookback_days=40)
    symbol_us = f"{ticker}.US"
    base = Path(base_dir)
    path = base / f"{symbol_us}_{fr}_{to}.json"

    if not path.exists():
        # 폴백: 동일 티커의 최신 파일 자동 선택
        cands = sorted(base.glob(f"{symbol_us}_*.json"))
        if cands:
            latest = cands[-1]
            print(f"[FinBERT] 캐시 미발견 → 최신 파일 사용: {latest.name}")
            path = latest
        else:
            print(f"[FinBERT] 뉴스 캐시 없음: {path}")
            return {
                "sentiment_summary": {"mean_7d": 0.0, "mean_30d": 0.0, "pos_ratio_7d": 0.0, "neg_ratio_7d": 0.0},
                "sentiment_volatility": {"vol_7d": 0.0},
                "news_count": {"count_1d": 0, "count_7d": 0},
                "trend_7d": 0.0,
                "has_news": False,
            }

    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        print(f"[FinBERT] 캐시 형식 경고(list 아님): {path}")
        return {
            "sentiment_summary": {"mean_7d": 0.0, "mean_30d": 0.0, "pos_ratio_7d": 0.0, "neg_ratio_7d": 0.0},
            "sentiment_volatility": {"vol_7d": 0.0},
            "news_count": {"count_1d": 0, "count_7d": 0},
            "trend_7d": 0.0,
            "has_news": False,
        }

    # 날짜 필드 비문자 방어 (공급자 변형/None 방지)
    for it in items:
        for k in ("date", "published_date", "time", "pubDate"):
            if not isinstance(it.get(k), str):
                it[k] = ""

    print(f"[FinBERT] {len(items)}건 뉴스 감성 분석 시작... ({path.name})")
    scorer = FinBertScorer()
    scores = score_news_items(items, scorer=scorer, text_fields=text_fields)
    items_scored = attach_scores_to_items(items, scores)

    feats = compute_finbert_features(items_scored, asof_utc_date=to_date_utc)

    # ▶ vol_30d는 버리고 vol_7d만 남김
    vol7 = feats.get("sentiment_volatility", {}).get("vol_7d", 0.0)
    feats["sentiment_volatility"] = {"vol_7d": vol7}

    print(
        f"[FinBERT] 7d_mean={feats['sentiment_summary']['mean_7d']:.3f} "
        f"7d_cnt={feats['news_count']['count_7d']}"
    )
    return feats


# ---------------------------------------------------------------
# 본체: SentimentalAgent
# ---------------------------------------------------------------
class SentimentalAgent(BaseAgent):  # type: ignore
    agent_id: str = "SentimentalAgent"

    def __init__(self, ticker: str, **kwargs):
        # ✅ BaseAgent.__init__(agent_id, ticker, ...) 순서
        try:
            super().__init__(self.agent_id, ticker, **kwargs)  # type: ignore
        except TypeError:
            super().__init__(agent_id=self.agent_id, ticker=ticker, **kwargs)  # type: ignore

        # 🔧 ticker 가드/정규화
        if not getattr(self, "ticker", None):
            self.ticker = ticker
        if self.ticker is None or str(self.ticker).strip() == "":
            raise ValueError("SentimentalAgent: ticker is None/empty")
        self.ticker = str(self.ticker).upper()
        setattr(self, "symbol", self.ticker)

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
        self.window_size = cfg.get("window_size", 40)
        self.hidden_dim = cfg.get("hidden_dim", 128)
        self.dropout = cfg.get("dropout", 0.2)

        # 모델 로드 (데이터셋 로드 후 input_dim 파악)
        self.model: Optional[nn.Module] = None
        try:
            self._load_model_if_exists()
        except Exception as e:
            print("[SentimentalAgent] 모델 로드 스킵:", e)

    # --------------------------
    # 모델 로드/세이브
    # --------------------------
    def model_path(self) -> str:
        mdir = dir_info.get("model_dir", "models")
        Path(mdir).mkdir(parents=True, exist_ok=True)
        return os.path.join(mdir, f"{self.ticker}_{self.agent_id}.pt")

    def _load_model_if_exists(self):
        p = self.model_path()
        if os.path.exists(p):
            if not self.ticker:
                raise ValueError("ticker is None in _load_model_if_exists")

            try:
                X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
            except Exception:
                _build_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
                X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)

            input_dim = X.shape[-1]
            net = SentimentalNet(input_dim=input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout)

            sd = torch.load(p, map_location="cpu")
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            net.load_state_dict(sd, strict=False)  # 관용 로드
            net.eval()
            self.model = net
            print(f"✅ {self.ticker} {self.agent_id} 모델 로드 완료 ({p})")

    # --------------------------
    # MC Dropout 유틸
    # --------------------------
    @torch.inference_mode()
    def _mc_dropout_predict(self, x: torch.Tensor, T: int = 30) -> Tuple[float, float]:
        """
        MC Dropout로 예측 분포(mean, std) 추정
        - 모델은 dropout 레이어를 포함해야 함
        """
        if self.model is None:
            raise RuntimeError("model is None for MC Dropout")

        self.model.train()  # Dropout 활성화
        outs = []
        for _ in range(T):
            outs.append(self.model(x).detach())  # [B, 1]
        self.model.eval()

        y = torch.stack(outs, dim=0).squeeze(-1)  # [T, B]
        mean = y.mean(dim=0)                      # [B]
        std = y.std(dim=0)                        # [B]
        return float(mean.squeeze().item()), float(std.squeeze().item())

    # --------------------------
    # 예측 (MC Dropout 사용)
    # --------------------------
    @torch.inference_mode()
    def _predict_next_close(self) -> Tuple[float, float, float, List[str]]:
        """
        반환: (pred_close, uncertainty_std, confidence, feature_cols)
        """
        if not self.ticker:
            raise ValueError("ticker is None in _predict_next_close")

        try:
            X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
        except Exception:
            _build_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
            X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)

        # 모델 없으면 마지막 close 폴백
        last_close_idx = cols.index("Close") if "Close" in cols else -1
        fallback = float(X[-1, -1, last_close_idx]) if last_close_idx >= 0 else float("nan")

        if self.model is None:
            pred_close = fallback
            uncertainty_std = 0.10
            confidence = 1.0 / (1.0 + uncertainty_std)
            return pred_close, uncertainty_std, confidence, cols

        x_last = torch.tensor(X[-1:]).float()  # [1, T, F]

        # ▶ MC Dropout로 mean/std 추정
        pred_close, uncertainty_std = self._mc_dropout_predict(x_last, T=30)

        # 간단 confidence 스케일링 (0~1): std가 작을수록 ↑
        confidence = float(1.0 / (1.0 + max(1e-6, uncertainty_std)))

        return pred_close, uncertainty_std, confidence, cols

    # --------------------------
    # ctx 생성: 여기서 FinBERT 뉴스 피처를 주입한다
    # --------------------------
    def build_ctx(self, asof_date_kst: Optional[str] = None) -> Dict[str, Any]:
        if asof_date_kst is None:
            # KST 기준 "오늘" 날짜 문자열
            asof_date_kst = datetime.now().strftime("%Y-%m-%d")

        pred_close, uncertainty_std, confidence, cols = self._predict_next_close()

        # 가격 스냅샷(가능한 범위에서 수집)
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

        # FinBERT 뉴스 피처
        news_feats = build_finbert_news_features(
            self.ticker, asof_date_kst, base_dir=os.path.join("data", "raw", "news")
        )

        # 스냅샷 상단부
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

    # --------------------------
    # Opinion 생성 (BaseAgent 규약을 따르거나, 폴백 제공)
    # --------------------------
    def get_opinion(self, idx: int = 0, ticker: Optional[str] = None) -> Opinion:  # type: ignore[override]
        """
        DebateAgent에서 호출되는 진입점일 가능성이 높음.
        - 가능한 한 BaseAgent의 LLM 프롬프트 흐름을 사용
        - 없다면 간단 reason을 만들어 Opinion 반환
        """
        _ = idx
        if ticker and ticker != self.ticker:
            self.ticker = str(ticker).upper()

        ctx = self.build_ctx()

        # BaseAgent가 LLM 기반 의견 생성기를 제공한다면 사용
        try:
            if hasattr(self, "reviewer_opinion"):
                op: Opinion = self.reviewer_opinion(ctx=ctx)  # type: ignore
                return op
        except Exception as e:
            print("[SentimentalAgent] reviewer_opinion 사용 실패:", e)

        # 폴백: 간단 reason 구성
        fi = ctx["feature_importance"]
        sent = fi["sentiment_summary"]
        reason = (
            f"{self.ticker}의 최근 7일 감성 평균은 {sent['mean_7d']:.3f}이며 "
            f"뉴스 개수(7d)는 {fi['news_count']['count_7d']}건입니다. "
            f"감성 변동성(vol_7d)={fi['sentiment_volatility']['vol_7d']:.3f}, "
            f"감성 추세(trend_7d)={fi['trend_7d']:.3f}입니다."
        )
        target = Target(
            next_close=float(ctx["prediction"]["pred_next_close"]),
            uncertainty=float(ctx["prediction"]["uncertainty"]["std"]),
            confidence=float(ctx["prediction"]["confidence"]),
        )
        return Opinion(agent_id=self.agent_id, target=target, reason=reason)
