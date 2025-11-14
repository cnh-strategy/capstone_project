# agents/sentimental_agent.py
# ===============================================================
# SentimentalAgent: 감성(뉴스/텍스트) + LSTM 기반 예측 에이전트
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
# FinBERT 유틸 (단일 경로로 고정)
try:
    from core.sentimental_classes.finbert_utils import (
        FinBertScorer,
        score_news_items,
        attach_scores_to_items,
        compute_finbert_features,
    )
except ImportError:
    # 🔔 FinBERT 관련 심볼이 없을 때: 기능을 비활성화하고, 깨지지 않도록 더미 함수 제공
    print(
        "[warn] core.sentimental_classes.finbert_utils 에서 FinBertScorer 등을 찾지 못했습니다. "
        "SentimentalAgent에서 FinBERT 관련 기능이 비활성화됩니다."
    )

    FinBertScorer = None  # type: ignore

    def score_news_items(items, *args, **kwargs):
        """FinBERT가 없을 때는 입력을 그대로 반환."""
        print("[warn] score_news_items 호출됨 (FinBERT 비활성화) → 입력 그대로 반환")
        return items

    def attach_scores_to_items(items, scores=None, *args, **kwargs):
        """점수는 무시하고 items 그대로 반환."""
        print("[warn] attach_scores_to_items 호출됨 (FinBERT 비활성화) → items 그대로 반환")
        return items

    def compute_finbert_features(df, *args, **kwargs):
        """
        df: 뉴스 데이터프레임
        반환: (df, features_dict)
        FinBERT 비활성화 시에는 df와 빈 feature dict를 반환.
        """
        print("[warn] compute_finbert_features 호출됨 (FinBERT 비활성화) → 원본 df, 빈 feature dict 반환")
        return df, {}


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


def build_finbert_news_features(
    ticker: str,
    asof_kst: str,
    base_dir: str = "data/raw/news",
    text_fields: Tuple[str, ...] = ("title", "content", "text", "summary"),
) -> Dict[str, Any]:
    """
    SentimentalAgent 가 사용할 FinBERT 입력용 뉴스 피처 생성.
    - base_dir 가 상대경로면, 항상 '프로젝트 루트/데이터' 기준으로 해석
    """
    fr, to, to_date_utc = _utc_from_kst_asof(asof_kst, lookback_days=40)
    symbol_us = f"{ticker}.US"

    project_root = Path(__file__).resolve().parent.parent
    base = Path(base_dir)
    if not base.is_absolute():
        base = project_root / base

    path = base / f"{symbol_us}_{fr}_{to}.json"
    print(f"[FinBERT] 캐시 탐색: {path} (exists={path.exists()})")

    if not path.exists():
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

    for it in items:
        for k in ("date", "published_date", "time", "pubDate"):
            if not isinstance(it.get(k), str):
                it[k] = ""

    print(f"[FinBERT] {len(items)}건 뉴스 감성 분석 시작... ({path.name})")
    scorer = FinBertScorer()
    scores = score_news_items(items, scorer=scorer, text_fields=text_fields)
    items_scored = attach_scores_to_items(items, scores)

    feats = compute_finbert_features(items_scored, asof_utc_date=to_date_utc)

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

    def __init__(
        self,
        ticker: str | None = None,
        agent_id: str = "SentimentalAgent",
        *args,
        **kwargs,
    ):
        # ✅ BaseAgent 한 번만 초기화
        agent_id = agent_id or "SentimentalAgent"
        super().__init__(agent_id=agent_id, ticker=ticker, *args, **kwargs)

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
        self.window_size = cfg.get("window_size", 40)
        self.hidden_dim = cfg.get("hidden_dim", 128)
        self.dropout = cfg.get("dropout", 0.2)

        self.model: Optional[nn.Module] = None
        self.feature_cols: List[str] = []  # ctx에서 써먹을 용도
        self.model_loaded: bool = False

        # ✅ 초기화 시 자동 모델 로드 (있으면)
        try:
            self._load_model_if_exists()
        except Exception as e:
            # 여기서는 조용히 넘기고, 실제 예측 시 BaseAgent.predict에서 다시 처리
            print(f"[SentimentalAgent] 초기 모델 로드 중 예외 발생 (무시하고 진행): {e}")

    # -----------------------------------------------------------
    # 모델 관련 유틸
    # -----------------------------------------------------------
    def _build_model(self) -> nn.Module:
        """
        BaseAgent.pretrain()에서 호출하는 모델 생성 함수.
        SentimentalNet(LSTM) 구조를 생성해서 반환.
        """
        try:
            X, y, cols = _load_dataset_compat(
                self.ticker,
                self.agent_id,
                window_size=self.window_size,
            )
        except Exception:
            # 데이터셋이 없다면 먼저 생성
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

        net = SentimentalNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        return net

    def model_path(self) -> str:
        try:
            from config.agents import dir_info  # 재-import 방지용
            model_dir = dir_info["model_dir"]
        except Exception:
            model_dir = "models"

        ticker = getattr(self, "ticker", "UNKNOWN")
        agent_id = getattr(self, "agent_id", "SentimentalAgent")
        return os.path.join(model_dir, f"{ticker}_{agent_id}.pt")

    def _load_model_if_exists(self) -> None:
        """
        TechnicalAgent 와 동일한 전략:
        - model_path() 기준으로 파일 존재 시 BaseAgent.load_model() 사용
        - 직접 torch.load / load_state_dict 호출 ❌
        """
        model_path = self.model_path()

        if not os.path.exists(model_path):
            # 모델 파일 없으면 조용히 패스 (경고 X)
            self.model_loaded = False
            return

        ok = False
        try:
            ok = self.load_model(model_path)
        except Exception as e:
            print(f"[SentimentalAgent] 모델 로드 실패: {e}")
            ok = False

        self.model_loaded = bool(ok)

    # -----------------------------------------------------------
    # MC Dropout 기반 helper (기존 기능 유지)
    # -----------------------------------------------------------
    @torch.inference_mode()
    def _mc_dropout_predict(self, x: torch.Tensor, T: int = 30) -> Tuple[float, float]:
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
        if not self.ticker:
            raise ValueError("ticker is None in _predict_next_close")

        try:
            X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
        except Exception:
            _build_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)
            X, y, cols = _load_dataset_compat(self.ticker, self.agent_id, window_size=self.window_size)

        last_close_idx = cols.index("Close") if "Close" in cols else -1
        fallback = float(X[-1, -1, last_close_idx]) if last_close_idx >= 0 else float("nan")

        if self.model is None:
            pred_close = fallback
            uncertainty_std = 0.10
            confidence = 1.0 / (1.0 + uncertainty_std)
            return pred_close, uncertainty_std, confidence, cols

        x_last = torch.tensor(X[-1:]).float()
        pred_close, uncertainty_std = self._mc_dropout_predict(x_last, T=30)
        confidence = float(1.0 / (1.0 + max(1e-6, uncertainty_std)))
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

    # --------------------------
    # BaseAgent 규약 충족: Opinion 프롬프트
    # --------------------------
    def _build_messages_opinion(
        self,
        stock_data: "StockData",
        target: "Target",
    ):
        from prompts import OPINION_PROMPTS  # 상단에 이미 있으면 생략 가능

        if stock_data is None:
            stock_data = self.stockdata

        ctx: Dict[str, Any] = {}

        ctx["ticker"] = getattr(stock_data, "ticker", self.ticker)
        ctx["currency"] = getattr(stock_data, "currency", "USD")
        last_close = getattr(stock_data, "last_price", None)
        ctx["last_close"] = last_close
        ctx["next_close"] = float(getattr(target, "next_close", None) or 0.0)

        change_ratio = None
        if isinstance(last_close, (int, float)) and last_close not in (0, None):
            change_ratio = ctx["next_close"] / float(last_close) - 1.0
        ctx["change_ratio"] = change_ratio

        ctx["uncertainty_std"] = getattr(target, "uncertainty", None)
        ctx["confidence"] = getattr(target, "confidence", None)

        snap = getattr(stock_data, "SentimentalAgent", {}) or {}
        for k, v in snap.items():
            if isinstance(v, (list, tuple)) and len(v) > 0:
                ctx[k] = v[-1]
            else:
                ctx[k] = v

        ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)

        prompts = OPINION_PROMPTS.get("SentimentalAgent", {})
        system_text = prompts.get("system", "너는 감성/뉴스 중심의 단기 주가 분석가다.")
        user_tmpl = prompts.get(
            "user",
            "ctx(JSON):\n{}\n\n위 ctx를 바탕으로 reason을 생성하라.",
        )

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

        pred_close = float(target.next_close) if target else float(ctx["prediction"]["pred_next_close"])
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
    def _build_messages_revision(self, *args, **kwargs) -> Tuple[str, str]:
        stock_data = args[0] if len(args) > 0 else kwargs.get("stock_data")
        target: Optional[Target] = args[1] if len(args) > 1 else kwargs.get("target")

        # previous opinion 찾기
        prev = None
        rebs = None
        for key in ("previous", "previous_opinion", "draft", "opinion"):
            if key in kwargs:
                prev = kwargs[key]
                break
        if prev is None and len(args) > 2:
            prev = args[2]

        # rebuttals 찾기
        for key in ("rebuttals", "replies", "responses"):
            if key in kwargs:
                rebs = kwargs[key]
                break
        if rebs is None and len(args) > 3:
            rebs = args[3]

        # Opinion/Dict/str → text
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

        # 공통 ctx 로딩
        ctx = self.build_ctx()

        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        # 불확실성/신뢰도 (Monte Carlo 결과가 ctx에 있다면 사용)
        uncertainty = ctx.get("uncertainty", {})
        unc_std = uncertainty.get("std", None)
        confidence = uncertainty.get("confidence", None)

        pred_close = float(target.next_close) if target else float(ctx["prediction"]["pred_next_close"])
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
            mean7=( "NA" if mean7 is None else f"{mean7:.4f}"),
            mean30=( "NA" if mean30 is None else f"{mean30:.4f}"),
            pos7=( "NA" if pos7 is None else f"{pos7:.4f}"),
            neg7=( "NA" if neg7 is None else f"{neg7:.4f}"),
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

        # context 필드는 base Opinion에는 없어서 제거
        return Opinion(
            agent_id=self.agent_id,
            target=target,
            reason=reason,
        )
