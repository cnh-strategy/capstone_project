# core/sentimental_classes/finbert_utils.py
from __future__ import annotations
from typing import List, Tuple, Iterable, Optional, Sequence, Dict, Any, Union
import os
import math
from datetime import datetime, date, timedelta
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.utils_datetime import safe_parse_iso_datetime as _safe_dt
from .utils_datetime import safe_parse_iso_datetime as _safe_dt

# --- 환경변수 기본값 유지 ---
_FINBERT_MODEL_ID = os.getenv("FINBERT_MODEL_ID", "ProsusAI/finbert")
_HF_CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or None

TextLike = Union[str, Dict[str, Any]]  # 뉴스 아이템(dict), 혹은 문자열
ScoreTuple = Tuple[float, float, float, float]  # (p_neg, p_neu, p_pos, score)

# def _parse_iso_datetime(s):
#     return _safe_dt(s)

__all__ = [
    "FinBertScorer",
    "score_news_items",
    "attach_scores_to_items",
    "compute_finbert_features",
]


class FinBertScorer:
    """
    간단 감성 점수기 (FinBERT)
      - 반환: [(p_neg, p_neu, p_pos, score), ...]
      - score = p_pos - p_neg  ∈ [-1, 1]
    """
    def __init__(
        self,
        device: Optional[str] = None,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        model_id = model_id or _FINBERT_MODEL_ID

        # 1) 우선순위: 인자로 들어온 cache_dir > 환경변수 기반 > 프로젝트 기본 폴더
        cache_dir = cache_dir or _HF_CACHE_DIR or str(_DEFAULT_HF_CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[FinBERT] using cache_dir: {cache_dir}")  # 디버그용, 나중에 지워도 됨

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()


# ---------------------------
# 편의 유틸: 뉴스 아이템 스코어링
# ---------------------------

def _extract_text(
    item: TextLike,
    text_fields: Sequence[str] = ("title", "content", "text", "summary")
) -> str:
    """dict/str 혼용 입력에서 텍스트 추출."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    parts: List[str] = []
    for f in text_fields:
        v = item.get(f)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts).strip()


def score_news_items(
    items: Iterable[TextLike],
    scorer: Optional[FinBertScorer] = None,
    batch_size: int = 16,
    max_length: int = 256,
    text_fields: Sequence[str] = ("title", "content", "text", "summary"),
    neutral_on_error: bool = True,
) -> List[ScoreTuple]:
    """
    뉴스 아이템(문자열/딕셔너리 혼용) 리스트를 FinBERT로 스코어링.
    반환: 각 아이템에 대응하는 (p_neg, p_neu, p_pos, score).
    """
    items = list(items or [])
    texts = [_extract_text(x, text_fields=text_fields) for x in items]
    if scorer is None:
        scorer = FinBertScorer()

    try:
        return scorer.score_texts(texts, batch_size=batch_size, max_length=max_length)
    except Exception as e:
        if not neutral_on_error:
            raise
        # 에러 시 전부 중립으로 복원
        n = len(texts)
        return [(0.0, 1.0, 0.0, 0.0) for _ in range(n)]


def attach_scores_to_items(
    items: List[Dict[str, Any]],
    scores: List[ScoreTuple],
    out_keys: Sequence[str] = ("p_neg", "p_neu", "p_pos", "sentiment_score")
) -> List[Dict[str, Any]]:
    """
    기존 뉴스 딕셔너리 리스트에 점수 필드를 붙여 반환.
    """
    assert len(items) == len(scores), "items와 scores 길이가 다릅니다."
    out: List[Dict[str, Any]] = []
    k_neg, k_neu, k_pos, k_score = out_keys
    for it, (p_neg, p_neu, p_pos, score) in zip(items, scores):
        new_it = dict(it)
        new_it[k_neg] = p_neg
        new_it[k_neu] = p_neu
        new_it[k_pos] = p_pos
        new_it[k_score] = score
        out.append(new_it)
    return out


# ---------------------------
# 피처 집계(7d/30d 등)
# ---------------------------

def _parse_iso_datetime(s) -> Optional[datetime]:
    # 문자열이 아니면 바로 None
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # 'Z' → '+00:00' 보정
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _to_date_utc(d: Optional[datetime]) -> Optional[date]:
    return d.date() if d else None


def _safe_mean(a: Sequence[float]) -> float:
    return float(mean(a)) if a else 0.0


def _safe_vol(a: Sequence[float]) -> float:
    if len(a) <= 1:
        return 0.0
    try:
        return float(pstdev(a))
    except Exception:
        return 0.0


def _ratio(a: Sequence[float], cond) -> float:
    if not a:
        return 0.0
    c = sum(1 for x in a if cond(x))
    return c / len(a)


def compute_finbert_features(
    items: List[Dict[str, Any]],
    asof_utc_date: date,
    score_key: str = "sentiment_score",
    date_keys: Sequence[str] = ("date", "published_date"),
) -> Dict[str, Any]:
    """
    FinBERT 점수를 부착한 뉴스 리스트에서 기간 통계 피처 생성.
      - sentiment_summary: mean_7d, mean_30d, pos_ratio_7d, neg_ratio_7d
      - sentiment_volatility: vol_7d, vol_30d
      - news_count: count_1d, count_7d
      - trend_7d: mean_7d - mean_30d
    """
    # 날짜/점수 파싱
    parsed: List[Tuple[date, float]] = []
    for it in items:
        d = None
        for k in date_keys:
            d = _parse_iso_datetime(it.get(k))
            if d:
                break
        dd = _to_date_utc(d)
        if dd is None:
            continue
        s = it.get(score_key)
        if s is None:
            continue
        try:
            s = float(s)
        except Exception:
            continue
        parsed.append((dd, s))

    if not parsed:
        return {
            "sentiment_summary": {"mean_7d": 0.0, "mean_30d": 0.0, "pos_ratio_7d": 0.0, "neg_ratio_7d": 0.0},
            "sentiment_volatility": {"vol_7d": 0.0, "vol_30d": 0.0},
            "news_count": {"count_1d": 0, "count_7d": 0},
            "trend_7d": 0.0,
            "has_news": False,
        }

    d1 = asof_utc_date
    d7 = d1 - timedelta(days=7)
    d30 = d1 - timedelta(days=30)

    s1d = [s for (d, s) in parsed if d == d1]
    s7d = [s for (d, s) in parsed if d7 < d <= d1]
    s30d = [s for (d, s) in parsed if d30 < d <= d1]

    feat = {
        "sentiment_summary": {
            "mean_7d": _safe_mean(s7d),
            "mean_30d": _safe_mean(s30d),
            "pos_ratio_7d": _ratio(s7d, lambda x: x > 0),
            "neg_ratio_7d": _ratio(s7d, lambda x: x < 0),
        },
        "sentiment_volatility": {
            "vol_7d": _safe_vol(s7d),
            "vol_30d": _safe_vol(s30d),
        },
        "news_count": {
            "count_1d": len(s1d),
            "count_7d": len(s7d),
        },
        "trend_7d": _safe_mean(s7d) - _safe_mean(s30d),
        "has_news": True,
    }
    return feat
