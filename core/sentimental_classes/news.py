# core/sentimental_classes/news.py

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Any, List
import os
import json
from typing import Tuple
import pandas as pd

from core.sentimental_classes.finbert_utils import FinBertScorer
from core.sentimental_classes.eodhd_client import fetch_news_from_eodhd


def build_finbert_news_features(
    ticker: str,
    asof_kst: date,
    base_dir: str = os.path.join("data", "raw", "news"),
) -> Dict[str, Any]:
    """
    특정 종목에 대해 최근 7일간 뉴스 감성 피처를 계산한다.

    반환 예시:
    {
        "ticker": "NVDA",
        "asof": "2025-11-19",
        "news_count_7d": 23,
        "sentiment_mean_7d": 0.12,
        "sentiment_vol_7d": 0.35,
        "sentiment_trend_7d": -0.05,
    }
    """

    scorer = FinBertScorer()

    end_date = asof_kst
    start_date = asof_kst - timedelta(days=7)

    os.makedirs(base_dir, exist_ok=True)
    cache_name = f"{ticker}_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}.json"
    cache_path = os.path.join(base_dir, cache_name)

    # 1) 캐시가 있으면 먼저 사용
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_news: List[Dict[str, Any]] = json.load(f)
    else:
        # 2) 없으면 EODHD에서 뉴스 수집
        raw_news = fetch_news_from_eodhd(ticker, start_date, end_date)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_news, f, ensure_ascii=False)

    # 뉴스가 아예 없으면 0 반환 (이 경우만 0 허용)
    if not raw_news:
        return {
            "ticker": ticker,
            "asof": end_date.isoformat(),
            "news_count_7d": 0,
            "sentiment_mean_7d": 0.0,
            "sentiment_vol_7d": 0.0,
            "sentiment_trend_7d": 0.0,
        }

    rows = []
    for item in raw_news:
        title = item.get("title") or ""
        text = item.get("content") or item.get("description") or ""

        scored = scorer.score(title + " " + text)

        # 너가 방금 출력한 형태: [{'pos':..., 'neg':..., 'neu':..., 'score': ...}]
        if isinstance(scored, list) and scored:
            s = float(scored[0].get("score", 0.0))
        elif isinstance(scored, dict):
            s = float(scored.get("score", 0.0))
        else:
            continue

        # 날짜 파싱 (형식이 다양할 수 있어서 앞 10자리만 쓰는 방식)
        dt_str = (
            item.get("date")
            or item.get("published")
            or item.get("time")
            or item.get("datetime")
        )
        if dt_str:
            d = pd.to_datetime(str(dt_str)[:10]).date()
        else:
            d = end_date

        rows.append({"date": d, "score": s})

    if not rows:
        # 뉴스 리스트는 있는데 점수 계산에 실패한 경우 -> 오류로 취급
        raise RuntimeError(
            "[FinBERT] 뉴스는 있으나 감성 점수를 계산하지 못했습니다. "
            "finbert_utils.FinBertScorer.score 반환 형식을 확인해 주세요."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date")

    news_count = int(df.shape[0])
    sentiment_mean = float(df["score"].mean())
    sentiment_vol = float(df["score"].std(ddof=0) or 0.0)

    # 간단한 추세: 앞 절반 평균 vs 뒤 절반 평균 차이
    mid = max(1, len(df) // 2)
    early_mean = float(df.iloc[:mid]["score"].mean())
    late_mean = float(df.iloc[mid:]["score"].mean())
    sentiment_trend = late_mean - early_mean

    feats = {
        "ticker": ticker,
        "asof": end_date.isoformat(),
        "news_count_7d": news_count,
        "sentiment_mean_7d": sentiment_mean,
        "sentiment_vol_7d": sentiment_vol,
        "sentiment_trend_7d": sentiment_trend,
    }

    print(
        f"{ticker}의 최근 7일 감성 평균은 {sentiment_mean:.3f}이며 "
        f"뉴스 개수(7d)는 {news_count}건입니다. "
        f"감성 변동성(vol_7d)={sentiment_vol:.3f}, "
        f"감성 추세(trend_7d)={sentiment_trend:.3f}입니다."
    )

    return feats

def merge_price_with_news_features(
    df_price: pd.DataFrame,
    ticker: str,
    asof_kst: date,
    base_dir: str = os.path.join("data", "raw", "news"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    가격 데이터프레임(df_price)에 FinBERT 기반 뉴스 피처를 붙인다.
    - df_price: OHLCV 등 가격 시계열 (index: Datetime, columns: open/high/...)
    - return: (확장된 df_price, news_feats dict)
    """

    news_feats = build_finbert_news_features(
        ticker=ticker,
        asof_kst=asof_kst,
        base_dir=base_dir,
    )

    # df_price에 컬럼으로 브로드캐스트 (최근 7일 요약값이라 모든 행에 동일하게 넣어도 됨)
    df_price = df_price.copy()
    df_price["news_count_7d"] = news_feats["news_count_7d"]
    df_price["sentiment_mean_7d"] = news_feats["sentiment_mean_7d"]
    df_price["sentiment_vol_7d"] = news_feats["sentiment_vol_7d"]
    df_price["sentiment_trend_7d"] = news_feats["sentiment_trend_7d"]

    return df_price, news_feats
