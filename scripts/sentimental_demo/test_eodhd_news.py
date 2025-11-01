# scripts/sentimental_demo/test_eodhd_news.py
# -*- coding: utf-8 -*-
"""
EODHD 뉴스 수집 + FinBERT 감성 스코어링 -> CSV 저장
사용법 (PowerShell 예시):
  $env:PYTHONPATH='.'
  python ./scripts/sentimental_demo/test_eodhd_news.py --ticker TSLA --days 30

출력:
  data/news/<TICKER>_news.csv
  - columns: date, title, summary(있으면), ..., finbert_p_neg/neu/pos, sentiment
  - sentiment = p_pos - p_neg  (연속 점수, [-1, 1])
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

sys.path.append(os.path.abspath("."))  # 루트 경로 추가
import pandas as pd

# ✅ main 브랜치 경로에 맞게 수정됨
from core.finbert_utils import FinBertScorer
from core.eodhd_client import EODHDNewsClient

# EODHDNewsClient 유무 확인
_HAS_CLIENT = True


def fetch_news_eodhd(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str,
    limit: int = 200
) -> List[Dict[str, Any]]:
    """
    EODHD에서 뉴스 목록을 가져와 dict 리스트로 반환.
    """
    if _HAS_CLIENT:
        client = EODHDNewsClient(api_key=api_key)
        try:
            items = client.fetch_company_news(
                ticker=ticker,
                from_date=from_date,
                to_date=to_date,
                limit=limit,
            )
            return list(items or [])
        except Exception as e:
            print(f"[warn] client fetch failed: {e}")

    # ---- HTTP Fallback ----
    import requests
    url = (
        f"https://eodhd.com/api/news"
        f"?s={ticker}&from={from_date}&to={to_date}&api_token={api_key}&fmt=json"
    )
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    data = r.json()
    return list(data or [])


def normalize_date_column(df: pd.DataFrame) -> pd.Series:
    """날짜 컬럼 통일: 'date' > 'published' > 'time' > 'pubDate'"""
    for c in ["date", "published", "time", "pubDate"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            if s.notna().any():
                return s
    return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce", utc=True)


def build_text_for_sentiment(df: pd.DataFrame) -> List[str]:
    """FinBERT 입력용 텍스트(title + summary/description/content/snippet 결합)"""
    use_cols = ["title", "summary", "description", "content", "snippet"]
    cols = [c for c in use_cols if c in df.columns]
    if not cols:
        return ["" for _ in range(len(df))]
    return (
        df[cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="티커 (예: TSLA)")
    parser.add_argument("--days", type=int, default=30, help="과거 며칠 전부터 수집할지")
    parser.add_argument("--max", type=int, default=200, help="최대 뉴스 개수 (API limit)")
    parser.add_argument("--outdir", type=str, default="data/news", help="CSV 저장 경로")
    args = parser.parse_args()

    api_key = os.getenv("EODHD_API_KEY", "")
    print(f"[check] EODHD_API_KEY loaded? {'YES' if api_key else 'NO'}")
    if not api_key:
        print("[error] 환경변수 EODHD_API_KEY가 없습니다.")
        sys.exit(1)

    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=int(args.days))

    print(f"[info] fetching news: {args.ticker} from {from_date} to {to_date}")
    items = fetch_news_eodhd(
        ticker=args.ticker,
        from_date=str(from_date),
        to_date=str(to_date),
        api_key=api_key,
        limit=int(args.max),
    )

    print(f"[result] fetched {len(items)} items")
    if not items:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{args.ticker}_news.csv"
        pd.DataFrame([], columns=["date", "title", "sentiment"]).to_csv(outpath, index=False, encoding="utf-8-sig")
        print(f"[save] CSV -> {outpath} (empty)")
        return

    raw = pd.DataFrame(items)
    raw["date"] = normalize_date_column(raw)
    raw = raw.sort_values("date").reset_index(drop=True)

    # FinBERT 감성 분석
    texts = build_text_for_sentiment(raw)
    scorer = FinBertScorer()
    scores = scorer.score_texts(texts, batch_size=16)

    p_neg, p_neu, p_pos, cont = [], [], [], []
    for (a, b, c, d) in scores:
        p_neg.append(a); p_neu.append(b); p_pos.append(c); cont.append(d)
    raw["finbert_p_neg"] = p_neg
    raw["finbert_p_neu"] = p_neu
    raw["finbert_p_pos"] = p_pos
    raw["sentiment"] = cont  # = p_pos - p_neg

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{args.ticker}_news.csv"
    raw.to_csv(outpath, index=False, encoding="utf-8-sig")
    print(f"[save] CSV -> {outpath}")

    try:
        cols_show = [c for c in ["date", "title", "finbert_p_neg", "finbert_p_pos", "sentiment"] if c in raw.columns]
        print("- head -")
        print(raw.head(3)[cols_show])
        print("- tail -")
        print(raw.tail(3)[cols_show])
    except Exception:
        pass


if __name__ == "__main__":
    main()
