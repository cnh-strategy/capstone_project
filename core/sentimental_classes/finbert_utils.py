# core/sentimental_classes/finbert_utils.py
from pathlib import Path
import json
from datetime import date

ROOT = Path(__file__).resolve().parents[2]  # capstone_project 루트 기준으로 조정

def _normalize_symbol(ticker: str) -> str:
    """EODHD 심볼 형식 통일 (NVDA -> NVDA.US)"""
    if ticker.endswith(".US"):
        return ticker
    return f"{ticker}.US"

def get_news_cache_path(ticker: str, start: date, end: date) -> Path:
    """뉴스 캐시 파일 경로를 한 곳에서만 정의"""
    news_dir = ROOT / "data" / "raw" / "news"
    news_dir.mkdir(parents=True, exist_ok=True)

    symbol = _normalize_symbol(ticker)
    filename = f"{symbol}_{start:%Y-%%m-%d}_{end:%Y-%m-%d}.json"
    return news_dir / filename

def load_or_fetch_news(ticker: str, start: date, end: date, api_key: str):
    cache_path = get_news_cache_path(ticker, start, end)
    print(f"[FinBERT] 캐시 탐색: {cache_path} (exists={cache_path.exists()})")

    # 1) 캐시 있으면 그대로 사용
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # 2) 없으면 폴백: 같은 심볼의 최신 캐시라도 있으면 사용
    news_dir = cache_path.parent
    symbol = _normalize_symbol(ticker)
    candidates = sorted(news_dir.glob(f"{symbol}_*.json"))
    if candidates:
        latest = candidates[-1]
        print(f"[FinBERT] 기존 다른 기간 캐시 사용: {latest.name}")
        with latest.open("r", encoding="utf-8") as f:
            return json.load(f)

    # 3) 그래도 없으면 EODHD 호출 후 저장
    print(f"[FinBERT] 뉴스 캐시 없음: {cache_path}")
    data = fetch_news_from_eodhd(symbol, start, end, api_key)  # 기존 함수 그대로 사용

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[FinBERT] 뉴스 캐시 저장: {cache_path}")

    return data
