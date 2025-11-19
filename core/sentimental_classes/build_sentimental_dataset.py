# core/sentimental_classes/build_sentimental_dataset.py

import os
import pandas as pd
import yfinance as yf
from core.sentimental_classes.news import merge_price_with_news_features

OUT_DIR = os.path.join("data", "datasets")
os.makedirs(OUT_DIR, exist_ok=True)


def build_dataset_for_ticker(ticker: str):
    # 5년치 기간: collect_news_and_price와 맞춰줌
    start = "2020-01-01"
    end = "2025-01-01"

    print(f"\n=== [{ticker}] 가격 + 뉴스 데이터셋 생성 ===")
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    # Date, Open, High, Low, Close, Volume 사용
    df_price = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # 🔹 뉴스 피처 병합
    df_merged = merge_price_with_news_features(
        df_price,
        ticker=ticker,
        window=7,
        show_tail=True,  # tail 디버깅용
    )

    df_merged = df_merged.sort_values("Date").copy()

    # 🔹 주가 피처 3개
    df_merged["return_1d"] = df_merged["Close"].pct_change()
    df_merged["hl_range"] = (df_merged["High"] - df_merged["Low"]) / df_merged["Close"]

    # 🔹 뉴스 변동성: sentiment_mean_1d 7일 롤링 std
    df_merged["sentiment_vol_7d"] = (
        df_merged["sentiment_mean_1d"]
        .rolling(window=7, min_periods=1)
        .std()
        .fillna(0.0)
    )

    feature_cols = [
        "return_1d",
        "hl_range",
        "Volume",
        "news_count_1d",
        "news_count_7d",
        "sentiment_mean_1d",
        "sentiment_mean_7d",
        "sentiment_vol_7d",
    ]

    final = df_merged[["Date"] + feature_cols].dropna().reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, f"{ticker.upper()}_sentimental_dataset.csv")
    final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ {ticker} 학습용 데이터 저장 완료: {out_path} (rows={len(final)})")


if __name__ == "__main__":
    for t in ["NVDA", "MSFT", "AAPL"]:
        build_dataset_for_ticker(t)
