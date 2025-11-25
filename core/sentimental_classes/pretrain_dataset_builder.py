import os
import pandas as pd
from core.sentimental_classes.news_history_builder import fetch_history_news
from core.sentimental_classes.finbert_scorer import FinBertScorer

def build_pretrain_dataset(ticker):
    print(f"[SentimentalAgent] Building pretrain dataset with news for {ticker}...")

    # 🔥 pretrain 디렉토리 자동 생성
    save_dir = "data/pretrain"
    os.makedirs(save_dir, exist_ok=True)

    # 1) 뉴스 수집
    start = "2020-01-01"
    end = "2025-01-01"
    news_list = fetch_history_news(ticker, start, end)

    # list → DataFrame
    if isinstance(news_list, list):
        df_news = pd.DataFrame(news_list)
    else:
        raise RuntimeError("fetch_history_news did not return list of dict")

    # content 없는 경우 방어
    if "content" not in df_news.columns:
        df_news["content"] = ""

    # 2) FinBERT 스코어
    scorer = FinBertScorer()
    df_news = scorer.score(df_news)

    # 3) 저장
    save_path = f"{save_dir}/{ticker}_news_pretrain.csv"
    df_news.to_csv(save_path, index=False)

    print(f"[SentimentalAgent] Pretrain news saved: {save_path}")
    return df_news
