# core/features/sentimental.py
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, requests, os, json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")
BASE_URL_EODHD = "https://eodhd.com/api/news"

def fetch_sentimental_features(ticker: str, window_size: int = 14):
    """
    최근 N일간 뉴스+주가 데이터를 기반으로 감성 피처 생성
    """
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=window_size * 2)
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()
    df["ret"] = df["Close"].pct_change()
    df = df.dropna()

    # 뉴스 수집
    params = {"s": ticker, "from": start.strftime("%Y-%m-%d"),
              "to": end.strftime("%Y-%m-%d"), "api_token": API_KEY}
    news = requests.get(BASE_URL_EODHD, params=params).json()
    news_df = pd.DataFrame(news)
    news_df["text"] = news_df["title"] + " " + news_df["content"]

    # FinBERT 감성 분석
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    model.eval()

    texts = news_df["text"].tolist()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()

    news_df["prob_positive"] = probs[:, 0]
    news_df["prob_negative"] = probs[:, 1]
    news_df["prob_neutral"] = probs[:, 2]

    daily = news_df.groupby(news_df["date"].str[:10]).mean(numeric_only=True).reset_index()
    merged = pd.merge(df, daily, left_on=df["Date"].dt.strftime("%Y-%m-%d"), right_on="date", how="left")
    merged = merged.fillna(0)

    features = merged[["prob_positive", "prob_negative", "prob_neutral", "ret", "Close"]].tail(window_size)
    sequence = features.values
    key_topics = news_df["title"].tail(3).tolist()
    return sequence, key_topics, float(df["Close"].iloc[-1])
