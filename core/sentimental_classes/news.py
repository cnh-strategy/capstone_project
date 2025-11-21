# core/sentimental_classes/news.py

import pandas as pd
import numpy as np

def merge_price_with_news_features(
    df_price: pd.DataFrame,
    ticker: str,
    asof_kst,
    base_dir: str = "data/raw/news",
):
    """
    뉴스 데이터를 찾을 수 없는 경우에도 코드가 깨지지 않도록
    기본 7일 집계 피처를 0으로 채워서 df_price에 붙여주는 간단한 버전

    실제 뉴스 수집/FinBERT 스코어링이 준비되면 여기만 교체하면 됨.
    """

    df = df_price.copy()

    # 기본 7일 감성/뉴스 카운트 피처(아직 뉴스 안 쓰는 버전)
    df["news_count_7d"] = 0
    df["sentiment_mean_7d"] = 0.0
    df["sentiment_vol_7d"] = 0.0

    return df
