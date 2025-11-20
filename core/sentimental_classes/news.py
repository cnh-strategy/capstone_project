# core/sentimental_classes/news.py

from __future__ import annotations

from typing import Optional
import pandas as pd


def merge_price_with_news_features(
    df_price: pd.DataFrame,
    df_news: Optional[pd.DataFrame],
    tz: str = "Asia/Seoul",
) -> pd.DataFrame:
    """
    가격 데이터(df_price)와 FinBERT가 계산된 뉴스 데이터(df_news)를 합쳐서 다음 컬럼을 생성해 붙여준다.

    - news_count_1d : 해당 일자의 뉴스 개수
    - news_count_7d : 최근 7일(포함) 누적 뉴스 개수
    - sentiment_mean_1d : 해당 일자의 FinBERT 감성 점수 평균
    - sentiment_mean_7d : 최근 7일(포함) 감성 평균
    - sentiment_vol_7d : 최근 7일(포함) 감성 점수 변동성(표준편차)

    """
    if df_news is None or len(df_news) == 0:
        raise ValueError(
            "[merge_price_with_news_features] df_news is empty. "
            "뉴스가 전혀 없는 상태에서는 피처를 만들지 않습니다."
        )

    # 1) 가격 데이터 인덱스를 KST 기준 날짜로 정리
    if not isinstance(df_price.index, pd.DatetimeIndex):
        raise ValueError("[merge_price_with_news_features] df_price.index must be DatetimeIndex")

    price_index = df_price.index

    # 타임존 정규화 (tz-aware / naive 모두 대응)
    if price_index.tz is None:
        price_index_kst = price_index.tz_localize(tz)
    else:
        price_index_kst = price_index.tz_convert(tz)

    # 날짜 단위로만 사용 (시·분·초 제거)
    price_dates = price_index_kst.normalize()

    # 2) 뉴스 데이터에서 날짜·감성 정보 추출
    if "sentiment" not in df_news.columns:
        raise ValueError("[merge_price_with_news_features] df_news must contain 'sentiment' column")

    # 날짜 컬럼 후보 찾기
    date_col = None
    for cand in ["date", "published_at", "datetime", "time"]:
        if cand in df_news.columns:
            date_col = cand
            break

    if date_col is None:
        raise ValueError(
            "[merge_price_with_news_features] df_news must have one of "
            "['date', 'published_at', 'datetime', 'time'] columns"
        )

    df_news = df_news.copy()
    dt = pd.to_datetime(df_news[date_col], errors="coerce")

    if dt.isna().all():
        raise ValueError(
            f"[merge_price_with_news_features] cannot parse datetime from df_news['{date_col}']"
        )

    # 타임존 정리
    if dt.dt.tz is None:
        dt_kst = dt.dt.tz_localize(tz)
    else:
        dt_kst = dt.dt.tz_convert(tz)

    df_news["_date_kst"] = dt_kst.dt.normalize()

    # 3) 일 단위 집계 (1d 피처)
    grouped = df_news.groupby("_date_kst")["sentiment"]

    df_daily = pd.DataFrame({
        "news_count_1d": grouped.size(),
        "sentiment_mean_1d": grouped.mean(),
        # 1일 단위 변동성은 거의 의미 없어서 7d 변동성에만 사용할 것.
    })

    # 변동성은 우선 0으로 초기화 (1일 기준으로는 정의 X)
    df_daily["sentiment_vol_1d"] = 0.0

    # 4) 가격 날짜 인덱스 기준으로 재인덱싱
    #    - 뉴스 없는 날은 뉴스 개수 0, 감성 0으로 처리
    # price_dates: DatetimeIndex (KST, normalized) -> 이를 기준으로 align
    df_daily = df_daily.reindex(price_dates)

    # 뉴스 없는 날 처리
    df_daily["news_count_1d"] = df_daily["news_count_1d"].fillna(0).astype(int)
    df_daily["sentiment_mean_1d"] = df_daily["sentiment_mean_1d"].fillna(0.0)
    df_daily["sentiment_vol_1d"] = df_daily["sentiment_vol_1d"].fillna(0.0)

    # 5) 7일 롤링 피처 계산
    # 최근 7일 누적 뉴스 개수
    df_daily["news_count_7d"] = (
        df_daily["news_count_1d"]
        .rolling(window=7, min_periods=1)
        .sum()
    )

    # 최근 7일 평균 감성 점수
    df_daily["sentiment_mean_7d"] = (
        df_daily["sentiment_mean_1d"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    # 최근 7일 감성 변동성 (표준편차)
    df_daily["sentiment_vol_7d"] = (
        df_daily["sentiment_mean_1d"]
        .rolling(window=7, min_periods=2)  # 최소 2일 이상 있어야 std 의미 있음
        .std(ddof=0)
        .fillna(0.0)
    )

    # 6) 가격 DF와 합치기 (인덱스: 원래 df_price.index 유지)
    df_daily.index = price_index  # 원래 인덱스(시간 포함)와 align

    df_merged = df_price.copy()
    for col in [
        "news_count_1d",
        "news_count_7d",
        "sentiment_mean_1d",
        "sentiment_mean_7d",
        "sentiment_vol_7d",
    ]:
        df_merged[col] = df_daily[col]

    return df_merged
