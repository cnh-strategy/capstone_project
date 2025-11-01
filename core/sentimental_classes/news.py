import os
from typing import List
import pandas as pd


def _ensure_date_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """Date 컬럼을 date 타입으로 통일."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def _fill_missing_cols(df: pd.DataFrame, cols: List[str], fill_value=0.0) -> pd.DataFrame:
    """없는 컬럼은 생성하고, 있는 컬럼은 결측을 채운다."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
        else:
            df[c] = df[c].fillna(fill_value)
    return df


def merge_price_with_news_features(
    df_price: pd.DataFrame,
    ticker: str,
    *,
    window: int = 7,
    show_tail: bool = False,
    news_features_path: str | None = None,
) -> pd.DataFrame:
    """
    가격 데이터(df_price)에 뉴스 피처를 병합하고, 병합 후 '영업일 기준'으로 7일 롤링 지표를 재계산한다.

    Parameters
    ----------
    df_price : pd.DataFrame
        'Date' 컬럼을 가진 일자 단위 가격 데이터프레임.
    ticker : str
        예) 'TSLA'
    window : int, default 7
        7영업일 롤링 윈도우 길이.
    show_tail : bool, default False
        병합/재계산 후 tail(10행)을 콘솔에 출력(디버깅용).
    news_features_path : str | None
        수동 경로 지정 (기본: data/features/news/{TICKER}_news_features.csv)

    Returns
    -------
    pd.DataFrame
        뉴스 피처가 결합된 가격 데이터프레임.
    """
    # 0) 파일 경로 확인
    path = news_features_path or os.path.join(
        "data", "features", "news", f"{ticker.upper()}_news_features.csv"
    )

    base_cols = ["news_count_1d", "news_count_7d", "sentiment_mean_1d", "sentiment_mean_7d"]

    # 1) 파일이 없으면 0으로 채운 컬럼 추가 후 반환
    if not os.path.exists(path):
        out = _ensure_date_col(df_price, "Date")
        out = _fill_missing_cols(out, base_cols, fill_value=0.0)
        out["news_count_1d"] = out["news_count_1d"].astype("int64")
        out["news_count_7d"] = out["news_count_7d"].astype("int64")

        if show_tail:
            print("[after merge] (no news file) tail")
            print(out.tail(10)[["Date"] + base_cols])
            print()
        return out

    # 2) 뉴스 피처 로드
    news = pd.read_csv(path, encoding="utf-8-sig")
    news = _ensure_date_col(news, "Date")

    expected_cols = [
        "news_count_1d",
        "news_count_7d",
        "sentiment_sum_1d",
        "sentiment_sum_7d",
        "sentiment_mean_1d",
        "sentiment_mean_7d",
    ]
    for c in expected_cols:
        if c not in news.columns:
            news[c] = 0.0

    # 3) 가격 데이터와 병합
    out = _ensure_date_col(df_price, "Date")
    out = out.merge(news[["Date"] + expected_cols], on="Date", how="left")

    # 4) 타입/결측 정리
    for c in ["news_count_1d", "news_count_7d", "sentiment_sum_1d", "sentiment_sum_7d"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    for c in ["sentiment_mean_1d", "sentiment_mean_7d"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # sum이 0인데 mean과 count가 있으면 sum ≈ mean * count
    if (out["sentiment_sum_1d"] == 0).all() and (out["sentiment_mean_1d"].abs().sum() > 0) and (out["news_count_1d"].sum() > 0):
        out["sentiment_sum_1d"] = out["sentiment_mean_1d"] * out["news_count_1d"]

    # 5) 7영업일 롤링 재계산
    out = out.sort_values("Date").copy()
    cnt7 = out["news_count_1d"].rolling(window=window, min_periods=1).sum()
    sum7 = out["sentiment_sum_1d"].rolling(window=window, min_periods=1).sum()

    out["news_count_7d"] = cnt7.astype("int64")
    out["sentiment_mean_7d"] = (sum7 / cnt7.replace(0, pd.NA)).fillna(0.0)

    # 6) 최종 결측 방지
    out = _fill_missing_cols(out, base_cols, fill_value=0.0)

    # 7) 디버그 출력
    if show_tail:
        try:
            print("[after merge] tail")
            print(out.tail(10)[["Date"] + base_cols])
            print()
        except Exception:
            pass

    return out
