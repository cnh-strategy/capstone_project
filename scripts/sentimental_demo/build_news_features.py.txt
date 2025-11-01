# scripts/sentimental_demo/build_news_features.py
# ---------------------------------------------------------------
# Build daily news features from raw news CSV.
# - Timezone-aware: aggregate by Asia/Seoul calendar days
# - Outputs:
#     Date, news_count_1d, news_count_7d,
#     sentiment_sum_1d, sentiment_sum_7d,
#     sentiment_mean_1d, sentiment_mean_7d
# ---------------------------------------------------------------

import argparse
import os
from pathlib import Path

import pandas as pd


def load_news_csv(ticker: str, path: str) -> pd.DataFrame:
    """
    Read raw news CSV and prepare KST-based calendar date column.
    Expected columns: ["date", "title", (optional) "sentiment"]
    - "date" must be ISO8601 timestamps (UTC).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[error] news csv not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    # tolerate missing columns
    if "date" not in df.columns:
        df["date"] = pd.NaT
    if "title" not in df.columns:
        df["title"] = ""

    # Make tz-aware time columns: UTC -> Asia/Seoul
    df["ts_utc"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["ts_kst"] = df["ts_utc"].dt.tz_convert("Asia/Seoul")
    df["DateKST"] = df["ts_kst"].dt.date

    # Provide sentiment=0.0 if absent
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    else:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0)

    return df


def build_daily_features(df_news: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Build daily (KST calendar) features:
      - news_count_1d, sentiment_sum_1d, sentiment_mean_1d
      - 7d rolling: news_count_7d, sentiment_sum_7d, sentiment_mean_7d
    """
    if df_news is None or df_news.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "news_count_1d",
                "news_count_7d",
                "sentiment_sum_1d",
                "sentiment_sum_7d",
                "sentiment_mean_1d",
                "sentiment_mean_7d",
            ]
        )

    day_col = "DateKST"

    if start is None:
        start = str(df_news[day_col].min())
    if end is None:
        end = str(df_news[day_col].max())

    idx = pd.date_range(start=start, end=end, freq="D")
    base = pd.DataFrame({"Date": idx.date})

    g = (
        df_news.groupby(day_col)
        .agg(
            news_count_1d=("title", "count"),
            sentiment_sum_1d=("sentiment", "sum"),
            sentiment_mean_1d=("sentiment", "mean"),
        )
        .reset_index()
        .rename(columns={day_col: "Date"})
    )

    out = base.merge(g, on="Date", how="left")

    out["news_count_1d"] = pd.to_numeric(out["news_count_1d"], errors="coerce").fillna(0).astype("int64")
    out["sentiment_sum_1d"] = pd.to_numeric(out["sentiment_sum_1d"], errors="coerce").fillna(0.0).astype("float64")
    out["sentiment_mean_1d"] = pd.to_numeric(out["sentiment_mean_1d"], errors="coerce").fillna(0.0).astype("float64")

    out = out.sort_values("Date")
    cnt7 = (
        pd.to_numeric(out["news_count_1d"], errors="coerce").fillna(0)
        .rolling(window=7, min_periods=1)
        .sum()
        .astype("float64")
    )
    sum7 = (
        pd.to_numeric(out["sentiment_sum_1d"], errors="coerce").fillna(0.0)
        .rolling(window=7, min_periods=1)
        .sum()
        .astype("float64")
    )

    out["news_count_7d"] = cnt7
    out["sentiment_sum_7d"] = sum7
    out["sentiment_mean_7d"] = (sum7 / cnt7.replace(0, pd.NA)).fillna(0.0).astype("float64")

    return out


def parse_args():
    p = argparse.ArgumentParser(description="Build KST-aggregated daily news features")
    p.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., TSLA)")
    p.add_argument(
        "--news-csv",
        type=str,
        default=None,
        help="Path to raw news CSV (default: data/news/{TICKER}_news.csv)",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output path (default: data/features/news/{TICKER}_news_features.csv)",
    )
    p.add_argument("--show-tail", action="store_true", help="Print tail(10) of the generated features")
    return p.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    in_csv = args.news_csv or os.path.join("data", "news", f"{ticker}_news.csv")
    out_csv = args.out_csv or os.path.join("data", "features", "news", f"{ticker}_news_features.csv")

    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)

    df_raw = load_news_csv(ticker, in_csv)
    feats = build_daily_features(df_raw)

    feats.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[save] {out_csv} ({len(feats)} rows)")

    if args.show_tail:
        try:
            tmp = feats.copy()
            tmp["Date"] = pd.to_datetime(tmp["Date"]).dt.date
            print(
                tmp.tail(10)[
                    [
                        "Date",
                        "news_count_1d",
                        "news_count_7d",
                        "sentiment_sum_1d",
                        "sentiment_sum_7d",
                        "sentiment_mean_1d",
                        "sentiment_mean_7d",
                    ]
                ]
            )
        except Exception as e:
            print(f"[warn] show-tail failed: {e}")


if __name__ == "__main__":
    main()
