# core/sentimental_classes/collect_news_and_price.py

import os
import time
import requests
import csv
import yfinance as yf
import pandas as pd
import json
import ast

from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("EODHD_API_KEY")
BASE_URL_EODHD = 'https://eodhd.com/api/news'

STATUS_FILE = 'collection_status.json'
SYMBOLS = ['NVDA', 'MSFT', 'AAPL']


# ================================
# 상태 파일
# ================================
def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {'completed_symbols': []}


def save_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=4)


# ================================
# 뉴스 수집 함수
# ================================
def collect_news_data_eodhd(symbol, from_date, to_date):
    all_news = []
    offset = 0
    limit = 1000

    while True:
        params = {
            's': symbol,
            'from': from_date,
            'to': to_date,
            'api_token': API_KEY,
            'limit': limit,
            'offset': offset,
            'extended': 1,
        }

        try:
            response = requests.get(BASE_URL_EODHD, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[{symbol}] 네트워크 오류: {e}")
            return all_news, offset

        if response.status_code == 200:
            news_list = response.json()
            if not news_list:
                print(f"[{symbol}] 더 이상 뉴스 데이터 없음.")
                break

            for news in news_list:
                data = {
                    'date': news.get('date', ''),
                    'title': news.get('title', ''),
                    'summary': news.get('content', ''),
                    'related': news.get('symbols', symbol),
                    'ticker': symbol,
                    'sentiment_score': news.get('sentiment', '')
                }
                all_news.append(data)

            if len(news_list) < limit:
                break
            else:
                offset += limit
                time.sleep(1)

        else:
            print(f"[{symbol}] API 오류 {response.status_code}: {response.text}")
            return all_news, offset

    return all_news, -1


# ================================
# CSV 저장 함수
# ================================
def save_news_to_csv(news_data, filename, mode='a'):
    fieldnames = ['date', 'title', 'summary', 'related', 'ticker', 'sentiment_score']

    file_exists = os.path.exists(filename) and mode == 'a'

    with open(filename, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if mode == 'w' or not file_exists:
            writer.writeheader()

        for record in news_data:
            writer.writerow(record)

    print(f"📁 {filename} → {len(news_data)}개 뉴스 저장 완료")


def _parse_sentiment_score(raw):
    """
    EODHD sentiment_score 컬럼 파싱:
    - "{'polarity': ..., 'neg': ..., 'neu': ..., 'pos': ...}" 형태
    - 있으면 polarity를 점수로 사용
    """
    if raw is None:
        return None

    # 이미 숫자면 그대로
    if isinstance(raw, (int, float)):
        return float(raw)

    s = str(raw).strip()
    if not s:
        return None

    # 1) 딕셔너리 문자열인 경우 (single quote라 json.loads 안 먹힘)
    try:
        data = ast.literal_eval(s)  # {'polarity': ..., 'neg': ..., ...}
        if isinstance(data, dict):
            if "polarity" in data:
                return float(data["polarity"])
            # 혹시 polarity 없고 pos/neg만 있으면 pos-neg 사용
            if "pos" in data and "neg" in data:
                return float(data["pos"]) - float(data["neg"])
    except Exception:
        pass

    # 2) 그냥 숫자 문자열일 수도 있으니 마지막으로 float 캐스팅 시도
    try:
        return float(s)
    except Exception:
        return None


def build_news_features_from_eodhd(
    news_csv: str = "news_data.csv",
    out_dir: str = os.path.join("data", "features", "news"),
):
    """
    EODHD에서 수집한 news_data.csv(기사 단위)를
    news.py에서 기대하는 형식의
    {TICKER}_news_features.csv 파일들로 변환.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(news_csv)

    # ['date', 'title', 'summary', 'related', 'ticker', 'sentiment_score']
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # 🔹 여기서 sentiment_score를 우리가 정의한 파서로 숫자로 변환
    df["sentiment_score_num"] = df["sentiment_score"].apply(_parse_sentiment_score)

    # 점수가 전혀 없는 행은 버려도 됨
    df = df.dropna(subset=["sentiment_score_num"])

    tickers = sorted(df["ticker"].dropna().unique())
    print("🎯 뉴스 피처 생성 대상 티커:", tickers)

    for tkr in tickers:
        sub = df[df["ticker"] == tkr].copy()
        if sub.empty:
            continue

        # 1) 날짜별 집계: 하루 기사 수, 감성 합/평균
        daily = (
            sub.groupby("date")["sentiment_score_num"]
            .agg(
                news_count_1d="count",
                sentiment_sum_1d="sum",
                sentiment_mean_1d="mean",
            )
            .reset_index()
        )

        # news.py는 'Date' 컬럼명을 사용하므로 통일
        daily = daily.sort_values("date")
        daily = daily.rename(columns={"date": "Date"})

        # 7일 롤링은 news.py에서 다시 계산하므로 틀만 맞춰둠
        daily["news_count_7d"] = 0
        daily["sentiment_sum_7d"] = 0.0
        daily["sentiment_mean_7d"] = 0.0

        out_path = os.path.join(out_dir, f"{tkr.upper()}_news_features.csv")
        daily.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ {tkr} → {out_path} 저장 완료 (rows={len(daily)})")


# ================================
# 메인 실행: 뉴스/주가 수집 + 피처 생성
# ================================
def main():
    NEWS_FILE = "news_data.csv"
    STOCK_FILE = "stock_data.csv"

    from_date = '2020-01-01'
    to_date_news = '2024-12-31'
    to_date_stock = '2025-01-01'

    print(f"\n📅 뉴스/주가 데이터 수집 기간: {from_date} ~ {to_date_news}")

    # 기존 파일 삭제
    if os.path.exists(NEWS_FILE):
        os.remove(NEWS_FILE)
        print(f"🗑 기존 {NEWS_FILE} 삭제")

    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
        print(f"🗑 기존 {STATUS_FILE} 삭제")

    # --- 뉴스 수집 ---
    print("\n=== 📰 뉴스 수집 시작 ===")
    is_first_symbol = True
    news_ok = True

    for symbol in SYMBOLS:
        print(f"\n[{symbol}] 뉴스 수집 중...")

        collected_news, last_offset = collect_news_data_eodhd(
            symbol, from_date, to_date_news
        )

        if collected_news:
            save_mode = 'w' if is_first_symbol else 'a'
            save_news_to_csv(collected_news, NEWS_FILE, mode=save_mode)
            is_first_symbol = False

        if last_offset != -1:
            news_ok = False
            print(f"[{symbol}] ⚠ 수집 중단됨 (offset={last_offset})")
            break

        print(f"[{symbol}] 🟢 뉴스 수집 완료.")

    # --- 주가 수집 ---
    print("\n=== 💹 주가 데이터 수집 시작 ===")

    if news_ok:
        all_stock = []

        for symbol in SYMBOLS:
            print(f"{symbol} 가격 데이터(yfinance) 다운로드 중...")

            df = yf.download(symbol, start=from_date, end=to_date_stock)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df = df.reset_index()
            df["Symbol"] = symbol
            df = df[["Symbol", "Date", "Open", "Close"]]

            all_stock.append(df)

        stock_df = pd.concat(all_stock, ignore_index=True)
        stock_df.to_csv(STOCK_FILE, index=False, encoding="utf-8")

        print(f"📁 {STOCK_FILE} 저장 완료!")
    else:
        print("⚠ 뉴스 수집 실패 → 가격 데이터 수집 생략")

    # --- 뉴스 피처 생성 ---
    print("\n=== 🧮 뉴스 피처 생성 ===")
    build_news_features_from_eodhd(NEWS_FILE)


if __name__ == "__main__":
    main()
