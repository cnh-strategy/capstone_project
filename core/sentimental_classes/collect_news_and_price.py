import os
import time
import requests
import csv
from datetime import datetime
import yfinance as yf
import pandas as pd
import json


# ================================
# 🔧 EODHD API 설정
# ================================
API_KEY = '68e3a8c46e9a65.00465987'
BASE_URL_EODHD = 'https://eodhd.com/api/news'

STATUS_FILE = 'collection_status.json'

SYMBOLS = ['NVDA', 'MSFT', 'AAPL']


# ================================
# 상태 파일 (지금은 필요 없음)
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
    """
    EODHD API로 뉴스 + sentiment_score 수집
    """
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
            'extended': 1,   # 감성 점수 포함
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


# ================================
# 메인 실행
# ================================
def main():

    NEWS_FILE = "news_data.csv"
    STOCK_FILE = "stock_data.csv"

    # 5년 데이터 수집 기간
    from_date = '2020-01-01'
    to_date_news = '2024-12-31'
    to_date_stock = '2025-01-01'

    print(f"\n📅 뉴스/주가 데이터 수집 기간: {from_date} ~ {to_date_news}")

    # 기존 파일 삭제 (초기화)
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

        if last_offset != -1:  # API 오류 등으로 중단
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


if __name__ == "__main__":
    main()
