import os
import time
import requests
import csv
from datetime import datetime, timedelta
from collections import Counter
import yfinance as yf
import pandas as pd

API_KEY = 'd30p739r01qnu2qv1tmgd30p739r01qnu2qv1tn0'
BASE_URL = 'https://finnhub.io/api/v1/company-news'

def safe_convert_timestamp(timestamp):
    # UNIX timestamp -> "YYYY-MM-DD HH:MM:SS"
    try:
        if timestamp is None or not isinstance(timestamp, (int, float)) or timestamp <= 0:
            return ''
        if timestamp > 32503680000:  # 3000-01-01 제한
            return ''
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ''

def collect_news_data(symbol, from_date, to_date, max_calls_per_minute=60):
    all_news = []
    call_count = 0
    current_from = datetime.strptime(from_date, "%Y-%m-%d")
    current_to = datetime.strptime(to_date, "%Y-%m-%d")
    while current_from <= current_to:
        from_str = current_from.strftime("%Y-%m-%d")
        to_str = (current_from + timedelta(days=7)).strftime("%Y-%m-%d")
        if current_from + timedelta(days=7) > current_to:
            to_str = to_date

        params = {
            'symbol': symbol,
            'from': from_str,
            'to': to_str,
            'token': API_KEY
        }
        response = requests.get(BASE_URL, params=params)
        call_count += 1

        if response.status_code == 200:
            news_list = response.json()
            for news in news_list:
                timestamp = news.get('datetime')
                readable_date = safe_convert_timestamp(timestamp)
                data = {
                    'date': readable_date,
                    'title': news.get('headline', ''),
                    'summary': news.get('summary', ''),
                    'related': news.get('related', symbol),  # 관련 종목
                    'ticker': symbol  # ticker 컬럼 추가 (종목 구분용)
                }
                all_news.append(data)
        else:
            print(f"API 호출 오류 {response.status_code} - {response.text}")

        if call_count >= max_calls_per_minute:
            print("60회 호출 도달, 60초 대기 중...")
            time.sleep(60)
            call_count = 0

        current_from += timedelta(days=7)
    return all_news

def save_news_to_csv(news_data, filename):
    # ticker 컬럼 추가 반영
    fieldnames=['date', 'title', 'summary', 'related', 'ticker']

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in news_data:
            writer.writerow(record)
    print(f"{filename} 파일에 저장 완료")

# (실행) 여러 기업 뉴스 일괄 수집
symbols = ['NVDA', 'MSFT', 'AAPL']
from_date = '2020-01-01'
to_date = '2024-12-31'
all_news_data = []
for symbol in symbols:
    print(f"{symbol} 뉴스 수집 시작...")
    news_data = collect_news_data(symbol, from_date, to_date)
    all_news_data.extend(news_data)
save_news_to_csv(all_news_data, "news_data.csv")

# 2. 주가 데이터 수집 및 저장
all_stock_data = []
for symbol in symbols:
    print(f"{symbol} 주가 데이터 수집 중...")
    df = yf.download(symbol, start=from_date, end=to_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index()
    df['Symbol'] = symbol
    df = df[['Symbol', 'Date', 'Open', 'Close']]
    all_stock_data.append(df)
result = pd.concat(all_stock_data, ignore_index=True)
result.to_csv("stock_data.csv", index=False, encoding='utf-8')
print("stock_data.csv 파일 저장 완료")