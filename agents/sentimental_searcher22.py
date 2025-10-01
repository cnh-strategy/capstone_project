#### 뉴스 데이터 수집

import os
import time
import requests
import csv
from datetime import datetime, timedelta
from collections import Counter

API_KEY ='d30p739r01qnu2qv1tmgd30p739r01qnu2qv1tn0'
# API_KEY = os.getenv('FINN_API_KEY')
BASE_URL = 'https://finnhub.io/api/v1/company-news'

def safe_convert_timestamp(timestamp):
    try:
        if timestamp is None or not isinstance(timestamp, (int, float)) or timestamp <= 0:
            return ''
        if timestamp > 32503680000:  # 3000-01-01 00:00:00 UTC 제한
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
                    'related': news.get('related', symbol)
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
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'title', 'summary', 'related'])
        writer.writeheader()
        for record in news_data:
            writer.writerow(record)
    print(f"{filename} 파일에 저장 완료")

def count_news_by_company(news_data):
    symbols = [news['related'] for news in news_data if 'related' in news]
    counts = Counter(symbols)
    return counts

# 전체 뉴스 데이터를 모아서 한 번에 저장하며, 기업별 건수 출력
symbols = ['NVDA', 'MSFT', 'AAPL']
from_date = '2020-01-01'
to_date = '2024-12-31'

all_news_data = []

for symbol in symbols:
    print(f"{symbol} 뉴스 수집 시작...")
    news_data = collect_news_data(symbol, from_date, to_date)
    all_news_data.extend(news_data)
    counts = count_news_by_company(news_data)
    print(f"{symbol} 뉴스 건수: {counts.get(symbol, 0)}개")

save_news_to_csv(all_news_data, "news_data.csv")