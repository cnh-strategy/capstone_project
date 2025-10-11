import os
import time
import requests
import csv
from datetime import datetime, timedelta
from collections import Counter
import yfinance as yf
import pandas as pd
import json

# EODhd API 설정
API_KEY = 'YOUR_KEY'
BASE_URL_EODHD = 'https://eodhd.com/api/news'

# 상태 파일명
STATUS_FILE = 'collection_status.json'

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {'completed_symbols': []}

def save_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=4)

def collect_news_data_eodhd(symbol, from_date, to_date):
    """
    EOD Historical Data API를 사용하여 특정 기간의 뉴스 데이터를 수집합니다.
    (감성 점수 필드 포함)
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
            'extended': 1      # 감성 점수와 같은 확장 필드를 요청
        }

        try:
            # API 호출
            response = requests.get(BASE_URL_EODHD, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[{symbol}] API 호출 중 네트워크 오류/타임아웃 발생: {e}")
            return all_news, offset 
        
        if response.status_code == 200:
            news_list = response.json()
            if not news_list:
                print(f"[{symbol}] 더 이상 뉴스 데이터가 없습니다.")
                break 

            for news in news_list:
                data = {
                    'date': news.get('date', ''),
                    'title': news.get('title', ''),
                    'summary': news.get('content', ''), 
                    'related': news.get('symbols', symbol), 
                    'ticker': symbol,
                    # 감성 분석 필드
                    'sentiment_score': news.get('sentiment', ''), 
                    'sentiment_text': news.get('sentiment_text', '') 
                }
                all_news.append(data)

            if len(news_list) < limit:
                break 
            else:
                offset += limit
                time.sleep(1) # API 부하를 줄이기 위한 1초 대기
            
        else:
            print(f"[{symbol}] API 호출 오류 {response.status_code} - {response.text}")
            print(f"[{symbol}] 오프셋 {offset}에서 수집 중단.")
            return all_news, offset 
            
    return all_news, -1 

def save_news_to_csv(news_data, filename):
    fieldnames=['date', 'title', 'summary', 'related', 'ticker', 'sentiment_score', 'sentiment_text']

    file_exists = os.path.exists(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for record in news_data:
            writer.writerow(record)
    print(f"추가 데이터 {len(news_data)}개를 {filename} 파일에 저장 완료")


# ==============================================================================
# ⭐⭐⭐ 메인 실행 블록: 데이터 수집 기간 수정 (2020-01-01 ~ 2024-12-31) ⭐⭐⭐

symbols = ['NVDA', 'MSFT', 'AAPL']

# 1. 5년 기간 설정
from_date = '2020-01-01'     # 시작 날짜: 2020년 1월 1일
to_date_news = '2024-12-31'  # 뉴스 종료 날짜: 2024년 12월 31일
to_date_stock = '2025-01-01' # yfinance 주가 종료 날짜 (2024년 12월 31일까지 포함)

print(f"**데이터 수집 기간:** {from_date} 부터 {to_date_news} 까지")

NEWS_FILE = "news_data.csv"

# 1. 뉴스 데이터 수집 및 저장
status = load_status()

# 기간 변경 시, 이전 수집 상태를 초기화하고 기존 파일 삭제
if os.path.exists(NEWS_FILE):
    print("수집 기간이 변경되어 기존 news_data.csv 및 상태 파일을 초기화합니다.")
    os.remove(NEWS_FILE)
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
    status = {'completed_symbols': []} # 상태 초기화


for symbol in symbols:
    if symbol in status.get('completed_symbols', []):
        print(f"[{symbol}] 이전 수집에서 완료되었으므로 건너뜁니다.")
        continue

    print(f"[{symbol}] 뉴스 수집 시작 (EODhd, 감성 점수 포함)...")
    
    collected_news, last_offset = collect_news_data_eodhd(symbol, from_date, to_date_news) 
    
    if collected_news:
        save_news_to_csv(collected_news, NEWS_FILE)
        
    if last_offset == -1:
        status['completed_symbols'].append(symbol)
        save_status(status)
        print(f"[{symbol}] 수집 및 상태 업데이트 완료.")
    else:
        print(f"[{symbol}] 수집이 중단되었습니다. 다음 실행 시 재개할 수 있도록 수집된 데이터는 저장되었습니다.")
        break 


# 2. 주가 데이터 수집 및 저장 
print("\n--- 주가 데이터 수집 시작 ---")
if len(status.get('completed_symbols', [])) == len(symbols):
    all_stock_data = []
    for symbol in symbols:
        print(f"{symbol} 주가 데이터 수집 중 (yfinance)...")
        # yfinance 다운로드 기간 수정 적용
        df = yf.download(symbol, start=from_date, end=to_date_stock) 
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df['Symbol'] = symbol
        df = df[['Symbol', 'Date', 'Open', 'Close']] 
        all_stock_data.append(df)
        
    result = pd.concat(all_stock_data, ignore_index=True)
    result.to_csv("stock_data.csv", index=False, encoding='utf-8') 
    print("stock_data.csv 파일 저장 완료")
else:
    print("뉴스 수집이 완료되지 않아 주가 데이터 수집은 건너뜁니다.")