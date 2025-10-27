import os
import time
import requests
import csv
from datetime import datetime, timedelta
from collections import Counter
import yfinance as yf
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore') # 경고 메시지 무시

# EODhd API 설정
API_KEY = '68e3a8c46e9a65.00465987' # 실제 API 키로 교체
BASE_URL_EODHD = 'https://eodhd.com/api/news'

# 상태 파일명 (이번 실행에서는 사용하지 않지만, 함수 정의는 유지)
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
    EOD Historical Data API를 사용하여 특정 기간의 뉴스 데이터를 수집（감성 점수 포함）
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
                    'sentiment_score': news.get('sentiment', '')
                }
                all_news.append(data)

            if len(news_list) < limit:
                break 
            else:
                offset += limit
                time.sleep(1)
            
        else:
            print(f"[{symbol}] API 호출 오류 {response.status_code} - {response.text}")
            print(f"[{symbol}] 오프셋 {offset}에서 수집 중단.")
            return all_news, offset 
            
    return all_news, -1 

def save_news_to_csv(news_data, filename, mode='a'):
    fieldnames=['date', 'title', 'summary', 'related', 'ticker', 'sentiment_score']

    file_exists = os.path.exists(filename) and mode == 'a' 
    
    with open(filename, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if mode == 'w' or not file_exists:
            writer.writeheader()
        
        for record in news_data:
            writer.writerow(record)
    print(f"데이터 {len(news_data)}개를 {filename} 파일에 저장 완료")


# ==============================================================================
# ⭐⭐⭐ 메인 실행 블록: 데이터 수집 기간 수정 적용 ⭐⭐⭐

symbols = ['NVDA', 'MSFT', 'AAPL']

# 1. 5년 기간 설정
from_date = '2020-01-01'        # 시작 날짜: 2020년 1월 1일
# ⭐ 뉴스 종료 날짜를 2025년 1월 31일로 변경
to_date_news = '2025-01-31'     
# 주식 종료 날짜는 2월 1일
to_date_stock = '2025-02-01' 

print(f"**데이터 수집 기간:** 뉴스: {from_date} 부터 {to_date_news} / 주가: {from_date} 부터 {to_date_stock} 이전까지")

NEWS_FILE = "news_data2.csv"
STOCK_FILE = "stock_data2.csv"

# 1. 기존 파일 삭제 (완전 초기화)
if os.path.exists(NEWS_FILE):
    os.remove(NEWS_FILE)
    print(f"기존 {NEWS_FILE} 파일을 삭제했습니다.")
if os.path.exists(STATUS_FILE):
    os.remove(STATUS_FILE)
    print(f"기존 {STATUS_FILE} 파일을 삭제했습니다.")
    

# 2. 뉴스 데이터 수집 및 저장 
print("\n--- 뉴스 데이터 수집 시작 (EODhd, 감성 점수 포함) ---")
is_first_symbol = True
news_collection_successful = True 

for symbol in symbols:
    print(f"[{symbol}] 뉴스 수집 시작...")
    
    collected_news, last_offset = collect_news_data_eodhd(symbol, from_date, to_date_news) 
    
    if collected_news:
        save_mode = 'w' if is_first_symbol else 'a'
        save_news_to_csv(collected_news, NEWS_FILE, mode=save_mode)
        is_first_symbol = False
        
    if last_offset != -1:
        news_collection_successful = False
        print(f"[{symbol}] 수집이 중간에 중단되었습니다.")
        break
    
    print(f"[{symbol}] 뉴스 수집 완료.")


# 3. 주가 데이터 수집 및 저장 
print("\n--- 주가 데이터 수집 시작 ---")
if news_collection_successful:
    all_stock_data = []
    for symbol in symbols:
        print(f"{symbol} 주가 데이터 수집 중 (yfinance)...")
        df = yf.download(symbol, start=from_date, end=to_date_stock) 
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df['Symbol'] = symbol
        df = df[['Symbol', 'Date', 'Open', 'Close']] 
        all_stock_data.append(df)
        
    result = pd.concat(all_stock_data, ignore_index=True)
    result.to_csv("stock_data.csv", index=False, encoding='utf-8') 
    print("stock_data2.csv 파일 저장 완료 (2025년 1월 데이터 포함)")
else:
    print("뉴스 수집이 완료되지 않아 주가 데이터 수집은 건너뜁니다.")