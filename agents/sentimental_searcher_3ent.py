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
API_KEY = 'YOUR_KEY' # 실제 API 키로 교체
BASE_URL_EODHD = 'https://eodhd.com/api/news'

# 상태 파일명 (이번 실행에서는 사용하지 않지만, 함수 정의는 유지)
STATUS_FILE = 'collection_status.json'

def load_status():
    # 이 함수는 사용되지 않지만, 다른 곳에서 호출될까봐 남겨둡니다.
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {'completed_symbols': []}

def save_status(status):
    # 이 함수는 사용되지 않지만, 다른 곳에서 호출될까봐 남겨둡니다.
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
                    # 감성 분석 필드
                    'sentiment_score': news.get('sentiment', '')
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

def save_news_to_csv(news_data, filename, mode='a'):
    fieldnames=['date', 'title', 'summary', 'related', 'ticker', 'sentiment_score']

    # 'a' (추가) 모드일 때만 파일 존재 여부를 확인
    file_exists = os.path.exists(filename) and mode == 'a' 
    
    with open(filename, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # 파일이 새로 생성되거나 'w' (쓰기) 모드일 때만 헤더를 작성
        if mode == 'w' or not file_exists:
            writer.writeheader()
        
        for record in news_data:
            writer.writerow(record)
    print(f"데이터 {len(news_data)}개를 {filename} 파일에 저장 완료")


# ==============================================================================
# ⭐⭐⭐ 메인 실행 블록: 데이터 수집 기간 및 초기화 수정 적용 ⭐⭐⭐

symbols = ['NVDA', 'MSFT', 'AAPL']

# 1. 5년 기간 설정
from_date = '2020-01-01'      # 시작 날짜: 2020년 1월 1일
to_date_news = '2024-12-31'  # 뉴스 종료 날짜: 2024년 12월 31일
to_date_stock = '2025-01-01' # yfinance 주가 종료 날짜 (2024년 12월 31일까지 포함)

print(f"**데이터 수집 기간:** {from_date} 부터 {to_date_news} 까지")

NEWS_FILE = "news_data.csv"
STOCK_FILE = "stock_data.csv"

# 1. 기존 파일 삭제 (완전 초기화)
# 이 블록을 통해 이전 수집된 news_data.csv와 상태 파일을 삭제하고 새로 시작합니다.
if os.path.exists(NEWS_FILE):
    os.remove(NEWS_FILE)
    print(f"기존 {NEWS_FILE} 파일을 삭제했습니다.")
if os.path.exists(STATUS_FILE):
    os.remove(STATUS_FILE)
    print(f"기존 {STATUS_FILE} 파일을 삭제했습니다.")
    

# 2. 뉴스 데이터 수집 및 저장 (처음부터)
print("\n--- 뉴스 데이터 수집 시작 (EODhd, 감성 점수 포함) ---")
# 첫 번째 종목은 'w' (덮어쓰기) 모드로, 이후 종목은 'a' (추가) 모드로 저장합니다.
is_first_symbol = True
news_collection_successful = True # 뉴스 수집 완료 상태 추적 변수

for symbol in symbols:
    print(f"[{symbol}] 뉴스 수집 시작...")
    
    # EODhd API 호출 및 데이터 수집
    collected_news, last_offset = collect_news_data_eodhd(symbol, from_date, to_date_news) 
    
    if collected_news:
        save_mode = 'w' if is_first_symbol else 'a'
        save_news_to_csv(collected_news, NEWS_FILE, mode=save_mode)
        is_first_symbol = False
        
    if last_offset != -1:
        # 수집이 중간에 중단된 경우 (API 오류 등)
        news_collection_successful = False
        print(f"[{symbol}] 수집이 중간에 중단되었습니다. 다음 실행 시 재개하려면 코드를 수정해야 합니다.")
        break
    
    print(f"[{symbol}] 뉴스 수집 완료.")


# 3. 주가 데이터 수집 및 저장 
print("\n--- 주가 데이터 수집 시작 ---")
if news_collection_successful:
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