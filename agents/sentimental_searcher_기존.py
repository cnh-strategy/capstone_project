import os
import time
import requests
import csv
from datetime import datetime, timedelta
from collections import Counter


API_KEY = 'd30p739r01qnu2qv1tmgd30p739r01qnu2qv1tn0'
BASE_URL = 'https://finnhub.io/api/v1/company-news'

MAX_CALLS_PER_MINUTE = 60
MAX_CALLS_PER_DAY = 500  # 실 예로 무료 플랜 일일 제한 예상값, 실제는 Finnhub 대시보드 확인 필수

def safe_convert_timestamp(timestamp):
    try:
        if timestamp is None or not isinstance(timestamp, (int, float)) or timestamp <= 0:
            return ''
        if timestamp > 32503680000:
            return ''
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ''

def collect_news_data(symbol, from_date, to_date):
    all_news = []
    call_count_minute = 0  # 분당 호출수
    call_count_day = 0     # 일일 호출수

    current_from = datetime.strptime(from_date, "%Y-%m-%d")
    current_to = datetime.strptime(to_date, "%Y-%m-%d")

    while current_from <= current_to:
        if call_count_day >= MAX_CALLS_PER_DAY:
            print(f"일일 호출 제한({MAX_CALLS_PER_DAY}) 도달, 수집 종료")
            break

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
        call_count_minute += 1
        call_count_day += 1

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

        elif response.status_code == 429:
            print("API 요청 제한 도달, 5분 대기 후 재시도")
            time.sleep(300)
            call_count_minute = 0  # 대기 후 카운트 초기화
            continue  # 재시도

        else:
            print(f"API 호출 오류 {response.status_code} - {response.text}")

        if call_count_minute >= MAX_CALLS_PER_MINUTE:
            print(f"{MAX_CALLS_PER_MINUTE}회 호출 도달, 60초 대기 중...")
            time.sleep(60)
            call_count_minute = 0

        current_from += timedelta(days=7)

    return all_news


# 뉴스 데이터를 CSV 파일로
def save_news_to_csv(news_data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'title', 'summary', 'related'])
        writer.writeheader()   # 컬럼명 작성
        for record in news_data:
            writer.writerow(record)
    print(f"{filename} 파일에 저장 완료")


# 기업별 뉴스 개수 세기 (related 필드 사용)
def count_news_by_company(news_data):
    symbols = [news['related'] for news in news_data if 'related' in news]
    counts = Counter(symbols)
    return counts


# 메인 실행 부분
symbols = ['NVDA', 'MSFT', 'AAPL']
from_date = '2020-01-01'
to_date = '2024-12-31'


all_news_data = []


# 각 기업의 뉴스 데이터 수집 및 개수 출력
for symbol in symbols:
    print(f"{symbol} 뉴스 수집 시작...")
    news_data = collect_news_data(symbol, from_date, to_date)
    all_news_data.extend(news_data)   # 전체 리스트에 합치기
    counts = count_news_by_company(news_data)
    print(f"{symbol} 뉴스 건수: {counts.get(symbol, 0)}개")

# 모두 합쳐서 CSV로 저장
save_news_to_csv(all_news_data, "news_data.csv")