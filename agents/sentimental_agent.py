import datetime
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()  # 현재 디렉토리의 .env 파일을 읽어 환경변수를 로드


import nltk
nltk.download('punkt')

nltk.download
nltk.download

api_key = os.getenv('FINN_API_KEY')  # FINN_API_KEY 환경변수 가져오기


nasdaq100_eng = {
    "NVIDIA": "NVDA", "Microsoft": "MSFT", "Apple": "AAPL", "Alphabet Class C": "GOOG",
    "Alphabet Class A": "GOOGL", "Amazon.com": "AMZN", "Meta Platforms": "META",
    "Broadcom": "AVGO", "Tesla": "TSLA", "Netflix": "NFLX", "Costco Wholesale": "COST",
    "Palantir Technologies Class A": "PLTR", "ASML Holding ADR": "ASML",
    "T-Mobile US": "TMUS", "Cisco Systems": "CSCO", "Advanced Micro Devices": "AMD",
    "AstraZeneca ADR": "AZN", "Linde": "LIN", "PepsiCo": "PEP", "AppLovin Class A": "APP",
    "Intuit": "INTU", "Shopify": "SHOP", "Booking Holdings": "BKNG",
    "Pinduoduo ADR": "PDD", "Qualcomm": "QCOM", "Texas Instruments": "TXN",
    "Intuitive Surgical": "ISRG", "Micron Technology": "MU", "Amgen": "AMGN",
    "Adobe": "ADBE", "Arm Holdings ADR": "ARM", "Gilead Sciences": "GILD",
    "Honeywell International": "HON", "Lam Research": "LRCX", "Palo Alto Networks": "PANW",
    "Applied Materials": "AMAT", "Comcast": "CMCSA", "Analog Devices": "ADI",
    "KLA": "KLAC", "Automatic Data Processing": "ADP", "MercadoLibre": "MELI",
    "Intel": "INTC", "Synopsys": "SNPS", "DoorDash": "DASH",
    "CrowdStrike Holdings": "CRWD", "Vertex Pharmaceuticals": "VRTX",
    "Cadence Design Systems": "CDNS", "Starbucks": "SBUX",
    "Constellation Energy": "CEG", "MicroStrategy": "MSTR",
    "O'Reilly Automotive": "ORLY", "Cintas": "CTAS", "Mondelez International": "MDLZ",
    "Thomson Reuters": "TRI", "Airbnb": "ABNB", "Marriott International": "MAR",
    "Autodesk": "ADSK", "PayPal Holdings": "PYPL", "Monster Beverage": "MNST",
    "Workday": "WDAY", "Fortinet": "FTNT", "CSX": "CSX",
    "Regeneron Pharmaceuticals": "REGN", "American Electric Power": "AEP",
    "Marvell Technology": "MRVL", "Axon Enterprise": "AXON",
    "NXP Semiconductors": "NXPI", "Roper Technologies": "ROP", "Fastenal": "FAST",
    "IDEXX Laboratories": "IDXX", "PACCAR": "PCAR", "Datadog": "DDOG",
    "Ross Stores": "ROST", "Paychex": "PAYX", "Atlassian": "TEAM", "Copart": "CPRT",
    "Take-Two Interactive": "TTWO", "Baker Hughes": "BKR", "Zscaler": "ZS",
    "Exelon": "EXC", "Xcel Energy": "XEL",
    "Coca-Cola Europacific Partners": "CCEP", "Electronic Arts": "EA",
    "Diamondback Energy": "FANG", "Verisk Analytics": "VRSK",
    "Keurig Dr Pepper": "KDP", "CoStar Group": "CSGP", "Charter Communications": "CHTR",
    "GE HealthCare Technologies": "GEHC", "Microchip Technology": "MCHP",
    "Cognizant Technology Solutions": "CTSH", "Kraft Heinz": "KHC",
    "Old Dominion Freight Line": "ODFL", "Dexcom": "DXCM",
    "Warner Bros. Discovery": "WBD", "Trade Desk": "TTD", "CDW": "CDW",
    "Biogen": "BIIB", "ON Semiconductor": "ON", "Lululemon Athletica": "LULU",
    "GlobalFoundries": "GFS"
}

# --- 1. 뉴스 데이터 수집 ---
today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=365)
today_str = today.strftime("%Y-%m-%d")
one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")

all_news_data = []
nasdaq_100_tickers = list(nasdaq100_eng.values())

for ticker in nasdaq_100_tickers:
    print(f"{ticker} 기업 뉴스 데이터 수집 중...")
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={one_year_ago_str}&to={today_str}&token={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        if news_data:
            for news in news_data:
                news['symbol'] = ticker
            all_news_data.extend(news_data)
        else:
            print(f"  -> {ticker} 기업의 뉴스가 없습니다.")
    except requests.exceptions.RequestException as e:
        print(f"  -> HTTP 오류가 발생했습니다: {e}")
        continue
    time.sleep(0.5)

news_df = pd.DataFrame(all_news_data)
if not news_df.empty:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_name = f"finnhub_nasdaq100_news_{timestamp}.csv"
    news_df.to_csv(csv_file_name, index=False, encoding='utf-8-sig')
    print(f"\n뉴스 데이터 수집 완료! 총 {len(news_df)}건을 {csv_file_name}에 저장했습니다.")
else:
    print("\n뉴스 데이터 수집 실패: 데이터가 없습니다.")

# --- 2. 주가 데이터 수집 ---
all_stock_data = []
for ticker in nasdaq_100_tickers:
    print(f"{ticker} 주가 데이터 수집 중...")
    data = yf.download(ticker, start=one_year_ago, end=today, auto_adjust=True, progress=False)
    if data.empty:
        print(f"  -> 경고: {ticker}에 대한 데이터가 없습니다.")
        continue
    data = data.reset_index()
    data['symbol'] = ticker
    data = data.rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    data['date'] = pd.to_datetime(data['date']).dt.date
    data = data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
    all_stock_data.append(data)

if all_stock_data:
    stock_df = pd.concat(all_stock_data, ignore_index=True)
    stock_df.to_csv("nasdaq100_stock_data.csv", index=False, encoding='utf-8-sig')
    print(f"\n주가 데이터 수집 완료! 총 {len(stock_df)}건을 nasdaq100_stock_data.csv에 저장했습니다.")
else:
    print("\n주가 데이터 수집 실패: 데이터가 없습니다.")

# --- 3. 데이터 전처리 및 모델 학습 (당일+전일 뉴스 병합) ---
try:
    news_file_path = glob.glob('finnhub_nasdaq100_news_*.csv')[-1]
    news_df = pd.read_csv(news_file_path)
    stock_df = pd.read_csv('nasdaq100_stock_data.csv')
    print("\n--- 데이터 로드 완료 ---")
    print("뉴스 데이터:", news_df.shape)
    print("주가 데이터:", stock_df.shape)
except IndexError:
    print("\n오류: 데이터 파일이 존재하지 않습니다. 먼저 데이터 수집 코드를 실행해주세요.")
    exit()

# 뉴스 데이터 전처리
news_df['date'] = pd.to_datetime(news_df['datetime'], unit='s').dt.date
news_df['headline_summary'] = news_df['headline'].fillna('') + ' ' + news_df['summary'].fillna('')

positive_words = ['growth', 'strong', 'increase', 'profit', 'gain', 'rise', 'up', 'boost', 'expand', 'win', 'success']
negative_words = ['decline', 'loss', 'down', 'fail', 'drop', 'cut', 'reduce', 'miss', 'slump', 'crisis']

def get_keyword_sentiment(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    pos_score = sum(1 for word in filtered_words if word in positive_words)
    neg_score = sum(1 for word in filtered_words if word in negative_words)
    return pos_score - neg_score

print("LLM 기반 감성 분석기 로딩 중...")
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
    def get_llm_sentiment(text):
        result = sentiment_analyzer(text[:512])
        if result[0]['label'] == 'POSITIVE':
            return result[0]['score']
        else:
            return -result[0]['score']
except Exception as e:
    print(f"LLM 모델 로딩 또는 실행 오류: {e}. LLM 기반 감성 분석을 건너뜁니다.")
    def get_llm_sentiment(text):
        return 0

news_df['keyword_sentiment'] = news_df['headline_summary'].apply(get_keyword_sentiment)
news_df['llm_sentiment'] = news_df['headline_summary'].apply(get_llm_sentiment)

# 날짜 포맷 정리
news_df['date'] = pd.to_datetime(news_df['date'])
stock_df['date'] = pd.to_datetime(stock_df['date'])

# 종가/타겟 생성
stock_df['close'] = pd.to_numeric(stock_df['close'], errors='coerce')
stock_df.dropna(subset=['close'], inplace=True)
stock_df.sort_values(by=['symbol', 'date'], inplace=True)
stock_df['target'] = (stock_df.groupby('symbol')['close'].shift(-1) > stock_df['close']).astype(int)
stock_df.dropna(subset=['target'], inplace=True)

stock_min = stock_df[['symbol', 'date', 'close', 'target']]

# --- 당일+전일 뉴스 감성 feature 집계 ---
# 전일 뉴스 데이터 복사 (date +1)
prev_news_df = news_df.copy()
prev_news_df['date'] = prev_news_df['date'] + pd.Timedelta(days=1)

news_all = pd.concat([news_df, prev_news_df], axis=0)
news_all = news_all.groupby(['symbol', 'date']).agg({
    'headline_summary': ' '.join,
    'keyword_sentiment': 'sum',
    'llm_sentiment': 'sum'
}).reset_index()

# 병합: 각 거래일에 대해 당일+전일 뉴스 감성 feature 적용
merged_df = pd.merge(
    stock_min, news_all,
    how='left',
    left_on=['symbol', 'date'],
    right_on=['symbol', 'date']
)

# 결측치 처리(뉴스 없는 거래일에는 0)
for col in ['keyword_sentiment', 'llm_sentiment']:
    merged_df[col] = merged_df[col].fillna(0)

print(f"\n병합 완료: {merged_df.shape}")
print(merged_df[['symbol', 'date', 'headline_summary', 'keyword_sentiment', 'llm_sentiment', 'close', 'target']].head())

# 머신러닝 분류 모델 학습 및 평가 예시
features = ['keyword_sentiment', 'llm_sentiment']
X = merged_df[features]
y = merged_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("분류 정확도:", accuracy_score(y_test, y_pred))
print("분류 보고서:")
print(classification_report(y_test, y_pred))

# 정확도
accuracy = accuracy_score(y_test, y_pred)
# 정밀도 (positive 클래스에 대해)
precision = precision_score(y_test, y_pred)
# 재현율 (positive 클래스에 대해)
recall = recall_score(y_test, y_pred)
# F1 스코어 (precision과 recall의 조화평균)
f1 = f1_score(y_test, y_pred)

print(f"분류 정확도: {accuracy:.4f}")
print(f"정밀도(Precision): {precision:.4f}")
print(f"재현율(Recall): {recall:.4f}")
print(f"F1 스코어: {f1:.4f}")

print("\n분류 보고서2:")
print(classification_report(y_test, y_pred))