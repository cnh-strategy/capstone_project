import yfinance as yf
import pandas as pd

symbols = ['NVDA', 'MSFT', 'AAPL']
from_date = '2020-01-01'
to_date = '2024-12-31'

all_stock_data = []

for symbol in symbols:
    print(f"{symbol} 주가 데이터 수집 중...")
    df = yf.download(symbol, start=from_date, end=to_date)

    # MultiIndex 컬럼 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # 두 번째 레벨 무시

    df = df.reset_index()
    df['Symbol'] = symbol

    # 필요한 컬럼만 단일화
    df = df[['Symbol', 'Date', 'Open', 'Close']]
    all_stock_data.append(df)

result = pd.concat(all_stock_data, ignore_index=True)

print("Concat 결과 컬럼:", result.columns.tolist())

result.to_csv("stock_data.csv", index=False, encoding='utf-8')
print("stock_data.csv 파일 저장 완료")
