import requests
CAPSTONE_OPENAI_API='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'

FMP_API_KEY='4b6E6t9IEwN6uUTUzKWpx2WUsxCTVvC3'
url = f"https://financialmodelingprep.com/api/v3/income-statement/AAPL?period=quarter&apikey={FMP_API_KEY}"
data = requests.get(url).json()
print(type(data))
print(data)