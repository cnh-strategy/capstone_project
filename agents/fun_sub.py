import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_price_history(ticker: str, end_date: str) -> pd.DataFrame:
    """
    지정한 티커와 기준일(end_date)로부터 1년 전 ~ end_date까지의 종가를 불러오는 함수

    Parameters:
        ticker (str): 종목 티커 (예: "AAPL")
        end_date (str): 기준일 (형식: "YYYY-MM-DD")

    Returns:
        pd.DataFrame: 1년치 일별 종가 DataFrame
    """
    # end_date를 datetime으로 변환
    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end - timedelta(days=365)  # 1년 전

    # yfinance에서 데이터 다운로드
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d")
    )["Close"].reset_index()

    # 컬럼명을 통일: 항상 ["Date", "Close"]
    df = df.rename(columns={df.columns[1]: "Close"})

    return df

# 예시 실행
df_aapl = get_price_history("NVDA", "2025-01-02")
print(df_aapl.tail())

def get_multi_index_prices(end_date: str) -> pd.DataFrame:
    """
    여러 지표의 1년치 Close 가격을 하나의 DataFrame으로 반환
    (Date, VIX, S&P500, NASDAQ, DXY, DOWJONES, US10Y)
    """
    tickers = {
        "DXY": "DX-Y.NYB",
        "NASDAQ": "^IXIC",
        "S&P500": "^GSPC",
        "DOWJONES": "^DJI",
        "VIX": "^VIX",
        "US10Y": "^TNX"
    }

    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end - timedelta(days=365)

    df = yf.download(
        list(tickers.values()),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False
    )["Close"]

    # 멀티인덱스 컬럼이면 풀기
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    # 컬럼명 매핑
    rename_map = {v: k for k, v in tickers.items()}
    df = df.rename(columns=rename_map)

    # 원하는 순서
    ordered_cols = ["VIX", "S&P500", "NASDAQ", "DXY", "DOWJONES", "US10Y"]
    df = df[ordered_cols]

    # 컬럼 인덱스 이름 제거 → "Ticker" 안 뜨게 함
    df.columns.name = None

    return df

# 실행 예시
# if __name__ == "__main__":
#     df_indices = get_multi_index_prices("2025-01-02")
#     print(df_indices.tail())


def get_quarterly_report(symbol: str, date: str) -> pd.Series:
    """
    특정 일자와 티커 기준으로 45일 딜레이 적용 후 가장 최신 분기 보고서를 반환

    Parameters:
        symbol (str): 티커 (예: "AAPL")
        date (str): 기준일 (예: "2024-12-31")

    Returns:
        pd.Series: 분기 보고서 데이터 (없으면 None 반환)
    """
    tk = yf.Ticker(symbol)

    # 분기 데이터
    income = tk.quarterly_financials.T
    balance = tk.quarterly_balance_sheet.T
    cashflow = tk.quarterly_cashflow.T
    info = tk.info               # 시가총액, 배당, 베타 등

    target_date = datetime.strptime(date, "%Y-%m-%d")

    # 분기별 loop
    valid_periods = []
    for period in income.index:
        report_available_date = period + timedelta(days=45)
        if report_available_date <= target_date:
            valid_periods.append(period)

    if not valid_periods:
        return None  # 해당 날짜까지 보고서 없음

    # 가장 최근 분기 선택
    latest_period = max(valid_periods)

    row_income = income.loc[latest_period] if latest_period in income.index else {}
    row_balance = balance.loc[latest_period] if latest_period in balance.index else {}
    row_cash = cashflow.loc[latest_period] if latest_period in cashflow.index else {}

    net_income = row_income.get("Net Income")
    revenue = row_income.get("Total Revenue")
    operating_income = row_income.get("Operating Income")
    gross_profit = row_income.get("Gross Profit")

    total_assets = row_balance.get("Total Assets")
    total_liabilities = row_balance.get("Total Liabilities")
    current_assets = row_balance.get("Current Assets")
    current_liabilities = row_balance.get("Current Liabilities")

    operating_cf = row_cash.get("Total Cash From Operating Activities")
    capex = row_cash.get("Capital Expenditures")
    free_cf = (operating_cf or 0) + (capex or 0)

    # 파생 지표
    profit_margin = net_income / revenue if revenue else None
    debt_to_equity = (
        total_liabilities / (total_assets - total_liabilities)
        if total_assets and total_liabilities else None
    )
    current_ratio = (
        current_assets / current_liabilities
        if current_assets and current_liabilities else None
    )

    # 티커 info 기반
    market_cap = info.get("marketCap")
    dividend_yield = info.get("dividendYield")
    beta = info.get("beta")
    forward_pe = info.get("forwardPE")
    pe = info.get("trailingPE")
    eps = info.get("trailingEps")
    pbr = info.get("priceToBook")

    return pd.Series({
        "symbol": symbol,
        "period": latest_period.strftime("%Y-%m-%d"),
        "year": latest_period.year,
        "net_income": net_income,
        "eps": eps,
        "roe": None,
        "pe": pe,
        "pbr": pbr,
        "revenue": revenue,
        "operating_income": operating_income,
        "gross_profit": gross_profit,
        "profit_margin": profit_margin,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "market_cap": market_cap,
        "dividend_yield": dividend_yield,
        "beta": beta,
        "forward_pe": forward_pe,
        "operating_cashflow": operating_cf,
        "capex": capex,
        "free_cashflow": free_cf,
    })
