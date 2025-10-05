import pandas as pd

"""
[지금 파이프라인에서 라벨 정의]
y_close_t1 = 다음날 종가
ret_t1 = 다음날 수익률
즉, 학습 타깃은 y_close_t1 또는 ret_t1 중 하나입니다.
close는 “오늘 종가”라서 타깃은 아니고 설명 변수로 쓸 수 있음입니다.

[상대 재무제표 가격은 반영하지 않음]
"""

# =========================
# 1. 파일 불러오기
# =========================
df_prices = pd.read_csv("2025/daily_closePrice.csv", parse_dates=["Date"])
df_markets = pd.read_csv("2025/markets_csv.csv", parse_dates=["Date"])
df_fund = pd.read_csv("2025/yfina_quarterly_summary.csv", parse_dates=["period"])

# =========================
# 2. 종가 데이터 (long 변환)
# =========================
df_prices_long = df_prices.melt(
    id_vars="Date", var_name="symbol", value_name="close"
).dropna(subset=["close"])

# =========================
# 3. 분기 재무제표 (사용 컬럼만 선별)
# =========================
use_cols = [
    "symbol", "period", "net_income", "revenue", "operating_income",
    "gross_profit", "profit_margin",
    "total_assets", "current_ratio",
    "operating_cashflow", "capex", "free_cashflow",
    "eps", "pe", "forward_pe", "pbr", "market_cap",
    "dividend_yield", "beta"
]
df_fund = df_fund[use_cols].sort_values(["symbol", "period"])

# =========================
# 4. 재무제표 → 종가 매핑 (asof merge, 45일 지연 반영)
# =========================
panels = []
for sym, grp in df_prices_long.groupby("symbol"):
    fund_sub = df_fund[df_fund["symbol"] == sym].copy()
    if fund_sub.empty:
        continue

    # 45일 딜레이 반영
    fund_sub["period_with_lag"] = fund_sub["period"] + pd.Timedelta(days=45)

    merged = pd.merge_asof(
        grp.sort_values("Date"),
        fund_sub.sort_values("period_with_lag"),
        left_on="Date", right_on="period_with_lag",
        by="symbol", direction="backward"
    )
    panels.append(merged)

panel = pd.concat(panels, ignore_index=True)

# =========================
# 5. 시장 데이터 매핑 + 파생변수 생성
# =========================
panel = panel.merge(df_markets, on="Date", how="left")

# 시장지표 수익률 (1일 변동률)
for col in ["NASDAQ", "S&P500", "DOWJONES", "DXY"]:
    panel[f"{col}_ret1"] = panel[col].pct_change(fill_method=None)

# 시장지표 차분
for col in ["US10Y", "VIX"]:
    panel[f"d{col}"] = panel[col].diff()

# =========================
# 6. 재무제표 결측치 처리 (심볼별 ffill → bfill)
# =========================
fill_cols = [
    "net_income", "revenue", "operating_income", "gross_profit", "profit_margin",
    "total_assets", "current_ratio",
    "operating_cashflow", "capex", "free_cashflow",
    "eps", "pe", "forward_pe", "pbr", "market_cap",
    "dividend_yield", "beta"
]
panel[fill_cols] = panel.groupby("symbol")[fill_cols].transform(lambda g: g.ffill().bfill())

# =========================
# 7. 기술적 지표 추가
# =========================
panel = panel.sort_values(["symbol", "Date"])
panel["ret_1"] = panel.groupby("symbol")["close"].pct_change(1)

# 이동평균 (5, 20, 60일)
for w in [5, 20, 60]:
    panel[f"ma_{w}"] = panel.groupby("symbol")["close"].transform(lambda x: x.rolling(w, min_periods=3).mean())

# 20일 변동성
panel["vol_20"] = panel.groupby("symbol")["ret_1"].transform(lambda x: x.rolling(20, min_periods=10).std())

# 모멘텀 (close / ma_20)
panel["mom_20"] = panel["close"] / panel["ma_20"]

# =========================
# 8. 레이블 생성 (다음날 종가 & 수익률)
# =========================
panel["y_close_t1"] = panel.groupby("symbol")["close"].shift(-1)               # 다음날 종가
panel["ret_t1"] = (panel["y_close_t1"] / panel["close"]) - 1                   # 다음날 수익률

# =========================
# 9. NaN 처리 및 저장
# =========================
dataset = panel.dropna(subset=["y_close_t1", "ret_t1", "close"]).copy()

dataset.to_csv("2025/fallback_data.csv", index=False)

print("최종 dataset 크기:", dataset.shape)
print("컬럼 예시:", dataset.columns.tolist()[:25])
print(dataset.head())


"""

"""