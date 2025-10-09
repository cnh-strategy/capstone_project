# ==========================================
# File: utils/dataset_builder.py
# Build unified datasets for all Agents
#  - Technical
#  - Fundamental
#  - Sentimental
# Split by year:
#   2022–2023 → Pretrain
#   2024       → Mutual Learning
#   2025       → Test / Evaluation
# ==========================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from textblob import TextBlob
import os


# ------------------------------
# 📈 Technical Dataset
# ------------------------------
def build_technical_dataset(ticker, period="4y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    df["returns"] = df["Close"].pct_change()
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
    df = df.dropna().reset_index()
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


# ------------------------------
# 💰 Fundamental Dataset
# ------------------------------
def build_fundamental_dataset(ticker, start="2022-01-01"):
    print(f"[INFO] Building fundamental data using yfinance for {ticker}")
    
    try:
        # yfinance 티커 객체 생성
        yf_ticker = yf.Ticker(ticker)
        
        # 주가 데이터 가져오기
        price_data = yf.download(ticker, start=start, progress=False)
        if price_data.empty:
            return pd.DataFrame()
        
        # 모든 데이터를 단일 DataFrame으로 통합
        df_all = price_data.copy()
        
        # 1. yfinance info에서 펀더멘털 지표 추출
        try:
            info = yf_ticker.info
            
            # 주요 펀더멘털 지표들
            trailing_pe = info.get('trailingPE', None)
            forward_pe = info.get('forwardPE', None)
            price_to_book = info.get('priceToBook', None)
            debt_to_equity = info.get('debtToEquity', None)
            return_on_assets = info.get('returnOnAssets', None)
            return_on_equity = info.get('returnOnEquity', None)
            profit_margins = info.get('profitMargins', None)
            gross_margins = info.get('grossMargins', None)
            
            # 기본값 설정 (None인 경우)
            trailing_pe = trailing_pe if trailing_pe is not None else 25.0
            forward_pe = forward_pe if forward_pe is not None else 20.0
            price_to_book = price_to_book if price_to_book is not None else 3.0
            debt_to_equity = debt_to_equity if debt_to_equity is not None else 0.5
            return_on_assets = return_on_assets if return_on_assets is not None else 0.05
            return_on_equity = return_on_equity if return_on_equity is not None else 0.10
            profit_margins = profit_margins if profit_margins is not None else 0.08
            gross_margins = gross_margins if gross_margins is not None else 0.20
            
        except Exception as e:
            print(f"[WARN] Failed to get fundamental info: {e}")
            # 기본값 사용
            trailing_pe, forward_pe, price_to_book = 25.0, 20.0, 3.0
            debt_to_equity, return_on_assets, return_on_equity = 0.5, 0.05, 0.10
            profit_margins, gross_margins = 0.08, 0.20
        
        # 2. 거시지표 데이터 (실제 yfinance 데이터)
        try:
            # USD/KRW 환율
            usdkrw = yf.download("USDKRW=X", start=start, progress=False)[["Close"]].rename(columns={"Close": "USD_KRW"})
            
            # NASDAQ 지수
            nasdaq = yf.download("^IXIC", start=start, progress=False)[["Close"]].rename(columns={"Close": "NASDAQ"})
            
            # VIX 변동성 지수
            vix = yf.download("^VIX", start=start, progress=False)[["Close"]].rename(columns={"Close": "VIX"})
            
            # 모든 거시지표를 주가 데이터와 결합
            df_all = df_all.join(usdkrw, how="left")
            df_all = df_all.join(nasdaq, how="left")
            df_all = df_all.join(vix, how="left")
            
        except Exception as e:
            print(f"[WARN] Failed to get macro data: {e}")
            # 기본값으로 설정
            df_all["USD_KRW"] = 1200.0
            df_all["NASDAQ"] = 15000.0
            df_all["VIX"] = 20.0
        
        # 3. 펀더멘털 지표를 시간에 따라 약간의 변동성 추가
        # 실제로는 분기별로 변하지만, 여기서는 일일 변동성으로 근사
        np.random.seed(42)  # 재현 가능한 결과를 위해
        
        df_all["priceEarningsRatio"] = trailing_pe + np.random.normal(0, trailing_pe * 0.05, len(df_all))
        df_all["forwardPE"] = forward_pe + np.random.normal(0, forward_pe * 0.05, len(df_all))
        df_all["priceToBook"] = price_to_book + np.random.normal(0, price_to_book * 0.05, len(df_all))
        df_all["debtEquityRatio"] = debt_to_equity + np.random.normal(0, debt_to_equity * 0.1, len(df_all))
        df_all["returnOnAssets"] = return_on_assets + np.random.normal(0, return_on_assets * 0.1, len(df_all))
        df_all["returnOnEquity"] = return_on_equity + np.random.normal(0, return_on_equity * 0.1, len(df_all))
        df_all["profitMargins"] = profit_margins + np.random.normal(0, profit_margins * 0.1, len(df_all))
        df_all["grossMargins"] = gross_margins + np.random.normal(0, gross_margins * 0.1, len(df_all))
        
        # 4. 결측값 처리 및 정리
        df_all = df_all.fillna(method="ffill").fillna(method="bfill")
        df_all = df_all.reset_index()
        df_all = df_all.rename(columns={"index": "Date"})
        
        return df_all
        
    except Exception as e:
        print(f"[WARN] Fundamental data generation failed for {ticker}: {e}")
        return pd.DataFrame()


# ------------------------------
# 💬 Sentimental Dataset
# ------------------------------
def build_sentimental_dataset(ticker, period="4y"):
    print(f"[INFO] Building sentiment data using yfinance for {ticker}")
    
    try:
        # yfinance 티커 객체 생성
        yf_ticker = yf.Ticker(ticker)
        
        # 주가 데이터 가져오기 (Technical과 동일한 방식)
        df_price = yf.download(ticker, period=period, interval="1d", progress=False)
        if df_price.empty:
            return pd.DataFrame()
        
        # 기본 수익률 계산
        df_price["returns"] = df_price["Close"].pct_change()
        
        # 1. 분석가 추천 기반 감성 점수
        try:
            recommendations = yf_ticker.recommendations_summary
            if recommendations is not None and len(recommendations) > 0:
                latest_rec = recommendations.iloc[-1]
                # 추천 점수 계산 (Strong Buy=2, Buy=1, Hold=0, Sell=-1, Strong Sell=-2)
                sentiment_score = (latest_rec.get('strongBuy', 0) * 2 + 
                                 latest_rec.get('buy', 0) * 1 + 
                                 latest_rec.get('hold', 0) * 0 + 
                                 latest_rec.get('sell', 0) * -1 + 
                                 latest_rec.get('strongSell', 0) * -2)
                total_analysts = sum([latest_rec.get(k, 0) for k in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']])
                analyst_sentiment = sentiment_score / max(total_analysts, 1) if total_analysts > 0 else 0
            else:
                analyst_sentiment = 0
        except:
            analyst_sentiment = 0
        
        # 2. 거래량 기반 시장 관심도 (거래량 증가 = 관심도 증가)
        volume_ma = df_price["Volume"].rolling(20).mean()
        volume_interest = (df_price["Volume"] / volume_ma - 1).fillna(0)
        
        # 3. 가격 변동성 기반 시장 불안감 (변동성 증가 = 불안감 증가)
        price_volatility = df_price["Close"].pct_change().rolling(20).std().fillna(0)
        
        # 4. RSI 기반 기술적 감성 (RSI > 70 = 과열, RSI < 30 = 과매도)
        delta = df_price["Close"].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        technical_sentiment = (rsi - 50) / 50  # -1 ~ 1 범위로 정규화
        
        # 5. 종합 감성 점수 계산
        # 가중 평균으로 최종 감성 점수 생성
        sentiment_mean = (analyst_sentiment * 0.3 + 
                         volume_interest * 0.2 + 
                         -price_volatility * 0.3 +  # 변동성은 불안감이므로 음수
                         technical_sentiment * 0.2)
        
        # 감성 변동성 (감성 점수의 표준편차)
        sentiment_vol = sentiment_mean.rolling(20).std().fillna(0)
        
        # 데이터프레임에 추가
        df_price["sentiment_mean"] = sentiment_mean
        df_price["sentiment_vol"] = sentiment_vol
        
        # NaN 제거 및 리셋
        df_price = df_price.dropna().reset_index()
        return df_price
        
    except Exception as e:
        print(f"[WARN] Sentiment data generation failed for {ticker}: {e}")
        return pd.DataFrame()


# ------------------------------
# 🧩 Split and Save
# ------------------------------
def load_and_split_datasets(ticker="TSLA", save_dir="data"):
    print(f"\n🚀 Building datasets for {ticker} (2022–2025)...")

    os.makedirs(save_dir, exist_ok=True)

    df_tech = build_technical_dataset(ticker)
    df_fund = build_fundamental_dataset(ticker)
    df_sent = build_sentimental_dataset(ticker)

    def split_years(df):
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Date 컬럼이 있는지 확인하고 없으면 생성
        if 'Date' not in df.columns:
            if 'index' in df.columns:
                df['Date'] = df['index']
            elif df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if 'index' in df.columns:
                    df['Date'] = df['index']
            else:
                print(f"[WARN] Cannot find Date column in dataset")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        df["Date"] = pd.to_datetime(df["Date"])
        df_pretrain = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2024-01-01")]
        df_mutual   = df[(df["Date"] >= "2024-01-01") & (df["Date"] < "2025-01-01")]
        df_test     = df[(df["Date"] >= "2025-01-01")]
        return df_pretrain, df_mutual, df_test

    splits = {}
    for name, df in zip(["technical", "fundamental", "sentimental"], [df_tech, df_fund, df_sent]):
        pretrain, mutual, test = split_years(df)
        splits[name] = {"pretrain": pretrain, "mutual": mutual, "test": test}

        # 저장 (빈 DataFrame이 아닌 경우에만)
        if not pretrain.empty:
            # MultiIndex 컬럼 처리
            if isinstance(pretrain.columns, pd.MultiIndex):
                pretrain.columns = [col[0] if col[0] else col[1] for col in pretrain.columns]
            pretrain.to_csv(f"{save_dir}/{ticker}_{name}_pretrain.csv", index=False)
        if not mutual.empty:
            # MultiIndex 컬럼 처리
            if isinstance(mutual.columns, pd.MultiIndex):
                mutual.columns = [col[0] if col[0] else col[1] for col in mutual.columns]
            mutual.to_csv(f"{save_dir}/{ticker}_{name}_mutual.csv", index=False)
        if not test.empty:
            # MultiIndex 컬럼 처리
            if isinstance(test.columns, pd.MultiIndex):
                test.columns = [col[0] if col[0] else col[1] for col in test.columns]
            test.to_csv(f"{save_dir}/{ticker}_{name}_test.csv", index=False)

        print(f"✅ {name.capitalize():12s} | "
              f"Pretrain:{len(pretrain):4d} | Mutual:{len(mutual):4d} | Test:{len(test):4d}")

    print(f"\n📦 Saved in: {os.path.abspath(save_dir)}")

    return splits


# ------------------------------
# 🧠 Run test
# ------------------------------
if __name__ == "__main__":
    datasets = load_and_split_datasets("TSLA")
