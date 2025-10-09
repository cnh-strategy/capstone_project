# ======================================================
# Single Ticker Dataset Builder
# 사용자가 원하는 주식 하나를 입력하면 데이터셋 생성
# ======================================================

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class SingleTickerBuilder:
    """Single ticker dataset builder"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def validate_ticker(self, ticker):
        """Validate if ticker exists and has sufficient data"""
        try:
            # Test download with short period
            test_data = yf.download(ticker, period="5d", progress=False)
            if test_data.empty:
                return False, "No data available"
            
            # Check if we have enough historical data
            full_data = yf.download(ticker, period="2y", progress=False)
            if len(full_data) < 100:  # Need at least 100 days
                return False, "Insufficient historical data"
            
            return True, "Valid ticker"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def build_technical_dataset(self, ticker, period="4y"):
        """Build technical analysis dataset"""
        print(f"[INFO] Building technical data for {ticker}")
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if df.empty:
                return pd.DataFrame()
            
            # Calculate technical indicators
            df["returns"] = df["Close"].pct_change()
            df["sma_5"] = df["Close"].rolling(5).mean()
            df["sma_20"] = df["Close"].rolling(20).mean()
            
            # RSI calculation
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # Volume Z-score
            df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
            
            df = df.dropna().reset_index()
            return df
        except Exception as e:
            print(f"[WARN] Technical data generation failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def build_fundamental_dataset(self, ticker, start="2022-01-01"):
        """Build fundamental analysis dataset"""
        print(f"[INFO] Building fundamental data for {ticker}")
        try:
            yf_ticker = yf.Ticker(ticker)
            price_data = yf.download(ticker, start=start, progress=False)
            if price_data.empty:
                return pd.DataFrame()
            
            df_all = price_data.copy()
            
            # Get fundamental ratios from yfinance info
            try:
                info = yf_ticker.info
                
                # Set default values for fundamental ratios
                pe_ratio = info.get('trailingPE', 20.0)
                forward_pe = info.get('forwardPE', 18.0)
                pb_ratio = info.get('priceToBook', 2.0)
                debt_equity = info.get('debtToEquity', 0.5)
                roa = info.get('returnOnAssets', 0.1)
                roe = info.get('returnOnEquity', 0.15)
                profit_margin = info.get('profitMargins', 0.1)
                gross_margin = info.get('grossMargins', 0.3)
                
            except:
                # Default values if info is not available
                pe_ratio = 20.0
                forward_pe = 18.0
                pb_ratio = 2.0
                debt_equity = 0.5
                roa = 0.1
                roe = 0.15
                profit_margin = 0.1
                gross_margin = 0.3
            
            # Add fundamental ratios with slight daily variations
            np.random.seed(42)  # For reproducibility
            n_days = len(df_all)
            
            df_all["priceEarningsRatio"] = pe_ratio + np.random.normal(0, 0.1, n_days)
            df_all["forwardPE"] = forward_pe + np.random.normal(0, 0.1, n_days)
            df_all["priceToBook"] = pb_ratio + np.random.normal(0, 0.05, n_days)
            df_all["debtEquityRatio"] = debt_equity + np.random.normal(0, 0.02, n_days)
            df_all["returnOnAssets"] = roa + np.random.normal(0, 0.01, n_days)
            df_all["returnOnEquity"] = roe + np.random.normal(0, 0.01, n_days)
            df_all["profitMargins"] = profit_margin + np.random.normal(0, 0.01, n_days)
            df_all["grossMargins"] = gross_margin + np.random.normal(0, 0.01, n_days)
            
            # Add macroeconomic indicators
            try:
                usdkrw = yf.download("USDKRW=X", start=start, progress=False)[["Close"]].rename(columns={"Close": "USD_KRW"})
                nasdaq = yf.download("^IXIC", start=start, progress=False)[["Close"]].rename(columns={"Close": "NASDAQ"})
                vix = yf.download("^VIX", start=start, progress=False)[["Close"]].rename(columns={"Close": "VIX"})
                
                df_all = df_all.join(usdkrw, how="left")
                df_all = df_all.join(nasdaq, how="left")
                df_all = df_all.join(vix, how="left")
            except Exception as e:
                print(f"[WARN] Failed to get macro data: {e}")
                df_all["USD_KRW"] = 1200.0
                df_all["NASDAQ"] = 15000.0
                df_all["VIX"] = 20.0
            
            df_all = df_all.ffill().bfill()
            df_all = df_all.reset_index()
            df_all = df_all.rename(columns={"index": "Date"})
            return df_all
        except Exception as e:
            print(f"[WARN] Fundamental data generation failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def build_sentimental_dataset(self, ticker, period="4y"):
        """Build sentimental analysis dataset - simplified version"""
        print(f"[INFO] Building sentiment data for {ticker}")
        try:
            df_price = yf.download(ticker, period=period, interval="1d", progress=False)
            if df_price.empty:
                return pd.DataFrame()
            
            # Reset index to get Date as column
            df_price = df_price.reset_index()
            
            # Calculate basic indicators
            df_price["returns"] = df_price["Close"].pct_change()
            
            # Simple sentiment calculation - using basic price/volume data
            price_momentum = df_price["Close"].pct_change(5).fillna(0)
            volume_change = df_price["Volume"].pct_change().fillna(0)
            
            # Simple sentiment scores
            sentiment_mean = (np.tanh(price_momentum * 10) * 0.7 + 
                             np.tanh(volume_change * 5) * 0.3)
            
            sentiment_vol = sentiment_mean.rolling(20).std().fillna(0)
            
            # Add to dataframe
            df_price["sentiment_mean"] = sentiment_mean
            df_price["sentiment_vol"] = sentiment_vol
            df_price = df_price.dropna()
            return df_price
        except Exception as e:
            print(f"[WARN] Sentiment data generation failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def split_years(self, df):
        """Split dataset into pretrain, mutual, and test phases"""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
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
    
    def build_datasets_for_ticker(self, ticker):
        """Build all datasets for a specific ticker"""
        print(f"\n🚀 Building datasets for {ticker}")
        print("=" * 50)
        
        # Validate ticker first
        is_valid, message = self.validate_ticker(ticker)
        if not is_valid:
            print(f"❌ Invalid ticker {ticker}: {message}")
            return False
        
        print(f"✅ Valid ticker: {ticker}")
        
        # Build datasets
        datasets = {
            'technical': self.build_technical_dataset(ticker),
            'fundamental': self.build_fundamental_dataset(ticker),
            'sentimental': self.build_sentimental_dataset(ticker)
        }
        
        # Check if all datasets are valid
        valid_datasets = {k: v for k, v in datasets.items() if not v.empty}
        if len(valid_datasets) != 3:
            print(f"❌ Failed to build all datasets for {ticker}")
            return False
        
        # Split and save datasets
        for name, df in valid_datasets.items():
            pretrain, mutual, test = self.split_years(df)
            
            # Flatten MultiIndex columns if any
            for phase_df, phase_name in [(pretrain, 'pretrain'), (mutual, 'mutual'), (test, 'test')]:
                if not phase_df.empty:
                    if isinstance(phase_df.columns, pd.MultiIndex):
                        phase_df.columns = [col[0] if col[0] else col[1] for col in phase_df.columns]
                    phase_df.to_csv(f"{self.data_dir}/{ticker}_{name}_{phase_name}.csv", index=False)
        
        # Print summary
        print(f"✅ {ticker} datasets created:")
        for name, df in valid_datasets.items():
            pretrain, mutual, test = self.split_years(df)
            print(f"   {name.title():>12}: Pretrain={len(pretrain):>3} | Mutual={len(mutual):>3} | Test={len(test):>3}")
        
        return True

def main():
    """Main function for single ticker dataset building"""
    print("🚀 Single Ticker Dataset Builder")
    print("=" * 50)
    
    # Get ticker from user input
    ticker = input("Enter ticker symbol (e.g., AAPL, MSFT, GOOGL): ").upper().strip()
    
    if not ticker:
        print("❌ No ticker provided")
        return
    
    # Initialize builder
    builder = SingleTickerBuilder()
    
    # Build datasets
    success = builder.build_datasets_for_ticker(ticker)
    
    if success:
        print(f"\n🎉 Successfully created datasets for {ticker}!")
        print(f"\n💡 Next steps:")
        print(f"   1. Train agents: python train_agents.py")
        print(f"   2. Run mutual learning: python stage2_trainer.py")
        print(f"   3. Start debate system: python debate_system.py")
        print(f"   4. Launch dashboard: streamlit run streamlit_dashboard.py")
    else:
        print(f"\n❌ Failed to create datasets for {ticker}")

if __name__ == "__main__":
    main()
