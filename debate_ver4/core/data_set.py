import pandas as pd
import numpy as np
import os
import yfinance as yf
from debate_ver4.config.agents import agents_info, dir_info
from agents.macro_classes.macro_funcs import macro_dataset


# Raw Dataset 생성
def fetch_ticker_data(ticker: str) -> pd.DataFrame:
    period = "2y"
    interval = "1d"
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    
    # 컬럼명 정리 (튜플 형태를 단순 문자열로 변환)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # 기본 기술적 지표
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)
    
    # Fundamental Agent용 추가 데이터
    try:
        # 환율 데이터 (USD/KRW)
        usd_krw = yf.download("USDKRW=X", period=period, interval=interval, auto_adjust=True)
        if not usd_krw.empty:
            df["USD_KRW"] = usd_krw["Close"].reindex(df.index, method='ffill')
        else:
            df["USD_KRW"] = 1300.0  # 기본값
        
        # 나스닥 지수
        nasdaq = yf.download("^IXIC", period=period, interval=interval, auto_adjust=True)
        if not nasdaq.empty:
            df["NASDAQ"] = nasdaq["Close"].reindex(df.index, method='ffill')
        else:
            df["NASDAQ"] = 15000.0  # 기본값
        
        # VIX 지수
        vix = yf.download("^VIX", period=period, interval=interval, auto_adjust=True)
        if not vix.empty:
            df["VIX"] = vix["Close"].reindex(df.index, method='ffill')
        else:
            df["VIX"] = 20.0  # 기본값
    except Exception as e:
        print(f"⚠️ 추가 지표 다운로드 실패: {e}")
        df["USD_KRW"] = 1300.0
        df["NASDAQ"] = 15000.0
        df["VIX"] = 20.0
    

    
    # Sentimental Agent용 감성 지표
    df["sentiment_mean"] = df["returns"].rolling(3).mean().fillna(0)
    df["sentiment_vol"] = df["returns"].rolling(3).std().fillna(0)
    
    df.dropna(inplace=True)
    return df

# 시퀀스 생성
def create_sequences(features, target, window_size=14):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

# 통합 함수
def build_dataset(ticker: str = "TSLA", save_dir=dir_info["data_dir"]):
    os.makedirs(save_dir, exist_ok=True)

    # Raw Dataset 생성
    df = fetch_ticker_data(ticker)
    
    # 원본 데이터를 CSV로 저장
    df.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True)
    
    # Agent별 데이터셋을 CSV로 저장
    for agent_id, _ in agents_info.items():

        # MacroSentiAgent 경우
        if agent_id == 'MacroSentiAgent':
            macro_dataset(ticker)
            print(f"✅ {ticker} {agent_id} dataset saved to CSV")
        else:
            # 사용 가능 한 피처만 선택
            col = agents_info[agent_id]["data_cols"]
            X = df[col]

            # 타겟을 상승/하락율로 변경 (기존 종가 예측은 주석 처리)
            # y = df["Close"].values.reshape(-1, 1)  # 기존: 절대 종가 예측
            # 기존: NaN을 0으로 처리 (문제 원인 - 노이즈 증가)
            # y = df["Close"].pct_change().shift(-1).fillna(0).values.reshape(-1, 1)
            # 수정: NaN 값을 제거하여 더 깨끗한 데이터 사용
            returns = df["Close"].pct_change().shift(-1)
            valid_mask = ~returns.isna()
            y = returns[valid_mask].values.reshape(-1, 1)

            # X 데이터도 동일한 마스크 적용
            X = X[valid_mask]

            X_seq, y_seq = create_sequences(X, y, window_size=agents_info[agent_id]["window_size"])
            samples, time_steps, features = X_seq.shape
            print(f"[{agent_id}] X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")

            # 시퀀스 데이터를 평면화
            flattened_data = []
            for sample_idx in range(samples):
                for time_idx in range(time_steps):
                    row = {
                        'sample_id': sample_idx,
                        'time_step': time_idx,
                        'target': y_seq[sample_idx, 0] if time_idx == time_steps - 1 else np.nan,  # 해당 샘플의 타겟값 (예측 대상)
                    }
                    # 각 피처 추가
                    for feat_idx, feat_name in enumerate(col):
                        row[feat_name] = X_seq[sample_idx, time_idx, feat_idx]
                    flattened_data.append(row)

            # DataFrame으로 변환하고 CSV 저장
            agent_df = pd.DataFrame(flattened_data)
            csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
            agent_df.to_csv(csv_path, index=False)

            print(f"✅ {ticker} {agent_id} dataset saved to CSV ({len(X_seq)} samples, {len(col)} features)")

# --------------------------------------------
# CSV 데이터 로드 함수들
# --------------------------------------------
def load_dataset(ticker, agent_id=None, save_dir=dir_info["data_dir"]):
    csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 피처 컬럼 추출 (sample_id, time_step, target 제외)
    feature_cols = [col for col in df.columns if col not in ['sample_id', 'time_step', 'target']]
    
    # 시퀀스 데이터로 재구성
    unique_samples = df['sample_id'].nunique()
    time_steps = df['time_step'].nunique()
    n_features = len(feature_cols)
    
    X = np.zeros((unique_samples, time_steps, n_features), dtype=np.float32)
    y = np.zeros((unique_samples, 1), dtype=np.float32)
    
    for i, sample_id in enumerate(df['sample_id'].unique()):
        sample_data = df[df['sample_id'] == sample_id].sort_values('time_step')
        X[i] = sample_data[feature_cols].values
        y[i, 0] = sample_data['target'].iloc[-1]  # 각 샘플의 타겟값
    
    return X, y, feature_cols

def get_latest_close_price(ticker, save_dir=dir_info["data_dir"]):
    # 원본 데이터에서 최신 Close 가격 가져오기
    raw_data_path = os.path.join(save_dir, f"{ticker}_raw_data.csv")
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, index_col=0)
        return float(df['Close'].iloc[-1])
    else:
        # 원본 데이터가 없으면 yfinance로 직접 가져오기
        import yfinance as yf
        data = yf.download(ticker, period="1d", interval="1d")
        return float(data['Close'].iloc[-1])


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))