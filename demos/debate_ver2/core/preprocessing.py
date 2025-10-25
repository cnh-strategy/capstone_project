import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------
# 1️⃣ 데이터 로드 및 가공
# --------------------------------------------
def fetch_ticker_data(ticker: str, period="2y", interval="1d"):
    """yfinance로 주가 데이터 다운로드"""
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
    
    # Fundamental Agent용 재무 지표 (가상 데이터)
    df["priceEarningsRatio"] = df["Close"] / 2.0  # 가상 PE 비율
    df["forwardPE"] = df["Close"] / 1.8  # 가상 Forward PE
    df["priceToBook"] = df["Close"] / 1.5  # 가상 PB 비율
    
    # Sentimental Agent용 감성 지표
    df["sentiment_mean"] = df["returns"].rolling(3).mean().fillna(0)
    df["sentiment_vol"] = df["returns"].rolling(3).std().fillna(0)
    
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    """RSI 계산"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

# --------------------------------------------
# 2️⃣ 시퀀스 생성
# --------------------------------------------
def create_sequences(features, target, window_size=7):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

# --------------------------------------------
# 3️⃣ 데이터 전처리
# --------------------------------------------
def prepare_dataset(df, feature_cols, target_col="Close", window_size=7):
    # X만 스케일링, y는 원본 유지
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[feature_cols])
    y_scaled = df[target_col].values.reshape(-1, 1)  # y는 스케일링하지 않음

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)
    return (
        X_seq.astype(np.float32),
        y_seq.astype(np.float32),
        scaler_X,
        None,  # scaler_y는 더 이상 사용하지 않음
        feature_cols
    )

# --------------------------------------------
# 4️⃣ 통합 함수
# --------------------------------------------
def build_dataset(ticker, save_dir="data/processed", window_size=7):
    os.makedirs(save_dir, exist_ok=True)
    df = fetch_ticker_data(ticker)
    
    # 원본 데이터를 CSV로 저장
    df.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True)
    print(f"✅ {ticker} raw data saved to CSV ({len(df)} samples)")
    
    # Agent별로 다른 피처 세트 사용
    feature_cols = {
        "technical": ["Open", "High", "Low", "Close", "Volume", "returns", "sma_5", "sma_20", "rsi", "volume_z"],
        "fundamental": ["Open", "High", "Low", "Close", "Volume", "returns", "sma_5", "sma_20", "rsi", "volume_z",
                       "USD_KRW", "NASDAQ", "VIX", "priceEarningsRatio", "forwardPE", "priceToBook"],
        "sentimental": ["returns", "sentiment_mean", "sentiment_vol", "Close", "Volume", "Open", "High", "Low"]
    }
    
    # 기본 피처 (모든 Agent가 공통으로 사용)
    base_feature_cols = ["Open", "High", "Low", "Close", "Volume", "returns", "sma_5", "sma_20", "rsi", "volume_z"]
    
    # 기본 데이터셋 생성
    X, y, sx, sy, cols = prepare_dataset(df, base_feature_cols, "Close", window_size)

    # Agent별 데이터셋을 CSV로 저장
    for agent_type, agent_features in feature_cols.items():
        # 사용 가능한 피처만 선택
        available_features = [col for col in agent_features if col in df.columns]
        print(f"   {agent_type} Agent: {len(available_features)}개 피처 사용 - {available_features}")
        
        # 최소 피처 수 확인
        if len(available_features) < 3:
            print(f"   ⚠️ {agent_type} Agent 피처 부족, 기본 피처 사용")
            available_features = base_feature_cols
        
        X_agent, y_agent, sx_agent, sy_agent, cols_agent = prepare_dataset(df, available_features, "Close", window_size)
        
        # 시퀀스 데이터를 DataFrame으로 변환하여 CSV로 저장
        # X_agent shape: (samples, window_size, features)
        # 이를 (samples * window_size, features + sample_id + time_step) 형태로 변환
        samples, time_steps, features = X_agent.shape
        
        # 시퀀스 데이터를 평면화
        flattened_data = []
        for sample_idx in range(samples):
            for time_idx in range(time_steps):
                row = {
                    'sample_id': sample_idx,
                    'time_step': time_idx,
                    'target': y_agent[sample_idx, 0]  # 해당 샘플의 타겟값 (예측 대상)
                }
                # 각 피처 추가
                for feat_idx, feat_name in enumerate(cols_agent):
                    row[feat_name] = X_agent[sample_idx, time_idx, feat_idx]
                flattened_data.append(row)
        
        # DataFrame으로 변환하고 CSV 저장
        agent_df = pd.DataFrame(flattened_data)
        csv_path = os.path.join(save_dir, f"{ticker}_{agent_type}_dataset.csv")
        agent_df.to_csv(csv_path, index=False)
        
        # 스케일러 정보도 별도로 저장 (y 스케일러는 더 이상 사용하지 않음)
        scaler_info = {
            'feature_names': cols_agent,
            'scaler_X_scale': sx_agent.scale_.tolist() if hasattr(sx_agent, 'scale_') else None,
            'scaler_X_min': sx_agent.min_.tolist() if hasattr(sx_agent, 'min_') else None,
            'scaler_y_scale': None,  # y는 스케일링하지 않음
            'scaler_y_min': None,    # y는 스케일링하지 않음
        }
        
        scaler_df = pd.DataFrame([scaler_info])
        scaler_path = os.path.join(save_dir, f"{ticker}_{agent_type}_scaler.csv")
        scaler_df.to_csv(scaler_path, index=False)
        
        print(f"✅ {ticker} {agent_type} dataset saved to CSV ({len(X_agent)} samples, {len(available_features)} features)")
    
    # 기본 데이터셋도 CSV로 저장
    samples, time_steps, features = X.shape
    flattened_data = []
    for sample_idx in range(samples):
        for time_idx in range(time_steps):
            row = {
                'sample_id': sample_idx,
                'time_step': time_idx,
                'target': y[sample_idx, 0]
            }
            for feat_idx, feat_name in enumerate(cols):
                row[feat_name] = X[sample_idx, time_idx, feat_idx]
            flattened_data.append(row)
    
    base_df = pd.DataFrame(flattened_data)
    base_csv_path = os.path.join(save_dir, f"{ticker}_base_dataset.csv")
    base_df.to_csv(base_csv_path, index=False)
    
    # 기본 스케일러 정보 저장
    base_scaler_info = {
        'feature_names': cols,
        'scaler_X_scale': sx.scale_.tolist() if hasattr(sx, 'scale_') else None,
        'scaler_X_min': sx.min_.tolist() if hasattr(sx, 'min_') else None,
        'scaler_y_scale': sy.scale_.tolist() if hasattr(sy, 'scale_') else None,
        'scaler_y_min': sy.min_.tolist() if hasattr(sy, 'min_') else None,
    }
    
    base_scaler_df = pd.DataFrame([base_scaler_info])
    base_scaler_path = os.path.join(save_dir, f"{ticker}_base_scaler.csv")
    base_scaler_df.to_csv(base_scaler_path, index=False)
    
    print(f"✅ {ticker} base dataset saved to CSV ({len(X)} samples)")
    return X, y, sx, sy

# --------------------------------------------
# 5️⃣ CSV 데이터 로드 함수들
# --------------------------------------------
def load_csv_dataset(ticker, agent_type="base", save_dir="data/processed"):
    """CSV에서 데이터셋 로드"""
    csv_path = os.path.join(save_dir, f"{ticker}_{agent_type}_dataset.csv")
    scaler_path = os.path.join(save_dir, f"{ticker}_{agent_type}_scaler.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    
    # 스케일러 정보 로드
    scaler_df = pd.read_csv(scaler_path)
    scaler_info = scaler_df.iloc[0]
    
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
        y[i, 0] = sample_data['target'].iloc[0]  # 각 샘플의 타겟값
    
    # 스케일러 재구성 (y는 스케일링하지 않음)
    scaler_X = MinMaxScaler()
    scaler_y = None  # y는 스케일링하지 않음
    
    if scaler_info['scaler_X_scale'] is not None:
        scaler_X.scale_ = np.array(eval(scaler_info['scaler_X_scale']))
        scaler_X.min_ = np.array(eval(scaler_info['scaler_X_min']))
        scaler_X.feature_names_in_ = feature_cols
    
    return X, y, scaler_X, scaler_y, feature_cols

