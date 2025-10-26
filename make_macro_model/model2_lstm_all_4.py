import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from debate_ver3_tmp.config.agents import dir_info
from make_macro_model.make_longterm_dataset_1 import MacroSentimentAgentDataset
'''
티커 통합 모델
'''

model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]

def sercher(ticker_name):
    macro_agent = MacroSentimentAgentDataset()
    macro_agent.fetch_data()
    macro_agent.close_price_fetch(ticker_name)
    macro_agent.add_features()
    macro_agent.save_csv()
    print(f"macro: 데이터셋 생성> {ticker_name}")

def make_lstm_macro_model(ticker_name, agent_id):
    # -------------------------------------------------------------
    # 0. raw 데이터 만들기
    # -------------------------------------------------------------
    sercher(ticker_name)

    # -------------------------------------------------------------
    # 1. 데이터 불러오기
    # -------------------------------------------------------------
    macro_df = pd.read_csv(f"data/macro_data/macro_sentiment.csv")
    price_df = pd.read_csv(f"data/macro_data/daily_closePrice_{ticker_name}.csv")

    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    price_df['Date'] = pd.to_datetime(price_df['Date'])

    # -------------------------------------------------------------
    # 2. 매크로 피처 확장 (원본 + 변화율)
    # -------------------------------------------------------------
    macro_features = [c for c in macro_df.columns if c != 'Date']
    macro_ret = macro_df[macro_features].pct_change()
    macro_ret.columns = [f"{c}_ret" for c in macro_ret.columns]
    macro_full = pd.concat([macro_df, macro_ret], axis=1)
    macro_full = macro_full.replace([np.inf, -np.inf], np.nan).dropna(subset=['Date']).fillna(0)

    # -------------------------------------------------------------
    # 3. 주가 기반 피처 생성 (각 종목별)
    # -------------------------------------------------------------
    target_ticker_list = ['AAPL', 'MSFT', 'NVDA']   # ← 이름을 맞춤
    
    if ticker_name in price_df.columns:
        price_df[f"{ticker_name}_ret1"] = price_df[ticker_name].pct_change()
        price_df[f"{ticker_name}_ma5"] = price_df[ticker_name].rolling(5).mean()
        price_df[f"{ticker_name}_ma10"] = price_df[ticker_name].rolling(10).mean()
    else:
        print(f"[WARN] '{ticker_name}' column not found in price_df.columns: {price_df.columns.tolist()}")
    
    price_df = price_df.fillna(method='bfill')

    # -------------------------------------------------------------
    # 4. 날짜 기준 병합
    # -------------------------------------------------------------
    merged_df = pd.merge(price_df, macro_full, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    print(f"[INFO] 병합 후 데이터 shape: {merged_df.shape}")

    # -------------------------------------------------------------
    # 5. Feature 선택
    # -------------------------------------------------------------
    macro_cols = [c for c in macro_full.columns if c != 'Date']
    price_cols = [c for c in merged_df.columns if any(t in c for t in target_ticker_list) and ('_ret' in c or '_ma' in c)]
    feature_cols = macro_cols + price_cols

    X_all = merged_df[feature_cols]

    # -------------------------------------------------------------
    # 6. 입력 스케일링
    # -------------------------------------------------------------
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_all)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # -------------------------------------------------------------
    # 7. 타깃 (현재 ticker_name만 예측)
    # -------------------------------------------------------------
    if ticker_name in merged_df.columns:
        merged_df[f"{ticker_name}_target"] = merged_df[ticker_name].pct_change().shift(-1)
        y_all = merged_df[[f"{ticker_name}_target"]].dropna().reset_index(drop=True)
    else:
        print(f"[WARN] '{ticker_name}' not found in merged_df.columns: {merged_df.columns.tolist()}")
        return  # 혹은 raise Exception("Ticker not found in merged_df")

    X_scaled = X_scaled.iloc[:len(y_all)]

    # -------------------------------------------------------------
    # 8. 출력 스케일링
    # -------------------------------------------------------------
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = scaler_y.fit_transform(y_all)

    # -------------------------------------------------------------
    # 9. 시퀀스 생성 함수
    # -------------------------------------------------------------
    def create_sequences(X, y, window=40):
        Xs, ys = [], []
        for i in range(len(X) - window):
            Xs.append(X.iloc[i:(i + window)].values)
            ys.append(y[i + window])
        return np.array(Xs), np.array(ys)

    # -------------------------------------------------------------
    # 10. 시퀀스 변환
    # -------------------------------------------------------------
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=40)
    split_idx = int(len(X_seq) * 0.8)

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # -------------------------------------------------------------
    # 11. 단일 아웃풋 LSTM 모델 정의
    # -------------------------------------------------------------
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # 단일 종목 예측
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mae')

    # -------------------------------------------------------------
    # 12. 학습
    # -------------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=60,
        batch_size=16,
        verbose=1
    )


    # 전체 모델 저장
    model.save(f"{model_dir}/{ticker_name}_{agent_id}.h5")
    joblib.dump(scaler_X, f"{model_dir}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{model_dir}/scaler_y.pkl")
    print(f"✅ {agent_id} model saved.\n✅ pretraining finished.\n")