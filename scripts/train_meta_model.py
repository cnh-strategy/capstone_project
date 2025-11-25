
import os
import sys
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.agents import dir_info

def train_meta_model(
    data_path="data/processed/ensemble_train.csv",
    model_out_path="models/ensemble_lightgbm.pkl",
    test_size=0.2
):
    """
    LightGBM 메타 모델 학습
    """
    if not os.path.exists(data_path):
        print(f"[Error] 학습 데이터가 없습니다: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"[{datetime.now()}] 데이터 로드 완료: {len(df)}행")

    # -------------------------------------------------------
    # 1. Feature Engineering
    # -------------------------------------------------------
    # 절대 가격(Price)을 사용하면 주가 레벨 변화에 민감해지므로
    # "예측 수익률(Return)"로 변환하여 학습.
    # Return = (Pred - Last) / Last
    
    # 입력 피처
    # Tech
    df['Tech_Ret'] = (df['Tech_Pred'] - df['Last_Close']) / df['Last_Close']
    df['Macro_Ret'] = (df['Macro_Pred'] - df['Last_Close']) / df['Last_Close']
    df['Senti_Ret'] = (df['Senti_Pred'] - df['Last_Close']) / df['Last_Close']
    
    # Target
    df['Target_Ret'] = (df['Next_Close'] - df['Last_Close']) / df['Last_Close']
    
    feature_cols = [
        'Tech_Ret', 'Tech_Conf', 'Tech_Unc',
        'Macro_Ret', 'Macro_Conf', 'Macro_Unc',
        'Senti_Ret', 'Senti_Conf', 'Senti_Unc'
    ]
    
    # 결측치 제거 (안전장치)
    df_clean = df.dropna(subset=feature_cols + ['Target_Ret'])
    print(f"전처리 후 데이터: {len(df_clean)}행")
    
    # -------------------------------------------------------
    # 2. 강제 학습 (데이터 부족해도) - 데모 목적
    # -------------------------------------------------------
    # 실제로는 데이터가 더 많아야 하지만, 지금은 프로세스 완성을 위해 최소한으로 진행
    if len(df_clean) < 5: # 최소 5개라도 있으면 학습 시도
        print("[Warn] 데이터가 매우 적습니다. Overfitting 주의.")
        test_size = 0.0 # 테스트셋 없음
    
    X = df_clean[feature_cols]
    y = df_clean['Target_Ret']

    # -------------------------------------------------------
    # 3. Time-Series Split
    # -------------------------------------------------------
    if test_size > 0:
        split_idx = int(len(df_clean) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y # 평가용으로 동일 데이터 사용

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # -------------------------------------------------------
    # 4. LightGBM 학습
    # -------------------------------------------------------
    # Regression Model
    model = lgb.LGBMRegressor(
        n_estimators=100, # 줄임
        learning_rate=0.05,
        max_depth=3, # 줄임
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='mse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # -------------------------------------------------------
    # 5. 평가
    # -------------------------------------------------------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    
    # Baseline (단순 평균)과 비교
    avg_pred = (X_test['Tech_Ret'] + X_test['Macro_Ret'] + X_test['Senti_Ret']) / 3
    base_mse = mean_squared_error(y_test, avg_pred)
    print(f"Baseline(Simple Avg) MSE: {base_mse:.6f}")
    
    # Feature Importance
    print("\nFeature Importance:")
    imps = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(imps)

    # -------------------------------------------------------
    # 6. 모델 저장
    # -------------------------------------------------------
    # 모델 저장
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(model, model_out_path)
    print(f"모델 저장 완료: {model_out_path}")

if __name__ == "__main__":
    train_meta_model()
