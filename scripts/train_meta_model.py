
import os
import sys
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.agents import dir_info

def train_meta_model(
    data_path="data/processed/ensemble_train.csv",
    model_out_path="models/ensemble_lightgbm.pt"
):
    """
    LightGBM 메타 모델 학습
    개별 에이전트들과 동일하게 전체 데이터를 학습에 사용합니다.
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
    
    # 입력 피처: 예측값을 수익률로 변환
    # NaN이 있는 경우 0으로 채움 (해당 에이전트의 예측이 없을 때)
    df['Tech_Ret'] = ((df['Tech_Pred'] - df['Last_Close']) / df['Last_Close']).fillna(0.0)
    df['Macro_Ret'] = ((df['Macro_Pred'] - df['Last_Close']) / df['Last_Close']).fillna(0.0)
    df['Senti_Ret'] = ((df['Senti_Pred'] - df['Last_Close']) / df['Last_Close']).fillna(0.0)
    
    # Confidence와 Uncertainty도 NaN이면 0으로 채움
    df['Tech_Conf'] = df['Tech_Conf'].fillna(0.0)
    df['Tech_Unc'] = df['Tech_Unc'].fillna(0.0)
    df['Macro_Conf'] = df['Macro_Conf'].fillna(0.0)
    df['Macro_Unc'] = df['Macro_Unc'].fillna(0.0)
    df['Senti_Conf'] = df['Senti_Conf'].fillna(0.0)
    df['Senti_Unc'] = df['Senti_Unc'].fillna(0.0)
    
    # Target
    df['Target_Ret'] = (df['Next_Close'] - df['Last_Close']) / df['Last_Close']
    
    feature_cols = [
        'Tech_Ret', 'Tech_Conf', 'Tech_Unc',
        'Macro_Ret', 'Macro_Conf', 'Macro_Unc',
        'Senti_Ret', 'Senti_Conf', 'Senti_Unc'
    ]
    
    # Target_Ret만 필수 (예측값은 NaN이면 0으로 처리했으므로)
    df_clean = df.dropna(subset=['Target_Ret'])
    print(f"전처리 후 데이터: {len(df_clean)}행")
    
    # 디버깅: 각 피처의 NaN 개수 확인
    if len(df_clean) > 0:
        print(f"   피처별 NaN 개수:")
        for col in feature_cols:
            nan_count = df_clean[col].isna().sum()
            if nan_count > 0:
                print(f"     {col}: {nan_count}개")
    
    # -------------------------------------------------------
    # 2. 전체 데이터를 학습에 사용 (개별 에이전트와 동일)
    # -------------------------------------------------------
    # 개별 에이전트들(TechnicalAgent, MacroAgent, SentimentalAgent)과 동일하게
    # 전체 데이터를 학습에 사용합니다. 모델 성능 평가는 백테스팅으로 수행합니다.
    X = df_clean[feature_cols]
    y = df_clean['Target_Ret']
    
    X_train, y_train = X, y
    print(f"[INFO] 전체 {len(X_train)}개 샘플을 학습에 사용")

    # -------------------------------------------------------
    # 3. LightGBM 학습
    # -------------------------------------------------------
    # Regression Model
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )
    
    # 전체 데이터로 학습 (Early stopping 없음)
    model.fit(
        X_train, y_train,
        eval_metric='mse',
        callbacks=[
            lgb.log_evaluation(period=50)
        ]
    )
    
    print("[INFO] 모델 학습 완료. 실제 성능 평가는 백테스팅으로 수행하세요.")
    
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
