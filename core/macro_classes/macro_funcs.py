import os
from core.macro_classes.macro_sub import MakeDatasetMacro

def macro_dataset(ticker_name, train_model=False):
    """
    DebateAgent → MacroAgent → data_set 구조를 유지하며,
    스케일러가 없을 경우 자동으로 학습모드로 전환합니다.
    """
    print(f"[INFO] macro_dataset() 실행 중... ticker={ticker_name}")

    dataset = MakeDatasetMacro(ticker=ticker_name, window=40)

    dataset.fetch_data()
    dataset.add_features()

    # 스케일러 경로
    x_scaler_path = f"{dataset.model_dir}/scalers/{ticker_name}_{dataset.agent_id}_xscaler.pkl"

    # 스케일러가 없거나 강제 학습 요청이면 -> trainset 생성
    if not os.path.exists(x_scaler_path) or train_model:
        print("[INFO] 스케일러 파일이 없어 학습 데이터셋을 생성합니다.")
        X_scaled, y_scaled, features = dataset.build_trainset()
        print(f"[OK] Trainset 생성 완료: X={X_scaled.shape}, y={y_scaled.shape}")
        return X_scaled, y_scaled, features

    # 스케일러가 존재하면 예측용 데이터 생성
    print("[INFO] 기존 스케일러 감지 → 예측용 데이터셋 생성")
    X_scaled, features = dataset.build_predictset()
    print(f"[OK] Predictset 생성 완료: X={X_scaled.shape}")
    return X_scaled, features
