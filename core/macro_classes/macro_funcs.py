from core.macro_classes.macro_class_dataset import MacroAData

# 매크로 데이터셋 생성 함수 (build_dataset 기능)
# 데이터셋만 생성 (모델 학습은 별도 함수로 분리)
def macro_dataset(ticker_name, train_model=False):
    """
    매크로 데이터셋 생성 함수
    - train_model=False: 데이터셋만 생성 (기본값)
    - train_model=True: 데이터셋 + 모델 생성
    """
    macro_data_agent = MacroAData(ticker_name)
    macro_data_agent.fetch_data()
    macro_data_agent.add_features()
    macro_data_agent.save_csv()
    macro_data_agent.make_close_price()
    print(f"[macro_dataset] macro: 데이터셋 생성> {ticker_name}")

    return True
