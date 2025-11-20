from config.agents import dir_info
from core.macro_classes.macro_class_dataset import MacroAData
'''
티커 통합 모델
'''

model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]

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
    print(f"macro: 데이터셋 생성> {ticker_name}")

    if train_model:
        macro_data_agent.model_maker()
        print(f"macro: 모델 생성> {ticker_name}")


# 모델 학습 함수 (다른 에이전트와 동일하게 별도 함수로 분리)
def train_macro_model(ticker_name):
    """
    MacroAgent 모델 학습 함수
    - 데이터셋이 이미 생성되어 있어야 함
    """
    print(f"[INFO] MacroAgent 모델 학습 시작: {ticker_name}")
    macro_data_agent = MacroAData(ticker_name)
    macro_data_agent.model_maker()
    print(f"✅ MacroAgent 모델 학습 완료: {ticker_name}")



def macro_sercher(macro_agent, ticker_name):
    # macro_agent = MacroPredictor(
    #     agent_id='MacroSentiAgent',
    #     base_date=datetime.today(),
    #     window=40,
    #     ticker=ticker_name
    # )
    macro_agent.load_assets()             # 모델, 스케일러 등 불러오기
    macro_agent.fetch_macro_data()          # macro_df 불러오기
    X_tensor, X_scaled = macro_agent.prepare_features()  # 입력 시퀀스 준비
    print(f"■ macro_sercher StockData 생성 완료 ({ticker_name})")

    return X_tensor, X_scaled



# 사용 하지 않는 중
def macro_4_predictor(macro_agent, X_seq):
    pred_prices, target = macro_agent.m_predictor(X_seq)
    return pred_prices, target


