from agents.macro_classes.macro_class_dataset import MacroAData
from debate_ver4.config.agents import dir_info
'''
티커 통합 모델
'''

model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]

# 매크로 데이터셋 생성 함수 (build_dataset 기능)
# 모델도 함께 생성됨
def macro_dataset(ticker_name):
    print(f"[TRACE B] macro_dataset() start for {ticker_name}")
    macro_data_agent = MacroAData(ticker_name)
    macro_data_agent.fetch_data()
    macro_data_agent.add_features()
    macro_data_agent.save_csv()
    macro_data_agent.make_close_price()
    print(f"macro: 데이터셋 생성> {ticker_name}")

    macro_data_agent.model_maker()
    print(f"macro: 모델 생성> {ticker_name}")



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


