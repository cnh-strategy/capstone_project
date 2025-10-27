
from debate_ver3_tmp.config.agents import dir_info
from make_macro_model.make_longterm_dataset_1 import MacroSentimentAgentDataset
'''
티커 통합 모델
'''

model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]

def macro_dataset(ticker_name):
    macro_agent = MacroSentimentAgentDataset()
    macro_agent.fetch_data()
    macro_agent.close_price_fetch(ticker_name)
    macro_agent.add_features()
    macro_agent.save_csv()
    print(f"macro: 데이터셋 생성> {ticker_name}")


    X_train, X_test, y_train, y_test, X_seq, y_seq, feature_cols = macro_agent.make_dataset_seq(ticker_name)
    print(f"macro: 최종 학습 데이터셋 생성> {ticker_name}")



def macro_sercher(ticker_name):
    macro_agent = MacroSentimentAgentDataset()
    X_train, X_test, y_train, y_test, X_seq, y_seq, feature_cols = macro_agent.make_dataset_seq(ticker_name)
    X_tensor, stockdata = macro_agent.macro_searcher_add_funs(X_seq, feature_cols)
    print(f"■ macro_sercher StockData 생성 완료 ({ticker_name})")

    return X_tensor




