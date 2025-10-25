class DataManager:
    def __init__(self, data_dir="data"):  
        # 데이터 경로 초기화, 폴더 생성
    
    def fetch(self, ticker):  
        # Yahoo Finance 등에서 티커 데이터 다운로드 (raw/)
    
    def preprocess(self, ticker):  
        # CSV 로드 → 수익률, 이동평균 등 feature 생성 (processed/)