import pandas as pd
import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')

class SentimentalPredictor:
    """
    뉴스 감성 데이터를 기반으로 다음 날 주가를 예측하는 Predictor 에이전트입니다.
    """
    def __init__(self):
        # nltk 데이터 다운로드 확인 및 실행
        try:
            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')
        
        self.sentiment_analyzer = self.load_llm_model()

    def load_llm_model(self):
        """Hugging Face LLM 기반 감성 분석기 로드"""
        try:
            print("LLM 기반 감성 분석기 로딩 중...")
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
        except Exception as e:
            print(f"LLM 모델 로딩 또는 실행 오류: {e}. LLM 기반 감성 분석을 사용할 수 없습니다.")
            return None

    def get_llm_sentiment(self, text):
        """LLM 모델을 이용한 감성 점수 계산"""
        if self.sentiment_analyzer:
            try:
                # 모델 입력 길이를 512로 제한
                result = self.sentiment_analyzer(text[:512])
                if result[0]['label'] == 'POSITIVE':
                    return result[0]['score']
                else:
                    return -result[0]['score']
            except Exception as e:
                print(f"LLM 감성 분석 중 오류 발생: {e}")
                return 0
        return 0

    def get_keyword_sentiment(self, text):
        """키워드 기반 감성 점수 계산"""
        positive_words = ['growth', 'strong', 'increase', 'profit', 'gain', 'rise', 'up', 'boost', 'expand', 'win', 'success']
        negative_words = ['decline', 'loss', 'down', 'fail', 'drop', 'cut', 'reduce', 'miss', 'slump', 'crisis']
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        pos_score = sum(1 for word in filtered_words if word in positive_words)
        neg_score = sum(1 for word in filtered_words if word in negative_words)
        return pos_score - neg_score

    def prepare_data(self, news_file_path, stock_file_path):
        """
        뉴스 및 주가 데이터를 로드하고 전처리합니다.
        """
        try:
            news_df = pd.read_csv(news_file_path)
            stock_df = pd.read_csv(stock_file_path)
        except (FileNotFoundError, IndexError) as e:
            print(f"오류: 데이터 파일이 존재하지 않습니다. 먼저 데이터 수집 코드를 실행해주세요.")
            raise e

        # 뉴스 데이터 전처리
        news_df['date'] = pd.to_datetime(news_df['datetime'], unit='s').dt.date
        news_df['headline_summary'] = news_df['headline'].fillna('') + ' ' + news_df['summary'].fillna('')
        
        # 감성 점수 계산
        news_df['keyword_sentiment'] = news_df['headline_summary'].apply(self.get_keyword_sentiment)
        news_df['llm_sentiment'] = news_df['headline_summary'].apply(self.get_llm_sentiment)

        # 날짜 포맷 정리
        news_df['date'] = pd.to_datetime(news_df['date'])
        stock_df['date'] = pd.to_datetime(stock_df['date'])

        # 종가/타겟 생성
        stock_df['close'] = pd.to_numeric(stock_df['close'], errors='coerce')
        stock_df.dropna(subset=['close'], inplace=True)
        stock_df.sort_values(by=['symbol', 'date'], inplace=True)
        
        # 다음 날 종가(회귀 모델의 타겟)
        stock_df['next_day_close'] = stock_df.groupby('symbol')['close'].shift(-1)
        stock_df.dropna(subset=['next_day_close'], inplace=True)
        
        # 전일 뉴스 데이터 복사 (date +1)
        prev_news_df = news_df.copy()
        prev_news_df['date'] = prev_news_df['date'] + pd.Timedelta(days=1)
        news_all = pd.concat([news_df, prev_news_df], axis=0)

        # 종목 및 날짜별로 감성 점수 집계
        news_all = news_all.groupby(['symbol', 'date']).agg({
            'headline_summary': ' '.join,
            'keyword_sentiment': 'sum',
            'llm_sentiment': 'sum'
        }).reset_index()

        # 주가 데이터와 뉴스 감성 데이터 병합
        merged_df = pd.merge(
            stock_df, news_all,
            how='left',
            left_on=['symbol', 'date'],
            right_on=['symbol', 'date']
        )
        
        # 결측치 처리(뉴스가 없는 거래일은 0)
        for col in ['keyword_sentiment', 'llm_sentiment']:
            merged_df[col] = merged_df[col].fillna(0)

        return merged_df

    def predict(self, news_file_path, stock_file_path):
        """
        데이터를 로드, 전처리하고 회귀 모델로 다음 날 종가를 예측합니다.
        """
        try:
            data = self.prepare_data(news_file_path, stock_file_path)
        except Exception:
            return None, None, None

        features = ['keyword_sentiment', 'llm_sentiment']
        
        # 독립변수와 종속변수 분리
        X = data[features]
        y = data['next_day_close']

        # 학습 및 테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # RandomForestRegressor 모델 학습
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # XGBRegressor 모델 학습
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        
        # 두 모델의 예측값 평균
        y_pred_avg = (y_pred_rf + y_pred_xgb) / 2
        
        # 성능 평가
        r2 = r2_score(y_test, y_pred_avg)
        
        # 마지막 데이터를 이용해 다음 날 종가 예측
        last_data = data.iloc[-1][features].values.reshape(1, -1)
        next_day_price_rf = rf_model.predict(last_data)[0]
        next_day_price_xgb = xgb_model.predict(last_data)[0]
        predicted_price = float((next_day_price_rf + next_day_price_xgb) / 2)

        return predicted_price, r2, y_pred_avg

if __name__ == '__main__':
    try:
        # 가장 최근 뉴스 파일 찾기
        news_file = glob.glob('nasdaq100_stock_data*.csv')[-1]
        stock_file = 'nasdaq100_stock_data.csv'
        
        predictor = SentimentalPredictor()
        predicted_price, r2_score_value, _ = predictor.predict(news_file, stock_file)

        if predicted_price is not None:
            print("\n--- Sentimental Predictor 결과 ---")
            print(f"다음 날 예측 종가: ${predicted_price:.2f}")
            print(f"R² 스코어(모델 성능): {r2_score_value:.4f}")
            print("\n**참고: 이 코드는 회귀 모델을 사용해 종가를 예측하며, \n주가 변동성에 따라 R² 스코어가 낮을 수 있습니다.")
    except IndexError:
        print("\n오류: 데이터 파일이 존재하지 않습니다. 먼저 데이터 수집 코드를 실행해주세요.")
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
