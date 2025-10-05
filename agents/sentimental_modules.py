"""
Sentimental Agent의 모듈화된 Searcher와 Predictor
메인 브랜치의 SentimentalAgent에 선택적으로 통합 가능
"""

import os
import time
import requests
import csv
import pandas as pd
import torch
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import yfinance as yf

class SentimentalSearcher:
    """뉴스 데이터 수집 모듈"""
    
    def __init__(self, api_key: Optional[str] = None, max_calls_per_minute: int = 60):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        self.base_url = 'https://finnhub.io/api/v1/company-news'
        self.max_calls_per_minute = max_calls_per_minute
        self.call_count = 0
        
    def safe_convert_timestamp(self, timestamp) -> str:
        """안전한 타임스탬프 변환"""
        try:
            if timestamp is None or not isinstance(timestamp, (int, float)) or timestamp <= 0:
                return ''
            if timestamp > 32503680000:  # 3000-01-01 00:00:00 UTC 제한
                return ''
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ''
    
    def collect_news_data(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """특정 종목의 뉴스 데이터 수집"""
        if not self.api_key:
            print("⚠️ FINNHUB_API_KEY가 설정되지 않았습니다. 기본 뉴스 수집을 사용합니다.")
            return self._get_fallback_news(symbol)
            
        all_news = []
        current_from = datetime.strptime(from_date, "%Y-%m-%d")
        current_to = datetime.strptime(to_date, "%Y-%m-%d")
        
        while current_from <= current_to:
            from_str = current_from.strftime("%Y-%m-%d")
            to_str = (current_from + timedelta(days=7)).strftime("%Y-%m-%d")
            if current_from + timedelta(days=7) > current_to:
                to_str = to_date
            
            params = {
                'symbol': symbol,
                'from': from_str,
                'to': to_str,
                'token': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=20)
            self.call_count += 1
            
            if response.status_code == 200:
                news_list = response.json()
                for news in news_list:
                    timestamp = news.get('datetime')
                    readable_date = self.safe_convert_timestamp(timestamp)
                    
                    data = {
                        'date': readable_date,
                        'title': news.get('headline', ''),
                        'summary': news.get('summary', ''),
                        'related': news.get('related', symbol)
                    }
                    all_news.append(data)
            else:
                print(f"API 호출 오류 {response.status_code} - {response.text}")
            
            # API 호출 제한 관리
            if self.call_count >= self.max_calls_per_minute:
                print("60회 호출 도달, 60초 대기 중...")
                time.sleep(60)
                self.call_count = 0
            
            current_from += timedelta(days=7)
        
        return all_news
    
    def _get_fallback_news(self, symbol: str) -> List[Dict]:
        """API 키가 없을 때 사용할 기본 뉴스 데이터"""
        return [{
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'title': f"{symbol} 관련 뉴스",
            'summary': f"{symbol} 종목에 대한 최신 시장 동향 분석이 필요합니다.",
            'related': symbol
        }]
    
    def get_recent_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """최근 N일간의 뉴스 데이터 수집"""
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.collect_news_data(symbol, from_date, to_date)


class SentimentalPredictor:
    """ML 기반 센티멘탈 예측 모듈"""
    
    def __init__(self, model_path: Optional[str] = None, use_finbert: bool = True):
        self.model_path = model_path or "mlp_stock_model.pt"
        self.use_finbert = use_finbert
        self.model = None
        self.tokenizer = None
        self.finbert_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.use_finbert:
            self._load_finbert()
        
        self._load_model()
    
    def _load_finbert(self):
        """FINBERT 모델 로드"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            self.finbert_model = AutoModel.from_pretrained('yiyanghkust/finbert-tone')
            print("✅ FINBERT 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ FINBERT 로드 실패: {e}")
            self.use_finbert = False
    
    def _load_model(self):
        """훈련된 MLP 모델 로드"""
        try:
            if os.path.exists(self.model_path):
                # 모델 구조 정의 (sentimental_predictor_3ent.py와 동일)
                class MLPRegressor(torch.nn.Module):
                    def __init__(self, input_dim):
                        super().__init__()
                        self.net = torch.nn.Sequential(
                            torch.nn.Linear(input_dim, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, 1)
                        )
                    def forward(self, x):
                        return self.net(x).squeeze()
                
                self.model = MLPRegressor(input_dim=768).to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                print("✅ MLP 모델 로드 완료")
            else:
                print(f"⚠️ 모델 파일을 찾을 수 없습니다: {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            self.model = None
    
    def get_finbert_embedding(self, text: str) -> np.ndarray:
        """FINBERT를 사용한 텍스트 임베딩"""
        if not self.use_finbert or not self.tokenizer or not self.finbert_model:
            # 기본 임베딩 (768차원 0으로 채움)
            return np.zeros(768)
        
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        except Exception as e:
            print(f"⚠️ FINBERT 임베딩 실패: {e}")
            return np.zeros(768)
    
    def predict_log_return(self, news_texts: List[str]) -> float:
        """뉴스 텍스트들로부터 로그 수익률 예측"""
        if not self.model:
            print("⚠️ 모델이 로드되지 않았습니다. 기본값 0.0 반환")
            return 0.0
        
        try:
            # 모든 뉴스 텍스트를 하나로 합치기
            combined_text = " ".join(news_texts)
            
            # FINBERT 임베딩
            embedding = self.get_finbert_embedding(combined_text)
            
            # 모델 예측
            with torch.no_grad():
                input_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                prediction = self.model(input_tensor)
                log_return = prediction.cpu().item()
            
            return log_return
        except Exception as e:
            print(f"⚠️ 예측 실패: {e}")
            return 0.0
    
    def predict_next_close(self, current_price: float, news_texts: List[str]) -> float:
        """현재가와 뉴스를 바탕으로 다음 종가 예측"""
        log_return = self.predict_log_return(news_texts)
        predicted_close = current_price * np.exp(log_return)
        return predicted_close


class SentimentalModuleManager:
    """센티멘탈 모듈 통합 관리자"""
    
    def __init__(self, 
                 use_ml_searcher: bool = False,
                 use_ml_predictor: bool = False,
                 finnhub_api_key: Optional[str] = None,
                 model_path: Optional[str] = None):
        
        self.use_ml_searcher = use_ml_searcher
        self.use_ml_predictor = use_ml_predictor
        
        # 모듈 초기화
        if self.use_ml_searcher:
            self.searcher = SentimentalSearcher(api_key=finnhub_api_key)
        else:
            self.searcher = None
            
        if self.use_ml_predictor:
            self.predictor = SentimentalPredictor(model_path=model_path)
        else:
            self.predictor = None
    
    def get_enhanced_sentimental_data(self, ticker: str, current_price: float) -> Dict:
        """ML 모듈을 활용한 향상된 센티멘탈 데이터 생성"""
        result = {
            "sentiment": "neutral",
            "positives": [],
            "negatives": [],
            "evidence": [],
            "summary": "",
            "ml_prediction": None,
            "ml_confidence": 0.0
        }
        
        # ML Searcher 사용
        if self.use_ml_searcher and self.searcher:
            try:
                news_data = self.searcher.get_recent_news(ticker, days=7)
                news_texts = [f"{news['title']} {news['summary']}" for news in news_data]
                
                # 뉴스 데이터를 센티멘탈 분석에 활용
                result["evidence"] = [news['title'] for news in news_data[:5]]  # 최근 5개 뉴스
                result["summary"] = f"최근 {len(news_data)}개의 뉴스가 수집되었습니다."
                
                # ML Predictor 사용
                if self.use_ml_predictor and self.predictor:
                    try:
                        predicted_close = self.predictor.predict_next_close(current_price, news_texts)
                        log_return = np.log(predicted_close / current_price)
                        
                        result["ml_prediction"] = predicted_close
                        result["ml_confidence"] = min(abs(log_return) * 10, 1.0)  # 0-1 범위로 정규화
                        
                        # 로그 수익률에 따른 센티멘탈 분류
                        if log_return > 0.02:  # 2% 이상 상승 예측
                            result["sentiment"] = "positive"
                            result["positives"] = ["ML 모델이 상승을 예측"]
                        elif log_return < -0.02:  # 2% 이상 하락 예측
                            result["sentiment"] = "negative"
                            result["negatives"] = ["ML 모델이 하락을 예측"]
                        else:
                            result["sentiment"] = "neutral"
                            
                    except Exception as e:
                        print(f"⚠️ ML 예측 실패: {e}")
                        
            except Exception as e:
                print(f"⚠️ ML 뉴스 수집 실패: {e}")
        
        return result


# 사용 예제
if __name__ == "__main__":
    # 모듈 매니저 초기화
    manager = SentimentalModuleManager(
        use_ml_searcher=True,
        use_ml_predictor=True,
        finnhub_api_key=os.getenv('FINNHUB_API_KEY'),
        model_path="mlp_stock_model.pt"
    )
    
    # 테스트
    ticker = "AAPL"
    current_price = 150.0
    
    enhanced_data = manager.get_enhanced_sentimental_data(ticker, current_price)
    print("향상된 센티멘탈 데이터:", enhanced_data)
