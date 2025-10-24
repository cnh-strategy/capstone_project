import torch
import torch.nn as nn
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
import requests
from datetime import datetime, timedelta
import json
import time 
import os
import csv
import yfinance as yf
import pandas as pd

# EODhd API 설정 (더미 키)
API_KEY = 'YOUR_KEY' # 실제 API 키로 교체 필요
BASE_URL_EODHD = 'https://eodhd.com/api/news'
STATUS_FILE = 'collection_status.json' # 상태 파일명

from base_agent import BaseAgent 

# ==============================================================================
# 공통 데이터 포맷
# ==============================================================================

@dataclass
class Target:
    """예측 목표값 묶음 (필요 시 필드 확장)
    - next_close: 다음 거래일 종가 예측치
    - uncertainty: 예측의 불확실성 (몬테 카를로 분산)
    - confidence: 예측의 신뢰도 (불확률성 역수)
    - idea: 모델의 설명 가능성 근거 (SHAP, Feature Importance 등)
    """
    next_close : float
    uncertainty: float = 0.0 # 필드 추가
    confidence: float = 0.0  # 필드 추가
    idea: Dict[str, List[Any]] = field(default_factory=dict) # 필드 추가

@dataclass
class Opinion:
    """에이전트의 의견(초안/수정본 공통 포맷)
    - agent_id: 의견을 낸 에이전트 식별자
    - target  : 예측 타깃(예: next_close)
    - reason  : 근거 텍스트(LLM/룰 기반)
    """
    agent_id: str
    target: Target
    reason: str  # TODO: LLM/룰 기반 사유 텍스트 생성

@dataclass
class Rebuttal:
    """에이전트 간 반박/지지 메시지
    - from_agent_id: 보낸 쪽
    - to_agent_id  : 받는 쪽
    - stance       : REBUT(반박) | SUPPORT(지지)
    - message      : 근거 텍스트(간결 요약)
    """
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str  # TODO: LLM/룰 기반 한 줄 근거 생성

@dataclass
class RoundLog:
    """라운드별 기록 스냅샷(옵셔널로 사용)
    - round_no : 라운드 번호
    - opinions : 라운드 내 각 에이전트 최종 Opinion
    - rebuttals: 라운드 내 교환된 반박/지지
    - summary  : {"agent_id": Target(...)} 형태의 집계 요약
    """
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    """에이전트 입력 원천 데이터(필요 시 자유 확장)
    - sentimental: 심리/커뮤니티/뉴스 스냅샷
    - fundamental: 재무/밸류에이션 요약
    - technical  : 가격/지표 스냅샷
    - last_price : 최신 종가
    - currency   : 통화코드
    """
    sentimental: Dict
    fundamental: Dict
    technical: Dict
    last_price: Optional[float] = None
    currency: Optional[str] = None


# ==============================================================================
# SentimentalAgent 클래스 구현
# ==============================================================================
class SentimentalAgent(BaseAgent):
    def __init__(self, agent_id: str, input_features: int = 5):
        super().__init__(agent_id)
        
        # === [FinBERT 로드] ===
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_loaded = True
            print("FinBERT 모델 로드 완료")
        except ImportError:
            self.finbert_loaded = False
            print("경고: transformers 라이브러리가 없어 FinBERT 시뮬레이션 모드로 작동")
        # ================================
        
        self.input_features = input_features
        
        # 🌟 실제 모델 구조 정의 (임시 모델 대체)
        # 💡 참고: 모델 로드 시 동일한 구조를 정의해야 함
        self.model = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1) # 주가 변동률을 예측하는 출력 레이어
        )
        
        # 🌟 실제 모델 가중치 로드
        try:
            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH))
                self.model.eval() # 추론 모드 설정 (Dropout 유지)
                print(f"**실제 모델 가중치 로드 완료:** {MODEL_PATH}")
            else:
                print(f"경고: 모델 파일 '{MODEL_PATH}'이(가) 없어 임시 가중치를 사용합니다.")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}. 임시 가중치를 사용합니다.")
            
# MODEL_PATH를 SentimentalAgent 클래스 외부에 정의하거나 클래스 상수로 정의해야 함
MODEL_PATH = 'sentimental_model.pth'

    # ==========================================================================
    # 🌟 기존 파이프라인 코드 통합
    # ==========================================================================

    # load_status 함수
    def load_status(self):
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        return {'completed_symbols': []}

    # save_status 함수
    def save_status(self, status):
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=4)

    # collect_news_data_eodhd 함수 (배치 수집용)
    def collect_news_data_eodhd_batch(self, ticker, from_date, to_date):
        """
        EOD Historical Data API를 사용하여 특정 기간의 뉴스 데이터를 수집（감성 점수 포함）
        """
        all_news = []
        offset = 0
        limit = 1000 

        while True:
            params = {
                's': ticker,
                'from': from_date,
                'to': to_date,
                'api_token': API_KEY,
                'limit': limit,
                'offset': offset,
                'extended': 1 # 감성 점수 포함해서 저장 (FinBERT 학습용 데이터)
            }

            try:
                response = requests.get(BASE_URL_EODHD, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"[{ticker}] API 호출 중 네트워크 오류/타임아웃 발생: {e}")
                return all_news, offset
                
            if response.status_code == 200:
                news_list = response.json()
                if not news_list:
                    print(f"[{ticker}] 더 이상 뉴스 데이터가 없습니다.")
                    break 

                for news in news_list:
                    data = {
                        'date': news.get('date', ''),
                        'title': news.get('title', ''),
                        'summary': news.get('content', ''), 
                        'related': news.get('symbols', ticker), 
                        'ticker': ticker,
                        'sentiment_score': news.get('sentiment', '')
                    }
                    all_news.append(data)

                if len(news_list) < limit:
                    break 
                else:
                    offset += limit
                    time.sleep(1)
                    
            else:
                print(f"[{ticker}] API 호출 오류 {response.status_code} - {response.text}")
                print(f"[{ticker}] 오프셋 {offset}에서 수집 중단.")
                return all_news, offset
                
        return all_news, -1 

    # save_news_to_csv 함수
    def save_news_to_csv(self, news_data, filename, mode='a'):
        fieldnames=['date', 'title', 'summary', 'related', 'ticker', 'sentiment_score']
        file_exists = os.path.exists(filename) and mode == 'a' 
        
        with open(filename, mode=mode, newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if mode == 'w' or not file_exists:
                writer.writeheader()
            
            for record in news_data:
                writer.writerow(record)
        print(f"데이터 {len(news_data)}개를 {filename} 파일에 저장 완료")

    # 메인 실행 블록
    def collect_historical_data(self, tickers: List[str]):
        """
        5년치 뉴스 및 주가 데이터를 수집하고 CSV 파일로 저장
        (모델 훈련을 위한 일회성/배치 수집 메서드)
        """
        from_date = '2020-01-01'
        to_date_news = '2024-12-31'
        to_date_stock = '2025-01-01'

        print(f"**데이터 수집 기간:** {from_date} 부터 {to_date_news} 까지")

        NEWS_FILE = "news_data.csv"
        STOCK_FILE = "stock_data.csv"

        # 1. 기존 파일 삭제 (완전 초기화)
        if os.path.exists(NEWS_FILE):
            os.remove(NEWS_FILE)
            print(f"기존 {NEWS_FILE} 파일을 삭제했습니다.")
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)
            print(f"기존 {STATUS_FILE} 파일을 삭제했습니다.")
            
        # 2. 뉴스 데이터 수집 및 저장
        print("\n--- 뉴스 데이터 수집 시작 (EODhd, 감성 점수 포함) ---")
        is_first_symbol = True
        news_collection_successful = True

        for ticker in tickers:
            print(f"[{ticker}] 뉴스 수집 시작...")
            
            collected_news, last_offset = self.collect_news_data_eodhd_batch(ticker, from_date, to_date_news)
            
            if collected_news:
                save_mode = 'w' if is_first_symbol else 'a'
                self.save_news_to_csv(collected_news, NEWS_FILE, mode=save_mode)
                is_first_symbol = False
                
            if last_offset != -1:
                news_collection_successful = False
                print(f"[{ticker}] 수집이 중간에 중단되었습니다.")
                break
                
            print(f"[{ticker}] 뉴스 수집 완료.")

        # 3. 주가 데이터 수집 및 저장 
        print("\n--- 주가 데이터 수집 시작 ---")
        if news_collection_successful:
            all_stock_data = []
            for ticker in tickers:
                print(f"{ticker} 주가 데이터 수집 중 (yfinance)...")
                df = yf.download(ticker, start=from_date, end=to_date_stock)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df.reset_index()
                df['Symbol'] = ticker
                df = df[['Symbol', 'Date', 'Open', 'Close']]
                all_stock_data.append(df)
                
            result = pd.concat(all_stock_data, ignore_index=True)
            result.to_csv(STOCK_FILE, index=False, encoding='utf-8')
            print("stock_data.csv 파일 저장 완료")
        else:
            print("뉴스 수집이 완료되지 않아 주가 데이터 수집은 건너뜁니다.")


    # ==========================================================================
    # 실시간 예측
    # ==========================================================================

    # --------------------------------------------------------------------------
    # 내부 헬퍼 함수: FinBERT 분석 시뮬레이션
    # --------------------------------------------------------------------------
    def _simulate_finbert_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """
        FinBERT 모델을 사용하여 텍스트 목록을 분석하고 모델 입력 특징 벡터를 생성하는
        과정을 시뮬레이션

        실제 FinBERT 구현 시, 텍스트를 토크나이징하고 모델에 넣어 벡터(임베딩)를 추출
        """
        if not self.finbert_loaded:
            return {"avg_sentiment": 0.0, "vector": [0.5] * self.input_features}
            
        # 텍스트 분석 및 점수 집계 시뮬레이션
        # 여기서는 각 텍스트가 [긍정, 중립, 부정]의 확률을 반환한다고 가정
        
        # 1. 텍스트 분석 및 점수 집계
        total_score = 0
        score_count = 0
        
        for text in texts:
            # FinBERT 분석 과정 시뮬레이션
            # 긍정/중립/부정 확률을 시뮬레이션 (예: 3차원 벡터)
            pos = random.uniform(0, 1)
            neg = random.uniform(0, 1)
            neu = random.uniform(0, 1)
            
            # 정규화 (FinBERT는 보통 softmax 출력)
            total = pos + neg + neu
            if total > 0:
                pos, neg, neu = pos / total, neg / total, neu / total
            
            # 최종 감성 점수 계산 (예: 긍정 - 부정)
            current_score = pos - neg 
            total_score += current_score
            score_count += 1
            
        avg_sentiment = total_score / score_count if score_count > 0 else 0.0

        # 2. 모델 입력 벡터 생성 (예시: 평균 감성 점수와 몇 가지 추가 특징)
        # 5차원 입력 특징이라고 가정하고, FinBERT 임베딩 결과 3차원 + 추가 특징 2차원을 결합한다고 시뮬레이션
        # 여기서는 간단히 avg_sentiment를 복사한 벡터로 대체
        sentiment_vector = [avg_sentiment] * self.input_features

        return {
            "avg_sentiment": avg_sentiment,
            "vector": sentiment_vector
        }
    
    # --------------------------------------------------------------------------
    # 실시간 감성 데이터 수집 로직 (searcher를 위한 최신 데이터 호출)
    # --------------------------------------------------------------------------
    def _fetch_latest_sentiment_data(self, ticker: str) -> Dict[str, Any]:
        """
        특정 종목의 최신(최근 1일) 뉴스 텍스트를 API로 호출하고
        FinBERT 시뮬레이션을 통해 모델 입력 벡터를 도출
        """
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        params = {
            's': ticker,
            'from': yesterday,
            'to': today,
            'api_token': API_KEY, # 실제 키로 교체 필요
            'limit': 10, # 최신 10개 기사만
            'extended': 0 # 텍스트만 필요하므로 extended=0 (또는 생략)
        }
        
        news_texts = []
        key_topics = set()

        try:
            response = requests.get(BASE_URL_EODHD, params=params, timeout=10)
            response.raise_for_status() 
            news_list = response.json()
            
            for news in news_list:
                title = news.get('title', '')
                content = news.get('content', '')
                
                # FinBERT 분석을 위해 텍스트 수집
                if title:
                    news_texts.append(title)
                    key_topics.add(title)

                # 요약된 FinBERT 임베딩 결과 (vector)를 사용
            if news_texts:
                finbert_result = self._simulate_finbert_analysis(news_texts)
                avg_sentiment = finbert_result['avg_sentiment']
                sentiment_vector = finbert_result['vector']
            else:
                avg_sentiment = 0.0
                sentiment_vector = [0.5] * self.input_features
                
            return {
                "news_sentiment_score": avg_sentiment, # FinBERT 분석 결과
                "community_score": random.uniform(-0.5, 0.5), # 여전히 더미
                "sentiment_vector": sentiment_vector,
                "key_topics": list(key_topics)[:3] 
            }

        except requests.exceptions.RequestException as e:
            print(f"[{ticker}] API 호출 오류 발생: {e}. 더미 데이터 사용.")
            return {
                "news_sentiment_score": 0.0,
                "community_score": 0.0,
                "sentiment_vector": [0.5] * self.input_features, 
                "key_topics": [f"API 오류: {e}"]
            }
        except Exception as e:
            print(f"[{ticker}] 데이터 처리 중 오류 발생: {e}. 더미 데이터 사용.")
            return {
                "news_sentiment_score": 0.0,
                "community_score": 0.0,
                "sentiment_vector": [0.5] * self.input_features, 
                "key_topics": ["데이터 처리 오류 발생"]
            }

    # --------------------------------------------------------------------------
    # forward 메서드 구현, dropout layer 포함
    # --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파를 정의하며, Dropout Layer를 포함합니다.
        """
        return self.model(x)

    # --------------------------------------------------------------------------
    # agent.searcher(): predict로 내일 종가를 예측할때 입력값을 호출하는 메서드
    # --------------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        """
        [필수 메서드] 특정 종목(ticker)의 감성 분석 예측에 필요한 최신 입력 데이터를 검색하고 반환
        """
        sentimental_data = self._fetch_latest_sentiment_data(ticker)
        
        # 마지막 종가는 yfinance 등으로 별도 호출이 필요하지만, 여기서는 더미값 사용
        last_price = 50000.0 * (1 + random.uniform(-0.01, 0.01)) 

        data = StockData(
            sentimental=sentimental_data,
            fundamental={},
            technical={},
            last_price=last_price,
            currency="KRW"
        )
        return data

    # --------------------------------------------------------------------------
    # agent.predictor(): 내일 종가를 Target 클래스로 반환하는 메서드, 몬테 카를로 예측 포함
    # --------------------------------------------------------------------------
    def predictor(self, ticker: str) -> Target:
        """
        [필수 메서드] 입력값으로 내일의 종가를 예측하고 Target 클래스 인스턴스로 반환
        몬테 카를로 예측 포함
        """
        # 1. 입력 데이터 검색 (agent.searcher() 호출)
        stock_data = self.searcher(ticker)
        last_price = stock_data.last_price or 50000.0

        # 2. 모델 입력 준비
        sentiment_vector = stock_data.sentimental["sentiment_vector"]
        input_data = torch.tensor([sentiment_vector], dtype=torch.float32)
        
        # 3. 몬테 카를로 드롭아웃 (Monte Carlo Dropout) 시뮬레이션
        num_samples = 150
        predictions_raw = [] 

        for _ in range(num_samples):
            with torch.no_grad():
                price_change_rate = self.forward(input_data).item() 
                predictions_raw.append(price_change_rate)

        predictions_np = np.array(predictions_raw)
        predicted_prices = last_price * (1 + predictions_np)

        # 4. Target 클래스 필드 계산
        next_close = float(np.mean(predicted_prices))
        uncertainty = float(np.std(predicted_prices))
        confidence = float(1.0 / (1.0 + uncertainty * 10))
        confidence = min(1.0, confidence)

        # 피쳐중요도 추가
        ## 첫 번째 선형 레이어의 가중치 가져오기 (특징 수: self.input_features)
        weights = self.model[0].weight.data.numpy() # (32, 5) 형태 가정

        ## 각 입력 피처에 대한 평균 절댓값 가중치 계산 (컬럼별 평균)
        ## 이 값을 피처 중요도로 사용 (간단한 모델에서 일반적)
        feature_importances_np = np.mean(np.abs(weights), axis=0)
        feature_importances = [float(f) for f in feature_importances_np]

        # shap 추가
        import shap
        try:
            # SHAP Explainer 정의 (딥러닝 모델에는 DeepExplainer 사용)
            # 💡 훈련된 데이터 샘플을 제공해야 정확도가 높음
            # 현재는 모델 정의가 nn.Sequential이므로, 훈련된 데이터 일부를 배경(background)으로 사용한다고 가정
            
            # 더미 배경 데이터 (실제로는 훈련 데이터의 일부여야 함)
            background_data = torch.randn(10, self.input_features) 
            explainer = shap.DeepExplainer(self.model, background_data)
            
            # 2. 현재 입력 데이터에 대한 SHAP 값 계산
            shap_values_np = explainer.shap_values(input_data)[0] # (1, 5) 형태
            shap_values = [float(s) for s in shap_values_np[0]]
            
        except Exception as e:
        # shap 라이브러리가 없거나 오류 발생 시
        shap_values = []
        print(f"SHAP 계산 중 오류 발생: {e}")

        # 5. Target 클래스 인스턴스 반환
        result = Target(
            next_close=next_close,
            uncertainty=uncertainty,
            confidence=confidence,
            idea={
                "sentiment_score": [stock_data.sentimental['news_sentiment_score']], 
                "feature_names": [f"feature_{i+1}" for i in range(self.input_features)],
                "related_news_summary": stock_data.sentimental["key_topics"],
                "mc_price_samples": [float(p) for p in predicted_prices[:5]],
                "feature_importances": feature_importances,
                "shap_values": shap_values
            }
        )
        
        return result

# ==============================================================================
# 에이전트 사용 예시
# ==============================================================================
if __name__ == '__main__':
    print("="*50)
    print("## SentimentalAgent 테스트 실행")
    print("="*50)
    
    # ----------------------------------------------------------------------
    # 🌟 데이터 파이프라인 테스트: 원본 코드의 기능을 실행
    # ----------------------------------------------------------------------
    agent = SentimentalAgent(agent_id="SentimentalAgent")
    tickers_for_collection = ['NVDA', 'MSFT', 'AAPL']
    
    # 이 메서드를 호출하면 5년치 데이터를 수집하고 news_data.csv 및 stock_data.csv를 생성
    # agent.collect_historical_data(tickers=tickers_for_collection) 
    # print("\n[Historical Data Pipeline] 실행 완료. CSV 파일 확인 필요.")
    # print("-" * 50)


    # ----------------------------------------------------------------------
    # 🚀 필수 메서드 (searcher/predictor) 테스트
    # ----------------------------------------------------------------------
    TEST_TICKER = "MSFT"

    # 예측 메서드 호출 (ticker 인자 전달) - agent.predictor() 사용
    agent_for_inference = SentimentalAgent(agent_id="SentimentalAgent")
    target_result = agent_for_inference.predictor(ticker=TEST_TICKER)
    
    # searcher() 테스트 데이터
    test_stock_data = agent_for_inference.searcher(TEST_TICKER) 

    print(f"## 최종 감성 예측 결과 ({TEST_TICKER}) (Target 클래스 출력 형태):")
    print(f"최신 종가 (가정): {test_stock_data.last_price:.2f} {test_stock_data.currency}")
    print(f"news_sentiment_score (FinBERT 시뮬레이션 결과): {test_stock_data.sentimental['news_sentiment_score']:.2f}")
    print(f"next_close (예측 종가): {target_result.next_close:.2f}")
    print(f"uncertainty (불확실성): {target_result.uncertainty:.4f}")
    print(f"confidence (신뢰도): {target_result.confidence:.4f}")
    print("\nidea (설명 가능성):")
    for key, value in target_result.idea.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"  - {key}: {value[:3]}...")
        else:
            print(f"  - {key}: {value}")
    print("="*50)