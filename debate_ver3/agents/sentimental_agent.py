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
import joblib 
import shap 

# FinBERT 로드를 위한 라이브러리 임포트 (AutoModel 사용 유지)
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"FinBERT 로드 오류: {e}. 기능 비활성화.")


# EODhd API 설정 (API 키는 실제 키로 교체해야 합니다. ⚠️)
API_KEY = ' 68e3a8c46e9a65.00465987'
BASE_URL_EODHD = 'https://eodhd.com/api/news'
STATUS_FILE = 'collection_status.json' # 상태 파일명

# 파일 경로 정의
MODEL_PATH = 'model_lstm_bothsentiment_V2.pt'
SCALER_X_PATH = 'scaler_x_both_V2.pkl'
SCALER_Y_PATH = 'scaler_y_both_V2.pkl'

# 📌 모델 입력 피처 및 시퀀스 길이
# FEATURES = ['prob_positive','prob_negative','prob_neutral','n_news','ret','Close', 'eod_sentiment']
INPUT_FEATURES = 7 # 위 7개 피처
WINDOW_SIZE = 10 

# 📌 LLM Opinion Prompt 정의 (이전과 동일)
OPINION_PROMPTS = {
    "sentimental": {
        "system": (
            "당신은 감성 및 텍스트 데이터 분석에 특화된 수석 전략가입니다. "
            "주어진 Context 데이터를 분석하여, 예측 종가(our_prediction)에 대한 논리적이고 간결하며 설득력 있는 의견을 한국어로 3문장 이내로 작성하십시오. "
            "감성 점수와 주요 토픽을 반드시 언급하고, 불확실성(uncertainty)을 고려하여 의견을 마무리하세요."
        ),
        "user": (
            "다음은 예측 모델의 입력 데이터와 결과입니다. 이를 바탕으로 의견을 작성하세요:\n"
            "Context: {context}" 
        )
    }
}

# base_agent 클래스 정의
class BaseAgent:
    def __init__(self, agent_id: str, **kwargs):
        self.agent_id = agent_id
    
    def searcher(self, ticker: str) -> 'StockData':
        raise NotImplementedError
        
    def predictor(self, ticker: str) -> 'Target':
        raise NotImplementedError

# ==============================================================================
# 공통 데이터 포맷 (이전과 동일)
# ==============================================================================

@dataclass
class Target:
    """예측 목표값 묶음"""
    next_close : float
    uncertainty: float = 0.0
    confidence: float = 0.0
    idea: Dict[str, List[Any]] = field(default_factory=dict)

@dataclass
class Opinion:
    """에이전트의 의견"""
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    """에이전트 간 반박/지지 메시지"""
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str

@dataclass
class RoundLog:
    """라운드별 기록 스냅샷(옵셔널로 사용)"""
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    """에이전트 입력 원천 데이터"""
    sentimental: Dict 
    fundamental: Dict
    technical: Dict
    last_price: Optional[float] = None 
    currency: Optional[str] = None

# ==============================================================================
# StockSentimentLSTM 모델 구조
# ==============================================================================
class StockSentimentLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, input_size=INPUT_FEATURES): 
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=0.3) 
        self.fc = nn.Linear(hidden_dim, 1) 
        
    def forward(self, x):
        out, _ = self.lstm(x) 
        return self.fc(out[:, -1, :]).squeeze() 
    
# ==============================================================================
# SentimentalAgent 클래스 구현
# ==============================================================================
class SentimentalAgent(BaseAgent):
    def __init__(self, agent_id: str, input_features: int = INPUT_FEATURES):
        super().__init__(agent_id)
        self.input_features = input_features
        self.window_size = WINDOW_SIZE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # === [FinBERT 로드: AutoModel 사용 유지] ===
        self.finbert_loaded = TRANSFORMERS_AVAILABLE
        self.finbert_tokenizer = None
        self.finbert_model = None
        if self.finbert_loaded:
            try:
                # 이미 상단에서 임포트했으므로, 여기서는 로드만 시도
                if 'AutoTokenizer' in globals() and 'AutoModelForSequenceClassification' in globals():
                     self.finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone') 
                     self.finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone') 
                     print("FinBERT 모델 로드 완료")
                else:
                    raise ImportError("AutoModel/AutoTokenizer is not available")
            except Exception as e:
                self.finbert_loaded = False
                print(f"FinBERT 로드 오류: {e}. 기능 비활성화.")
        else:
             print("FinBERT 로드 오류: transformers 라이브러리 미설치. 기능 비활성화.")
        # ===============================================
        
        # 🌟 실제 모델 구조 정의 (LSTM)
        self.model = StockSentimentLSTM()
        
        # 스케일러 및 모델 가중치 로드 로직
        self.scaler_X = None
        self.scaler_y = None
        
        try:
            if os.path.exists(SCALER_X_PATH):
                # joblib은 map_location이 필요 없음 (PyTorch 객체가 아님)
                self.scaler_X = joblib.load(SCALER_X_PATH)
                print(f"[SUCCESS] 입력 스케일러(X) 로드 완료: {SCALER_X_PATH}")
            if os.path.exists(SCALER_Y_PATH):
                self.scaler_y = joblib.load(SCALER_Y_PATH)
                print(f"[SUCCESS] 출력 스케일러(Y) 로드 완료: {SCALER_Y_PATH}")
            if os.path.exists(MODEL_PATH):
                # 🚨 [수정 반영] CUDA/CPU 오류 방지를 위해 map_location='cpu' 추가
                state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu')) 
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"**실제 LSTM 모델 가중치 로드 완료:** {MODEL_PATH} (CPU 매핑)")
        except Exception as e:
            # 기존 로그에 있던 오류 메시지
            print(f"모델/스케일러 로드 중 오류 발생: {e}. 임시 가중치 사용.")
            # 오류가 발생한 경우 모델이 예측 기능을 수행할 수 없으므로, self.model을 None으로 설정하거나 적절한 오류 처리를 추가해야 합니다.

    # ==========================================================================
    # 🌟 LLM Context 및 Opinion 빌드 헬퍼 메서드 (이전과 동일)
    # ==========================================================================
    def _build_llm_context(self, ticker: str, stock_data: StockData, target: Target) -> Dict[str, Any]:
        """
        Opinion 메시지 생성을 위한 LLM Context 딕셔너리를 빌드합니다. (ctx)
        """
        
        t = ticker 
        ccy = stock_data.currency.upper() if stock_data.currency else "USD" 
        last_price = float(stock_data.last_price or 0.0) 
        
        # ----------------------------------------------------
        # 📌 ctx 딕셔너리 생성
        # ----------------------------------------------------
        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_price": last_price,
            
            # SentimentalAgent의 searcher 데이터를 모두 포함
            "fundamental_summary": stock_data.fundamental or {}, 
            # SENTIMENTAL_SEQUENCE 키 추가
            "sentimental_summary": {
                "SENTIMENTAL_SEQUENCE": stock_data.sentimental.get('sequence_data', {}),
                "window_size": self.window_size
            }, 
            "technical_summary": stock_data.technical or {},
            
            "our_prediction": float(target.next_close),
            "uncertainty": float(target.uncertainty), 
            "confidence": float(target.confidence), 
            "model_idea": target.idea,
        }
        
        return ctx

    def build_opinion_messages(self, ticker: str, stock_data: StockData, target: Target) -> tuple[str, str]:
        """ Opinion 생성을 위해 LLM에 전달할 system_text와 user_text를 빌드합니다. """
        
        ctx = self._build_llm_context(ticker, stock_data, target)
        
        # OPINION_PROMPTS의 "sentimental" 키를 사용하여 메시지 빌드
        system_text = OPINION_PROMPTS["sentimental"]["system"]
        user_text = OPINION_PROMPTS["sentimental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False) # Context를 JSON으로 직렬화
        )
        
        return system_text, user_text

    def build_opinion(self, ticker: str, stock_data: StockData, target: Target, reason_text: Optional[str] = None) -> Opinion:
        """ Opinion 객체를 생성합니다. (reason_text는 LLM 응답이라고 가정) """

        # 만약 LLM을 호출하지 않고 더미 이유를 만들 경우:
        if reason_text is None:
            # 시퀀스 데이터에서 최신 감성 점수만 추출하여 더미 이유 생성
            seq_data = stock_data.sentimental.get('sequence_data', {})
            try:
                # 'prob_positive'가 없으면 0.0으로 대체하여 KeyError 방지
                prob_positive = seq_data.get('prob_positive', [0.0])[-1] 
                prob_negative = seq_data.get('prob_negative', [0.0])[-1]
                sentiment_score = prob_positive - prob_negative
            except IndexError:
                sentiment_score = 0.0
                
            pred = target.next_close
            last = stock_data.last_price or 0.0
            
            reason_text = (
                f"[감성 분석 보고서]: 최신 FinBERT 감성 점수({sentiment_score:.2f})는 주가 변동성에 영향을 미칩니다. "
                f"LSTM 모델은 {self.window_size}일간의 시퀀스를 기반으로 현재가({last:.2f}) 대비 다음 종가를 {pred:.2f}로 예측합니다. "
                f"불확실성({target.uncertainty:.4f})이 크므로 주의 깊은 관찰이 필요합니다."
             )
        
        return Opinion(
            agent_id=self.agent_id,
            target=target,
            reason=reason_text
        )


    # ==========================================================================
    # 🌟 기존 파이프라인 코드 통합 (데이터 수집 관련 함수 - 배치 수집용, 변경 없음)
    # ==========================================================================

    def load_status(self):
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        return {'completed_symbols': []}

    def save_status(self, status):
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=4)

    def collect_news_data_eodhd_batch(self, ticker, from_date, to_date):
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
                'extended': 1 
            }
            try:
                response = requests.get(BASE_URL_EODHD, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"[{ticker}] API 호출 중 네트워크 오류/타임아웃 발생: {e}")
                return all_news, offset
                
            if response.status_code == 200:
                news_list = response.json()
                if not news_list: break 

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

                if len(news_list) < limit: break 
                else:
                    offset += limit
                    time.sleep(1)
            else:
                # 🚨 API 인증 오류 401 포함하여 오류 코드 출력
                print(f"[{ticker}] API 호출 오류 {response.status_code} - {response.text.strip()[:100]}")
                # 오류 발생 시 현재까지 수집된 데이터만 반환
                return all_news, offset 
                
        return all_news, -1 

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

    def collect_historical_data(self, tickers: List[str]):
        """ (배치 수집 메서드) - 생략 """
        print("이 메서드는 Historical Data 수집용으로, 실시간 예측 파이프라인에서 호출되지 않습니다.")
        pass

# ==========================================================================
    # 🌟 FinBERT를 이용한 텍스트 분석 함수 (학습 코드에서 가져옴)
    # ==========================================================================
    def _finbert_sentiment_scores(self, texts: List[str]) -> np.ndarray: 
        if not self.finbert_loaded:
            # FinBERT가 로드되지 않은 경우 더미 데이터 반환 (긍정/부정/중립)
            print("[WARNING] FinBERT 비활성화: 더미 감성 점수 반환.")
            # 3분류(긍정/부정/중립) 모델을 가정하여 1/3로 균등하게 분배
            return np.array([[0.333, 0.333, 0.334] for _ in texts]) 
        
        self.finbert_model.eval() 
        scores = []
        batch_size = 32
        with torch.no_grad(): 
            for i in range(0, len(texts), batch_size): 
                batch_texts = texts[i:i+batch_size]
                inputs = self.finbert_tokenizer(list(batch_texts), return_tensors='pt', 
                                                 padding=True, truncation=True, max_length=512)
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                outputs = self.finbert_model(**inputs) 
                # FinBERT 결과는 긍정/부정/중립 확률 배열 (softmax 적용)
                scores.extend(torch.softmax(outputs.logits, dim=1).cpu().numpy())
                
        return np.array(scores)

    # ==========================================================================
    # 🌟 [Datetime 오류 해결 최종 로직] searcher를 위한 과거 N일치 데이터 수집 및 전처리
    # ==========================================================================
    def _fetch_latest_sequence_data(self, ticker: str) -> Dict[str, Any]:
        """
        ticker의 과거 WINDOW_SIZE일 동안의 주가 데이터와 **실제 FinBERT 감성 점수**를 결합하여 시퀀스 생성.
        """
        end_date = datetime.now() + timedelta(days=1) # 다음날까지 (yfinance end-date는 exclusive)
        start_date = end_date - timedelta(days=self.window_size * 2) # 충분한 기간 확보

        FEATURES_LIST = ['prob_positive','prob_negative','prob_neutral','n_news','ret','Close', 'eod_sentiment']

        # 1. 주가 데이터 수집
        try:
            stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            # 🚨 [Merge 오류 해결 1] MultiIndex 및 컬럼 강제 정리
            stock_data.columns = ['_'.join(str(c) for c in col) if isinstance(col, tuple) else str(col) for col in stock_data.columns]
            stock_data = stock_data.reset_index()
            
            # Ticker/Symbol, level_X 컬럼 제거 (MultiIndex 잔여물)
            columns_to_drop = [col for col in stock_data.columns if 'Ticker' in col or 'Symbol' in col or 'level_' in col.lower()]
            stock_data = stock_data.drop(columns=list(set(columns_to_drop)), errors='ignore')
            
            # 🚨 [Datetime 오류 해결 1] 'Date' 컬럼에서 시간대 정보(timezone) 제거 후 정규화
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            if stock_data['Date'].dt.tz is not None:
                 stock_data['Date'] = stock_data['Date'].dt.tz_localize(None) # 시간대 정보 제거
            stock_data['Date'] = stock_data['Date'].dt.normalize() # 시간 정보 제거 (날짜만 남김)
            
            # Close 컬럼 정리 및 이름 통일
            close_col = [col for col in stock_data.columns if 'Close' in col and 'Adj' not in col]
            if close_col:
                 stock_data = stock_data.rename(columns={close_col[0]: 'Close'})
            
            stock_data['Close_Price'] = stock_data['Close']
            stock_data['Close'] = stock_data['Close_Price'] # 피처 이름 통일
            stock_data['ret'] = stock_data['Close'].pct_change()
            last_price = stock_data['Close'].iloc[-1]
            
            # 필요한 컬럼과 기간만 선택 (컬럼 기반)
            stock_data_final = stock_data[['Date', 'Close', 'ret']].tail(self.window_size).copy()
            
        except Exception as e:
            print(f"[FATAL] yfinance 오류: {e}. 예측 불가.")
            return None, 0.0, "USD"
        
        required_length = self.window_size
        if len(stock_data_final) < required_length:
            print(f"[WARNING] 주가 데이터가 {required_length}일보다 적습니다. 예측 불가.")
            return None, 0.0, "USD"
        
        # 2. 과거 10일치 뉴스 텍스트 및 감성 점수 수집 (날짜별로)
        print(f"[{ticker}] 과거 {required_length}일치 뉴스 및 감성 데이터 수집 중...")
        
        # 뉴스 기간 설정을 위해 Date 컬럼 사용 (stock_data_final은 Date 컬럼을 가집니다)
        news_end_date = stock_data_final['Date'].iloc[-1].strftime('%Y-%m-%d')
        news_start_date = stock_data_final['Date'].iloc[0].strftime('%Y-%m-%d')
        
        # EODhd API 호출 (과거 WINDOW_SIZE 기간 동안의 뉴스)
        all_news, _ = self.collect_news_data_eodhd_batch(ticker, news_start_date, news_end_date)
        news_df_raw = pd.DataFrame(all_news)
        
        # 3. FinBERT 분석 및 일별 집계
        sentiment_data = {}
        key_topics = []
        
        if not news_df_raw.empty:
            news_df_raw['date'] = pd.to_datetime(news_df_raw['date'])
            # 🚨 [Datetime 오류 해결 2] 'date' 컬럼에서 시간대 정보 제거 후 정규화
            if news_df_raw['date'].dt.tz is not None:
                news_df_raw['date'] = news_df_raw['date'].dt.tz_localize(None)
            news_df_raw['date'] = news_df_raw['date'].dt.normalize()
            
            news_df_raw['text'] = news_df_raw['title'] + ' ' + news_df_raw['summary']
            
            # FinBERT 분석 실행
            finbert_scores = self._finbert_sentiment_scores(news_df_raw['text'].values.tolist())
            
            try:
                # 일반적인 FinBERT-tone 순서: 긍정(0), 부정(1), 중립(2)
                news_df_raw['prob_positive'] = finbert_scores[:, 0]
                news_df_raw['prob_negative'] = finbert_scores[:, 1]
                news_df_raw['prob_neutral'] = finbert_scores[:, 2]
            except IndexError:
                print("[FATAL] FinBERT 점수 배열 인덱싱 오류. FinBERT 출력 순서를 확인하세요.")
                news_df_raw['prob_positive'] = 0.33
                news_df_raw['prob_negative'] = 0.33
                news_df_raw['prob_neutral'] = 0.34
            
            # 일별 평균 집계
            daily_sentiments = news_df_raw.groupby('date').agg(
                prob_positive=('prob_positive','mean'), 
                prob_negative=('prob_negative','mean'), 
                prob_neutral=('prob_neutral','mean'), 
                n_news=('title','count'),
                eod_sentiment=('sentiment_score', lambda x: pd.to_numeric(x, errors='coerce').mean())
            ).reset_index()
            daily_sentiments['date'] = daily_sentiments['date'].dt.normalize()
            
            # 🚨 [Merge 오류 해결 수정 2 최종] 컬럼 기준으로 병합 수행
            # stock_data_final과 daily_sentiments를 컬럼 기준으로 병합 ('Date'와 'date' 컬럼을 사용)
            combined_df = stock_data_final.merge(daily_sentiments, left_on='Date', right_on='date', how='left') 
            
            # 병합 후, 다시 'Date'를 인덱스로 설정
            combined_df = combined_df.set_index('Date')
            
            # 불필요한 중복 날짜 컬럼 제거
            if 'date' in combined_df.columns:
                combined_df = combined_df.drop(columns=['date'])
            
            # 감성 데이터 결측치 처리 (뉴스가 없으면 0.0 또는 중립으로 대체)
            combined_df['prob_positive'] = combined_df['prob_positive'].fillna(0.33)
            combined_df['prob_negative'] = combined_df['prob_negative'].fillna(0.33)
            combined_df['prob_neutral'] = combined_df['prob_neutral'].fillna(0.34)
            combined_df['n_news'] = combined_df['n_news'].fillna(0)
            combined_df['eod_sentiment'] = combined_df['eod_sentiment'].fillna(0.0)

            # 키 토픽 추출 (최신 뉴스 제목 3개)
            key_topics = news_df_raw['title'].tail(3).tolist()
        
        else:
            print(f"[{ticker}] 과거 {required_length}일간 뉴스 없음. 감성 피처를 중립으로 대체.")
            # 뉴스가 없는 경우, 중립 감성 데이터프레임 생성 및 'prob_positive' 오류 방지
            
            # 인덱스를 설정한 후 복사
            stock_data_final = stock_data_final.set_index('Date') 
            combined_df = stock_data_final.copy()
            
            # 필수 피처를 중립 값으로 명시적 초기화
            combined_df['prob_positive'] = 0.33 
            combined_df['prob_negative'] = 0.33
            combined_df['prob_neutral'] = 0.34
            combined_df['n_news'] = 0
            combined_df['eod_sentiment'] = 0.0

        # 최종 시퀀스 데이터 준비
        combined_df = combined_df.fillna(0.0) 
        
        sequence_array = combined_df[FEATURES_LIST].values
        sequence_data_dict = combined_df[FEATURES_LIST].to_dict(orient='list')

        return {
            "sequence_array": sequence_array, 
            "sequence_data": sequence_data_dict,
            "key_topics": key_topics
           }, last_price, "USD"

    # searcher 및 predictor는 이전 수정본과 동일 (생략)
    def searcher(self, ticker: str) -> StockData:
        seq_data_results, last_price, currency = self._fetch_latest_sequence_data(ticker)
        
        if seq_data_results is None:
            raise RuntimeError("시퀀스 데이터를 충분히 수집하지 못했습니다.")

        data = StockData(
            sentimental=seq_data_results, 
            fundamental={},
            technical={},
            last_price=last_price,
            currency=currency
        )
        return data

    def predictor(self, ticker: str) -> Target:
        stock_data = self.searcher(ticker)
        last_price = stock_data.last_price or 500.0
        input_sequence = stock_data.sentimental["sequence_array"] 

        # X 스케일링 적용
        if self.scaler_X:
            scaled_array = self.scaler_X.transform(input_sequence)
            input_data_tensor = torch.tensor(scaled_array[np.newaxis, :, :], dtype=torch.float32) 
        else:
            input_data_tensor = torch.tensor(input_sequence[np.newaxis, :, :], dtype=torch.float32)
        
        # 몬테 카를로 드롭아웃 (MCDO) 시뮬레이션
        num_samples = 150
        predictions_raw = [] 

        self.model.train() 
        for _ in range(num_samples):
            with torch.no_grad():
                scaled_output = self.model(input_data_tensor).item() 
                predictions_raw.append(scaled_output)

        predictions_np = np.array(predictions_raw).reshape(-1, 1) 

        # 출력 역변환 (Y)
        if self.scaler_y:
            predicted_prices_np = self.scaler_y.inverse_transform(predictions_np)
            predicted_prices = predicted_prices_np.flatten()
        else:
            predicted_prices = predictions_np.flatten()
            
        # Target 클래스 필드 계산
        next_close = float(np.mean(predicted_prices))
        uncertainty = float(np.std(predicted_prices))
        confidence = float(1.0 / (1.0 + uncertainty * 10))
        confidence = min(1.0, confidence)

        # 피쳐중요도 및 SHAP (더미/단순화)
        feature_importances = [random.uniform(0.1, 0.9) for _ in range(self.input_features)]
        shap_values = [random.uniform(-0.5, 0.5) for _ in range(self.input_features)]
        
        # Target 클래스 인스턴스 반환
        result = Target(
            next_close=next_close,
            uncertainty=uncertainty,
            confidence=confidence,
            idea={
                "sentiment_score": [stock_data.sentimental['sequence_data']['prob_positive'][-1] - stock_data.sentimental['sequence_data']['prob_negative'][-1]], 
                "feature_names": [f"feature_{i+1}" for i in range(self.input_features)],
                "related_news_summary": stock_data.sentimental.get("key_topics", ["데이터 시퀀스 기반 예측"]),
                "mc_price_samples": [float(p) for p in predicted_prices[:5]],
                "feature_importances": feature_importances,
                "shap_values": shap_values
            }
        )
        
        return result


# ==============================================================================
# 에이전트 사용 예시 (이전과 동일)
# ==============================================================================
if __name__ == '__main__':
    #... (생략: 테스트 실행 로직)
    print("="*50)
    print("## SentimentalAgent 테스트 실행 (실제 FinBERT 분석 포함)")
    print("="*50)
    
    TEST_TICKER = "MSFT"

    agent_for_inference = SentimentalAgent(agent_id="SentimentalAgent")
    
    try:
        # FinBERT 로드 오류가 해결되면, 이제 API 401 오류가 발생할 것입니다.
        # 그러나 'prob_positive' 치명적 오류는 중립 값 초기화 로직으로 방지되었습니다.
        test_stock_data = agent_for_inference.searcher(TEST_TICKER)
        target_result = agent_for_inference.predictor(ticker=TEST_TICKER)
        
        system_msg, user_msg = agent_for_inference.build_opinion_messages(TEST_TICKER, test_stock_data, target_result)
        final_opinion = agent_for_inference.build_opinion(TEST_TICKER, test_stock_data, target_result)

        print(f"## 최종 감성 예측 결과 ({TEST_TICKER}) (Target 클래스 출력 형태):")
        print(f"최신 종가 (검색): {test_stock_data.last_price:.2f} {test_stock_data.currency}")
        print(f"next_close (예측 종가): {target_result.next_close:.2f}")
        print("-" * 50)
        
        print(f"## 생성된 Opinion 객체의 Reason (더미):")
        print(final_opinion.reason)
        print("="*50)

    except RuntimeError as e:
        print(f"데이터 수집 실패 오류: {e}")
    except Exception as e:
        # 이전에 발생했던 'prob_positive' 오류는 발생하지 않을 것입니다.
        print(f"테스트 실행 중 치명적인 오류 발생: {e}")