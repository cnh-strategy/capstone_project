"""
Technical Agent의 모듈화된 Searcher와 Predictor
메인 브랜치의 TechnicalAgent에 선택적으로 통합 가능
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import json

class TechnicalSearcher:
    """기술적 분석 데이터 수집 모듈"""
    
    def __init__(self, 
                 news_lang: str = "ko", 
                 news_country: str = "KR", 
                 max_news: int = 20,
                 use_macro: bool = True,
                 fred_api_key: Optional[str] = None):
        
        self.news_lang = news_lang
        self.news_country = news_country
        self.max_news = max_news
        self.use_macro = use_macro
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        
        # FRED 기본 지표 (미국 CPI, 실업률, 금리)
        self.fred_series_map = {
            "CPI(US)": "CPIAUCSL",
            "Unemployment(US)": "UNRATE",
            "PolicyRate(US)": "FEDFUNDS",
        }
    
    def fetch_google_news(self, query: str) -> List[Dict]:
        """구글 뉴스 RSS에서 특정 쿼리 관련 뉴스 가져오기"""
        try:
            base = "https://news.google.com/rss/search"
            params = {
                "q": query, 
                "hl": self.news_lang, 
                "gl": self.news_country, 
                "ceid": f"{self.news_country}:{self.news_lang}"
            }
            r = requests.get(base, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            
            items = []
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:self.max_news]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                desc = (item.findtext("description") or "").strip()
                items.append({"title": title, "summary": desc[:400], "url": link})
            
            return items
        except Exception as e:
            print(f"⚠️ 뉴스 수집 실패: {e}")
            return []
    
    def fetch_macro_snapshot(self, series_map: Dict[str, str]) -> Dict:
        """yfinance 기반 거시경제 지표 수집"""
        print("📊 yfinance 기반 거시경제 지표를 사용합니다.")
        return self._get_fallback_macro_data()
    
    def _get_fallback_macro_data(self) -> Dict:
        """yfinance 기반 기본 거시경제 데이터"""
        try:
            # yfinance로 주요 지표 수집
            vix = yf.download("^VIX", period="1d", progress=False)
            nasdaq = yf.download("^IXIC", period="1d", progress=False)
            sp500 = yf.download("^GSPC", period="1d", progress=False)
            
            return {
                'VIX': float(vix['Close'].iloc[-1]) if not vix.empty else 20.0,
                'NASDAQ': float(nasdaq['Close'].iloc[-1]) if not nasdaq.empty else 15000.0,
                'SP500': float(sp500['Close'].iloc[-1]) if not sp500.empty else 4000.0,
                'DXY': 103.0,  # 달러 인덱스 기본값
                'GOLD': 2000.0,  # 금 가격 기본값
                'OIL': 80.0  # 원유 가격 기본값
            }
        except Exception as e:
            print(f"⚠️ 거시경제 데이터 수집 실패: {e}")
            return {
                'VIX': 20.0,
                'NASDAQ': 15000.0,
                'SP500': 4000.0,
                'DXY': 103.0,
                'GOLD': 2000.0,
                'OIL': 80.0
            }
    
    def _fetch_fred_latest(self, series_id: str) -> Optional[Dict]:
        """FRED API 호출하여 특정 지표의 최신 데이터 반환"""
        try:
            base = "https://api.stlouisfed.org/fred/series/observations"
            start = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
            params = {
                "series_id": series_id, 
                "api_key": self.fred_api_key, 
                "file_type": "json", 
                "observation_start": start
            }
            r = requests.get(base, params=params, timeout=20)
            r.raise_for_status()
            
            obs = r.json().get("observations", [])
            if not obs:
                return None
            
            last = obs[-1]
            return {"date": last["date"], "value": last["value"]}
        except Exception as e:
            print(f"⚠️ FRED 데이터 수집 실패 ({series_id}): {e}")
            return None
    
    def get_price_snapshot(self, ticker: str, period: str = "3mo") -> Dict:
        """yfinance를 사용한 가격 스냅샷 수집"""
        try:
            df = yf.download(ticker, period=period, interval="1d")
            if df.empty:
                return {}
            
            return {
                "current_price": float(df["Close"].iloc[-1]),
                "high_3mo": float(df["High"].max()),
                "low_3mo": float(df["Low"].min()),
                "avg_volume": float(df["Volume"].mean()),
                "volatility": float(df["Close"].pct_change().std() * np.sqrt(252))
            }
        except Exception as e:
            print(f"⚠️ 가격 데이터 수집 실패: {e}")
            return {}
    
    def get_technical_indicators(self, ticker: str, period: str = "1y") -> Dict:
        """기술적 지표 계산"""
        try:
            df = yf.download(ticker, period=period, interval="1d")
            if df.empty:
                return {}
            
            # RSI 계산
            delta = df["Close"].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            avg_gain = up.rolling(14).mean()
            avg_loss = down.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # 이동평균
            ma_20 = df["Close"].rolling(20).mean()
            ma_50 = df["Close"].rolling(50).mean()
            
            # 볼린저 밴드
            bb_middle = df["Close"].rolling(20).mean()
            bb_std = df["Close"].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            return {
                "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                "ma_20": float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else float(df["Close"].iloc[-1]),
                "ma_50": float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else float(df["Close"].iloc[-1]),
                "bb_upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else float(df["Close"].iloc[-1]),
                "bb_lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else float(df["Close"].iloc[-1]),
                "current_price": float(df["Close"].iloc[-1]),
                "volume": float(df["Volume"].iloc[-1])
            }
        except Exception as e:
            print(f"⚠️ 기술적 지표 계산 실패: {e}")
            return {}


class TechnicalPredictor:
    """ML 기반 기술적 분석 예측 모듈"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "model_artifacts/final_best.keras"
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
        self._load_model()
    
    def _load_model(self):
        """훈련된 모델 로드"""
        try:
            if os.path.exists(self.model_path):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
                print("✅ Technical ML 모델 로드 완료")
            else:
                print(f"⚠️ 모델 파일을 찾을 수 없습니다: {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            self.model = None
    
    def predict_with_technical_analysis(self, 
                                      ticker: str, 
                                      current_price: float,
                                      technical_indicators: Dict) -> Dict:
        """기술적 분석을 통한 예측"""
        try:
            # 기본 기술적 분석 신호 생성
            signals = self._generate_technical_signals(technical_indicators)
            
            # ML 모델이 있는 경우 추가 예측
            ml_prediction = None
            if self.model:
                ml_prediction = self._predict_with_ml(ticker, technical_indicators)
            
            return {
                "signals": signals,
                "ml_prediction": ml_prediction,
                "confidence": self._calculate_confidence(signals, ml_prediction)
            }
        except Exception as e:
            print(f"⚠️ 기술적 예측 실패: {e}")
            return {"signals": {}, "ml_prediction": None, "confidence": 0.0}
    
    def _generate_technical_signals(self, indicators: Dict) -> Dict:
        """기술적 지표로부터 신호 생성"""
        signals = {}
        
        # RSI 신호
        rsi = indicators.get("rsi", 50)
        if rsi > 70:
            signals["rsi"] = "overbought"
        elif rsi < 30:
            signals["rsi"] = "oversold"
        else:
            signals["rsi"] = "neutral"
        
        # 이동평균 신호
        current_price = indicators.get("current_price", 0)
        ma_20 = indicators.get("ma_20", current_price)
        ma_50 = indicators.get("ma_50", current_price)
        
        if current_price > ma_20 > ma_50:
            signals["trend"] = "bullish"
        elif current_price < ma_20 < ma_50:
            signals["trend"] = "bearish"
        else:
            signals["trend"] = "sideways"
        
        # 볼린저 밴드 신호
        bb_upper = indicators.get("bb_upper", current_price)
        bb_lower = indicators.get("bb_lower", current_price)
        
        if current_price > bb_upper:
            signals["bollinger"] = "overbought"
        elif current_price < bb_lower:
            signals["bollinger"] = "oversold"
        else:
            signals["bollinger"] = "normal"
        
        return signals
    
    def _predict_with_ml(self, ticker: str, indicators: Dict) -> Optional[float]:
        """ML 모델을 사용한 예측"""
        try:
            # 간단한 피처 벡터 생성 (실제로는 더 복잡한 피처 엔지니어링 필요)
            features = np.array([
                indicators.get("rsi", 50) / 100,
                indicators.get("current_price", 0) / indicators.get("ma_20", 1),
                indicators.get("current_price", 0) / indicators.get("ma_50", 1),
                indicators.get("volume", 0) / 1000000,  # 정규화
            ]).reshape(1, -1)
            
            if self.model:
                prediction = self.model.predict(features, verbose=0)[0][0]
                return float(prediction)
        except Exception as e:
            print(f"⚠️ ML 예측 실패: {e}")
        
        return None
    
    def _calculate_confidence(self, signals: Dict, ml_prediction: Optional[float]) -> float:
        """예측 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 신호 일치도에 따른 신뢰도 조정
        signal_agreement = 0
        total_signals = 0
        
        for signal_type, signal_value in signals.items():
            total_signals += 1
            if signal_value in ["bullish", "oversold"]:
                signal_agreement += 1
            elif signal_value in ["bearish", "overbought"]:
                signal_agreement -= 1
        
        if total_signals > 0:
            signal_confidence = abs(signal_agreement) / total_signals
            confidence = 0.3 + (signal_confidence * 0.4)  # 0.3 ~ 0.7 범위
        
        # ML 예측이 있는 경우 추가 조정
        if ml_prediction is not None:
            confidence = min(confidence + 0.2, 0.9)  # 최대 0.9
        
        return confidence


class TechnicalModuleManager:
    """기술적 분석 모듈 통합 관리자"""
    
    def __init__(self, 
                 use_ml_searcher: bool = False,
                 use_ml_predictor: bool = False,
                 fred_api_key: Optional[str] = None,
                 model_path: Optional[str] = None):
        
        self.use_ml_searcher = use_ml_searcher
        self.use_ml_predictor = use_ml_predictor
        
        # 모듈 초기화
        if self.use_ml_searcher:
            self.searcher = TechnicalSearcher(fred_api_key=fred_api_key)
        else:
            self.searcher = None
            
        if self.use_ml_predictor:
            self.predictor = TechnicalPredictor(model_path=model_path)
        else:
            self.predictor = None
    
    def get_enhanced_technical_data(self, ticker: str, current_price: float) -> Dict:
        """ML 모듈을 활용한 향상된 기술적 분석 데이터 생성"""
        result = {
            "signals": {},
            "indicators": {},
            "ml_prediction": None,
            "confidence": 0.0,
            "summary": ""
        }
        
        # ML Searcher 사용
        if self.use_ml_searcher and self.searcher:
            try:
                # 뉴스 데이터 수집
                news_data = self.searcher.fetch_google_news(ticker)
                
                # 매크로 데이터 수집
                macro_data = self.searcher.fetch_macro_snapshot(self.searcher.fred_series_map)
                
                # 가격 스냅샷
                price_snapshot = self.searcher.get_price_snapshot(ticker)
                
                # 기술적 지표
                technical_indicators = self.searcher.get_technical_indicators(ticker)
                
                result["indicators"] = technical_indicators
                result["price_snapshot"] = price_snapshot
                result["macro_data"] = macro_data
                result["news_count"] = len(news_data)
                
                # ML Predictor 사용
                if self.use_ml_predictor and self.predictor:
                    try:
                        prediction_result = self.predictor.predict_with_technical_analysis(
                            ticker, current_price, technical_indicators
                        )
                        
                        result["signals"] = prediction_result["signals"]
                        result["ml_prediction"] = prediction_result["ml_prediction"]
                        result["confidence"] = prediction_result["confidence"]
                        
                        # 요약 생성
                        result["summary"] = self._generate_technical_summary(
                            prediction_result["signals"], 
                            prediction_result["confidence"]
                        )
                        
                    except Exception as e:
                        print(f"⚠️ ML 예측 실패: {e}")
                        
            except Exception as e:
                print(f"⚠️ ML 데이터 수집 실패: {e}")
        
        return result
    
    def _generate_technical_summary(self, signals: Dict, confidence: float) -> str:
        """기술적 분석 요약 생성"""
        summary_parts = []
        
        # RSI 신호
        rsi_signal = signals.get("rsi", "neutral")
        if rsi_signal == "overbought":
            summary_parts.append("RSI가 과매수 구간에 있어 조정 가능성이 있습니다.")
        elif rsi_signal == "oversold":
            summary_parts.append("RSI가 과매도 구간에 있어 반등 가능성이 있습니다.")
        
        # 추세 신호
        trend_signal = signals.get("trend", "sideways")
        if trend_signal == "bullish":
            summary_parts.append("이동평균선이 상승 추세를 보이고 있습니다.")
        elif trend_signal == "bearish":
            summary_parts.append("이동평균선이 하락 추세를 보이고 있습니다.")
        
        # 볼린저 밴드 신호
        bb_signal = signals.get("bollinger", "normal")
        if bb_signal == "overbought":
            summary_parts.append("볼린저 밴드 상단을 돌파하여 조정 압력이 있습니다.")
        elif bb_signal == "oversold":
            summary_parts.append("볼린저 밴드 하단 근처에서 지지받고 있습니다.")
        
        # 신뢰도 정보
        if confidence > 0.7:
            summary_parts.append("기술적 신호가 강하게 나타나고 있습니다.")
        elif confidence < 0.4:
            summary_parts.append("기술적 신호가 모호한 상황입니다.")
        
        return " ".join(summary_parts) if summary_parts else "기술적 분석 데이터를 수집했습니다."
    
    def train_model(self, ticker: str) -> bool:
        """모델 학습 (간단한 시뮬레이션)"""
        try:
            # 간단한 학습 시뮬레이션
            print(f"🎯 {ticker} 기술적 모델 학습 시작...")
            print(f"✅ {ticker} 기술적 모델 학습 완료 (시뮬레이션)")
            return True
        except Exception as e:
            print(f"❌ {ticker} 기술적 모델 학습 실패: {str(e)}")
            return False
    
    def predict_price(self, ticker: str) -> tuple:
        """가격 예측 (간단한 시뮬레이션)"""
        try:
            # 간단한 예측 시뮬레이션
            import random
            base_price = 100.0
            prediction = base_price * random.uniform(0.98, 1.02)  # ±2% 변동 (기술적 분석은 안정적)
            uncertainty = random.uniform(0.05, 0.15)
            
            return prediction, uncertainty
        except Exception as e:
            print(f"❌ {ticker} 기술적 예측 실패: {str(e)}")
            return 0.0, 1.0


# 사용 예제
if __name__ == "__main__":
    # 모듈 매니저 초기화
    manager = TechnicalModuleManager(
        use_ml_searcher=True,
        use_ml_predictor=True,
        fred_api_key=os.getenv('FRED_API_KEY'),
        model_path="model_artifacts/final_best.keras"
    )
    
    # 테스트
    ticker = "AAPL"
    current_price = 150.0
    
    enhanced_data = manager.get_enhanced_technical_data(ticker, current_price)
    print("향상된 기술적 분석 데이터:", enhanced_data)
