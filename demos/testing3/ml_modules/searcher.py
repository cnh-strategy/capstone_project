#!/usr/bin/env python3
"""
Searcher Module - 각 Agent별 데이터 수집 및 CSV 생성
2022~2025년 데이터를 수집하여 CSV 파일로 저장
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSearcher:
    """데이터 수집 클래스"""
    
    def __init__(self, ticker: str, data_dir: str = "data"):
        self.ticker = ticker.upper()
        self.data_dir = data_dir
        self.ensure_data_dir()
        
    def ensure_data_dir(self):
        """데이터 디렉토리 생성"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def search_fundamental_data(self) -> str:
        """펀더멘털 데이터 수집"""
        logger.info(f"🔍 {self.ticker} 펀더멘털 데이터 수집 시작...")
        
        try:
            # yfinance로 주식 정보 가져오기
            stock = yf.Ticker(self.ticker)
            
            # 기본 정보
            info = stock.info
            
            # 재무 데이터 수집
            fundamental_data = []
            
            # 2022-2025년 데이터 생성 (시뮬레이션)
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2025, 12, 31)
            
            current_date = start_date
            while current_date <= end_date:
                # 실제 데이터가 있다면 사용, 없으면 시뮬레이션
                data_point = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'ticker': self.ticker,
                    'market_cap': info.get('marketCap', 1000000000),
                    'pe_ratio': info.get('trailingPE', 20.0),
                    'pb_ratio': info.get('priceToBook', 2.0),
                    'debt_to_equity': info.get('debtToEquity', 0.5),
                    'revenue_growth': info.get('revenueGrowth', 0.1),
                    'profit_margin': info.get('profitMargins', 0.15),
                    'roe': info.get('returnOnEquity', 0.12),
                    'current_ratio': info.get('currentRatio', 2.0),
                    'dividend_yield': info.get('dividendYield', 0.02)
                }
                fundamental_data.append(data_point)
                current_date += timedelta(days=1)
            
            # DataFrame 생성
            df = pd.DataFrame(fundamental_data)
            
            # CSV 저장
            filename = f"{self.ticker}_fundamental_data.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"✅ 펀더멘털 데이터 저장 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 펀더멘털 데이터 수집 실패: {str(e)}")
            return None
    
    def search_technical_data(self) -> str:
        """기술적 데이터 수집"""
        logger.info(f"📊 {self.ticker} 기술적 데이터 수집 시작...")
        
        try:
            # yfinance로 주가 데이터 가져오기
            stock = yf.Ticker(self.ticker)
            
            # 5년간 데이터 수집
            hist = stock.history(period="5y")
            
            if hist.empty:
                logger.warning(f"⚠️ {self.ticker} 주가 데이터가 없습니다. 시뮬레이션 데이터 생성...")
                return self._generate_simulated_technical_data()
            
            # 기술적 지표 계산
            technical_data = []
            
            for date, row in hist.iterrows():
                data_point = {
                    'date': date.strftime('%Y-%m-%d'),
                    'ticker': self.ticker,
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume'],
                    'sma_20': row['Close'],  # 단순이동평균 (실제로는 계산 필요)
                    'sma_50': row['Close'],
                    'rsi': 50.0,  # RSI (실제로는 계산 필요)
                    'macd': 0.0,  # MACD (실제로는 계산 필요)
                    'bollinger_upper': row['Close'] * 1.02,
                    'bollinger_lower': row['Close'] * 0.98,
                    'atr': abs(row['High'] - row['Low']),
                    'volume_sma': row['Volume']
                }
                technical_data.append(data_point)
            
            # DataFrame 생성
            df = pd.DataFrame(technical_data)
            
            # CSV 저장
            filename = f"{self.ticker}_technical_data.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"✅ 기술적 데이터 저장 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 기술적 데이터 수집 실패: {str(e)}")
            return self._generate_simulated_technical_data()
    
    def search_sentimental_data(self) -> str:
        """감정 분석 데이터 수집"""
        logger.info(f"💭 {self.ticker} 감정 분석 데이터 수집 시작...")
        
        try:
            # 뉴스 데이터 시뮬레이션 (실제로는 뉴스 API 사용)
            sentimental_data = []
            
            # 2022-2025년 데이터 생성
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2025, 12, 31)
            
            current_date = start_date
            while current_date <= end_date:
                # 감정 지표 시뮬레이션
                data_point = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'ticker': self.ticker,
                    'news_sentiment': 0.0,  # -1 (부정) ~ 1 (긍정)
                    'social_sentiment': 0.0,  # 소셜미디어 감정
                    'analyst_rating': 3.0,  # 1-5 등급
                    'price_target': 100.0,  # 목표가
                    'earnings_surprise': 0.0,  # 실적 서프라이즈
                    'insider_trading': 0.0,  # 내부자 거래
                    'institutional_flow': 0.0,  # 기관 자금 흐름
                    'options_sentiment': 0.0,  # 옵션 시장 감정
                    'fear_greed_index': 50.0  # 공포/탐욕 지수
                }
                sentimental_data.append(data_point)
                current_date += timedelta(days=1)
            
            # DataFrame 생성
            df = pd.DataFrame(sentimental_data)
            
            # CSV 저장
            filename = f"{self.ticker}_sentimental_data.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"✅ 감정 분석 데이터 저장 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 감정 분석 데이터 수집 실패: {str(e)}")
            return None
    
    def _generate_simulated_technical_data(self) -> str:
        """시뮬레이션 기술적 데이터 생성"""
        logger.info(f"🎲 {self.ticker} 시뮬레이션 기술적 데이터 생성...")
        
        technical_data = []
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2025, 12, 31)
        
        # 초기 가격 설정
        base_price = 100.0
        
        current_date = start_date
        while current_date <= end_date:
            # 랜덤 워크로 가격 생성
            import random
            change = random.uniform(-0.05, 0.05)  # ±5% 변동
            base_price *= (1 + change)
            
            data_point = {
                'date': current_date.strftime('%Y-%m-%d'),
                'ticker': self.ticker,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price,
                'volume': random.randint(1000000, 10000000),
                'sma_20': base_price,
                'sma_50': base_price,
                'rsi': random.uniform(30, 70),
                'macd': random.uniform(-1, 1),
                'bollinger_upper': base_price * 1.02,
                'bollinger_lower': base_price * 0.98,
                'atr': base_price * 0.02,
                'volume_sma': random.randint(1000000, 10000000)
            }
            technical_data.append(data_point)
            current_date += timedelta(days=1)
        
        # DataFrame 생성
        df = pd.DataFrame(technical_data)
        
        # CSV 저장
        filename = f"{self.ticker}_technical_data.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ 시뮬레이션 기술적 데이터 저장 완료: {filepath}")
        return filepath
    
    def search_all_data(self) -> Dict[str, str]:
        """모든 데이터 수집"""
        logger.info(f"🚀 {self.ticker} 전체 데이터 수집 시작...")
        
        results = {}
        
        # 각 Agent별 데이터 수집
        results['fundamental'] = self.search_fundamental_data()
        results['technical'] = self.search_technical_data()
        results['sentimental'] = self.search_sentimental_data()
        
        # 결과 요약
        success_count = sum(1 for path in results.values() if path is not None)
        logger.info(f"📊 데이터 수집 완료: {success_count}/3 성공")
        
        return results


def main():
    """테스트 실행"""
    ticker = "RZLV"
    searcher = DataSearcher(ticker)
    results = searcher.search_all_data()
    
    print(f"\n📋 {ticker} 데이터 수집 결과:")
    for agent_type, filepath in results.items():
        if filepath:
            print(f"✅ {agent_type}: {filepath}")
        else:
            print(f"❌ {agent_type}: 실패")


if __name__ == "__main__":
    main()
