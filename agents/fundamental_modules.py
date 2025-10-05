"""
Fundamental Agent의 모듈화된 Searcher와 Predictor
메인 브랜치의 FundamentalAgent에 선택적으로 통합 가능
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import json

class FundamentalSearcher:
    """펀더멘털 분석 데이터 수집 모듈"""
    
    def __init__(self, use_quarterly_data: bool = True, delay_days: int = 45):
        self.use_quarterly_data = use_quarterly_data
        self.delay_days = delay_days
    
    def get_quarterly_report(self, symbol: str, date: str) -> Optional[Dict]:
        """특정 일자와 티커 기준으로 45일 딜레이 적용 후 가장 최신 분기 보고서를 반환"""
        try:
            tk = yf.Ticker(symbol)
            
            # 분기 데이터
            income = tk.quarterly_financials.T
            balance = tk.quarterly_balance_sheet.T
            cashflow = tk.quarterly_cashflow.T
            info = tk.info  # 시가총액, 배당, 베타 등
            
            target_date = datetime.strptime(date, "%Y-%m-%d")
            
            # 분기별 loop
            valid_periods = []
            for period in income.index:
                report_available_date = period + timedelta(days=self.delay_days)
                if report_available_date <= target_date:
                    valid_periods.append(period)
            
            if not valid_periods:
                return None  # 해당 날짜까지 보고서 없음
            
            # 가장 최근 분기 선택
            latest_period = max(valid_periods)
            
            row_income = income.loc[latest_period] if latest_period in income.index else {}
            row_balance = balance.loc[latest_period] if latest_period in balance.index else {}
            row_cash = cashflow.loc[latest_period] if latest_period in cashflow.index else {}
            
            # 기본 재무 지표
            net_income = row_income.get("Net Income")
            revenue = row_income.get("Total Revenue")
            operating_income = row_income.get("Operating Income")
            gross_profit = row_income.get("Gross Profit")
            
            total_assets = row_balance.get("Total Assets")
            total_liabilities = row_balance.get("Total Liabilities")
            current_assets = row_balance.get("Current Assets")
            current_liabilities = row_balance.get("Current Liabilities")
            
            operating_cf = row_cash.get("Total Cash From Operating Activities")
            capex = row_cash.get("Capital Expenditures")
            free_cf = (operating_cf or 0) + (capex or 0)
            
            # 파생 지표
            profit_margin = net_income / revenue if revenue else None
            debt_to_equity = (
                total_liabilities / (total_assets - total_liabilities)
                if total_assets and total_liabilities else None
            )
            current_ratio = (
                current_assets / current_liabilities
                if current_assets and current_liabilities else None
            )
            
            # 티커 info 기반
            market_cap = info.get("marketCap")
            dividend_yield = info.get("dividendYield")
            beta = info.get("beta")
            forward_pe = info.get("forwardPE")
            pe = info.get("trailingPE")
            eps = info.get("trailingEps")
            pbr = info.get("priceToBook")
            
            return {
                "symbol": symbol,
                "period": latest_period.strftime("%Y-%m-%d"),
                "year": latest_period.year,
                "net_income": net_income,
                "eps": eps,
                "pe": pe,
                "pbr": pbr,
                "revenue": revenue,
                "operating_income": operating_income,
                "gross_profit": gross_profit,
                "profit_margin": profit_margin,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "debt_to_equity": debt_to_equity,
                "current_ratio": current_ratio,
                "market_cap": market_cap,
                "dividend_yield": dividend_yield,
                "beta": beta,
                "forward_pe": forward_pe,
                "operating_cashflow": operating_cf,
                "capex": capex,
                "free_cashflow": free_cf,
            }
        except Exception as e:
            print(f"⚠️ 분기 보고서 수집 실패: {e}")
            return None
    
    def get_market_data(self, end_date: str) -> Dict:
        """여러 지표의 1년치 Close 가격을 하나의 DataFrame으로 반환"""
        try:
            tickers = {
                "DXY": "DX-Y.NYB",
                "NASDAQ": "^IXIC",
                "S&P500": "^GSPC",
                "DOWJONES": "^DJI",
                "VIX": "^VIX",
                "US10Y": "^TNX"
            }
            
            end = datetime.strptime(end_date, "%Y-%m-%d")
            start = end - timedelta(days=365)
            
            df = yf.download(
                list(tickers.values()),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False
            )["Close"]
            
            # 멀티인덱스 컬럼이면 풀기
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            # 컬럼명 매핑
            rename_map = {v: k for k, v in tickers.items()}
            df = df.rename(columns=rename_map)
            
            # 최신 값들 반환
            latest = df.iloc[-1]
            return {
                "VIX": float(latest.get("VIX", 0)),
                "S&P500": float(latest.get("S&P500", 0)),
                "NASDAQ": float(latest.get("NASDAQ", 0)),
                "DXY": float(latest.get("DXY", 0)),
                "DOWJONES": float(latest.get("DOWJONES", 0)),
                "US10Y": float(latest.get("US10Y", 0))
            }
        except Exception as e:
            print(f"⚠️ 시장 데이터 수집 실패: {e}")
            return {}
    
    def get_price_history(self, ticker: str, end_date: str) -> Dict:
        """지정한 티커와 기준일로부터 1년 전까지의 종가를 불러오는 함수"""
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d")
            start = end - timedelta(days=365)
            
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d")
            )["Close"].reset_index()
            
            # 컬럼명을 통일: 항상 ["Date", "Close"]
            df = df.rename(columns={df.columns[1]: "Close"})
            
            return {
                "price_data": df.to_dict('records'),
                "current_price": float(df["Close"].iloc[-1]),
                "year_high": float(df["Close"].max()),
                "year_low": float(df["Close"].min()),
                "year_return": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)
            }
        except Exception as e:
            print(f"⚠️ 가격 히스토리 수집 실패: {e}")
            return {}


class FundamentalPredictor:
    """ML 기반 펀더멘털 분석 예측 모듈"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "fundamental_model_maker/2025/models22/final_lgbm.pkl"
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
        self._load_model()
    
    def _load_model(self):
        """훈련된 모델 로드"""
        try:
            if os.path.exists(self.model_path):
                import joblib
                self.model = joblib.load(self.model_path)
                
                # 스케일러와 피처 컬럼도 로드 시도
                scaler_path = os.path.join(os.path.dirname(self.model_path), "scaler.pkl")
                feature_path = os.path.join(os.path.dirname(self.model_path), "feature_cols.json")
                
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                
                if os.path.exists(feature_path):
                    with open(feature_path, 'r') as f:
                        self.feature_cols = json.load(f)
                
                print("✅ Fundamental ML 모델 로드 완료")
            else:
                print(f"⚠️ 모델 파일을 찾을 수 없습니다: {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            self.model = None
    
    def predict_with_fundamental_analysis(self, 
                                        ticker: str, 
                                        current_price: float,
                                        fundamental_data: Dict,
                                        market_data: Dict) -> Dict:
        """펀더멘털 분석을 통한 예측"""
        try:
            # 기본 펀더멘털 신호 생성
            signals = self._generate_fundamental_signals(fundamental_data, market_data)
            
            # ML 모델이 있는 경우 추가 예측
            ml_prediction = None
            if self.model:
                ml_prediction = self._predict_with_ml(ticker, fundamental_data, market_data)
            
            return {
                "signals": signals,
                "ml_prediction": ml_prediction,
                "confidence": self._calculate_confidence(signals, ml_prediction)
            }
        except Exception as e:
            print(f"⚠️ 펀더멘털 예측 실패: {e}")
            return {"signals": {}, "ml_prediction": None, "confidence": 0.0}
    
    def _generate_fundamental_signals(self, fundamental_data: Dict, market_data: Dict) -> Dict:
        """펀더멘털 지표로부터 신호 생성"""
        signals = {}
        
        # 밸류에이션 신호
        pe = fundamental_data.get("pe")
        forward_pe = fundamental_data.get("forward_pe")
        pbr = fundamental_data.get("pbr")
        
        if pe and pe < 15:
            signals["valuation"] = "undervalued"
        elif pe and pe > 25:
            signals["valuation"] = "overvalued"
        else:
            signals["valuation"] = "fair"
        
        # 수익성 신호
        profit_margin = fundamental_data.get("profit_margin")
        if profit_margin and profit_margin > 0.15:
            signals["profitability"] = "strong"
        elif profit_margin and profit_margin < 0.05:
            signals["profitability"] = "weak"
        else:
            signals["profitability"] = "moderate"
        
        # 재무 건전성 신호
        debt_to_equity = fundamental_data.get("debt_to_equity")
        current_ratio = fundamental_data.get("current_ratio")
        
        if debt_to_equity and debt_to_equity < 0.3:
            signals["financial_health"] = "strong"
        elif debt_to_equity and debt_to_equity > 0.7:
            signals["financial_health"] = "weak"
        else:
            signals["financial_health"] = "moderate"
        
        # 현금흐름 신호
        free_cashflow = fundamental_data.get("free_cashflow")
        if free_cashflow and free_cashflow > 0:
            signals["cashflow"] = "positive"
        elif free_cashflow and free_cashflow < 0:
            signals["cashflow"] = "negative"
        else:
            signals["cashflow"] = "neutral"
        
        return signals
    
    def _predict_with_ml(self, ticker: str, fundamental_data: Dict, market_data: Dict) -> Optional[float]:
        """ML 모델을 사용한 예측"""
        try:
            if not self.model or not self.feature_cols:
                return None
            
            # 피처 벡터 생성 (실제로는 더 복잡한 피처 엔지니어링 필요)
            features = []
            for col in self.feature_cols:
                if col.startswith("sym_"):
                    # 심볼 원핫 인코딩
                    features.append(1.0 if col == f"sym_{ticker}" else 0.0)
                elif col in fundamental_data:
                    features.append(fundamental_data[col] or 0.0)
                elif col in market_data:
                    features.append(market_data[col] or 0.0)
                else:
                    features.append(0.0)
            
            features = np.array(features).reshape(1, -1)
            
            # 스케일링
            if self.scaler:
                features = self.scaler.transform(features)
            
            # 예측
            prediction = self.model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            print(f"⚠️ ML 예측 실패: {e}")
        
        return None
    
    def _calculate_confidence(self, signals: Dict, ml_prediction: Optional[float]) -> float:
        """예측 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 신호 일치도에 따른 신뢰도 조정
        positive_signals = 0
        total_signals = 0
        
        for signal_type, signal_value in signals.items():
            total_signals += 1
            if signal_value in ["undervalued", "strong", "positive"]:
                positive_signals += 1
            elif signal_value in ["overvalued", "weak", "negative"]:
                positive_signals -= 1
        
        if total_signals > 0:
            signal_confidence = abs(positive_signals) / total_signals
            confidence = 0.3 + (signal_confidence * 0.4)  # 0.3 ~ 0.7 범위
        
        # ML 예측이 있는 경우 추가 조정
        if ml_prediction is not None:
            confidence = min(confidence + 0.2, 0.9)  # 최대 0.9
        
        return confidence


class FundamentalModuleManager:
    """펀더멘털 분석 모듈 통합 관리자"""
    
    def __init__(self, 
                 use_ml_searcher: bool = False,
                 use_ml_predictor: bool = False,
                 model_path: Optional[str] = None):
        
        self.use_ml_searcher = use_ml_searcher
        self.use_ml_predictor = use_ml_predictor
        
        # 모듈 초기화
        if self.use_ml_searcher:
            self.searcher = FundamentalSearcher()
        else:
            self.searcher = None
            
        if self.use_ml_predictor:
            self.predictor = FundamentalPredictor(model_path=model_path)
        else:
            self.predictor = None
    
    def get_enhanced_fundamental_data(self, ticker: str, current_price: float) -> Dict:
        """ML 모듈을 활용한 향상된 펀더멘털 분석 데이터 생성"""
        result = {
            "signals": {},
            "fundamental_data": {},
            "market_data": {},
            "ml_prediction": None,
            "confidence": 0.0,
            "summary": ""
        }
        
        # ML Searcher 사용
        if self.use_ml_searcher and self.searcher:
            try:
                # 분기 보고서 수집
                today = datetime.now().strftime("%Y-%m-%d")
                fundamental_data = self.searcher.get_quarterly_report(ticker, today)
                
                # 시장 데이터 수집
                market_data = self.searcher.get_market_data(today)
                
                # 가격 히스토리
                price_history = self.searcher.get_price_history(ticker, today)
                
                if fundamental_data:
                    result["fundamental_data"] = fundamental_data
                    result["market_data"] = market_data
                    result["price_history"] = price_history
                    
                    # ML Predictor 사용
                    if self.use_ml_predictor and self.predictor:
                        try:
                            prediction_result = self.predictor.predict_with_fundamental_analysis(
                                ticker, current_price, fundamental_data, market_data
                            )
                            
                            result["signals"] = prediction_result["signals"]
                            result["ml_prediction"] = prediction_result["ml_prediction"]
                            result["confidence"] = prediction_result["confidence"]
                            
                            # 요약 생성
                            result["summary"] = self._generate_fundamental_summary(
                                prediction_result["signals"], 
                                prediction_result["confidence"]
                            )
                            
                        except Exception as e:
                            print(f"⚠️ ML 예측 실패: {e}")
                        
            except Exception as e:
                print(f"⚠️ ML 데이터 수집 실패: {e}")
        
        return result
    
    def _generate_fundamental_summary(self, signals: Dict, confidence: float) -> str:
        """펀더멘털 분석 요약 생성"""
        summary_parts = []
        
        # 밸류에이션 신호
        valuation = signals.get("valuation", "fair")
        if valuation == "undervalued":
            summary_parts.append("현재 주가가 내재가치 대비 저평가되어 있습니다.")
        elif valuation == "overvalued":
            summary_parts.append("현재 주가가 내재가치 대비 고평가되어 있습니다.")
        
        # 수익성 신호
        profitability = signals.get("profitability", "moderate")
        if profitability == "strong":
            summary_parts.append("수익성이 강한 편입니다.")
        elif profitability == "weak":
            summary_parts.append("수익성이 약한 편입니다.")
        
        # 재무 건전성 신호
        financial_health = signals.get("financial_health", "moderate")
        if financial_health == "strong":
            summary_parts.append("재무 건전성이 양호합니다.")
        elif financial_health == "weak":
            summary_parts.append("재무 건전성에 주의가 필요합니다.")
        
        # 현금흐름 신호
        cashflow = signals.get("cashflow", "neutral")
        if cashflow == "positive":
            summary_parts.append("자유현금흐름이 양호합니다.")
        elif cashflow == "negative":
            summary_parts.append("자유현금흐름이 부족합니다.")
        
        # 신뢰도 정보
        if confidence > 0.7:
            summary_parts.append("펀더멘털 분석 신호가 명확합니다.")
        elif confidence < 0.4:
            summary_parts.append("펀더멘털 분석 신호가 모호합니다.")
        
        return " ".join(summary_parts) if summary_parts else "펀더멘털 분석 데이터를 수집했습니다."


# 사용 예제
if __name__ == "__main__":
    # 모듈 매니저 초기화
    manager = FundamentalModuleManager(
        use_ml_searcher=True,
        use_ml_predictor=True,
        model_path="fundamental_model_maker/2025/models22/final_lgbm.pkl"
    )
    
    # 테스트
    ticker = "AAPL"
    current_price = 150.0
    
    enhanced_data = manager.get_enhanced_fundamental_data(ticker, current_price)
    print("향상된 펀더멘털 분석 데이터:", enhanced_data)
