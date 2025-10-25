import json
import numpy as np
import pandas as pd
import yfinance as yf
import os
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple
from agents.fundamental_modules import FundamentalModuleManager
from prompts import SEARCHER_PROMPTS, PREDICTER_PROMPTS, OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
# from agents.fundamental_modules import FundamentalModuleManager
class FundamentalAgent(BaseAgent):
    def __init__(self, 
                 agent_id: str = "FundamentalAgent",
                 use_ml_modules: bool = False,
                 model_path: Optional[str] = None,
                 **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        
        # ML 모듈 설정
        if self.use_ml_modules:
            self.ml_manager = FundamentalModuleManager(
                use_ml_searcher=True,
                use_ml_predictor=True,
                model_path=model_path or "fundamental_model_maker/2025/models22/final_lgbm.pkl"
            )
        else:
            self.ml_manager = None
    
    # ------------------------------------------------------------------
    # 1) 데이터 수집 
    # ------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        # 현재가와 통화 가져오기
        df = yf.download(ticker, period="5d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()
        
        schema_fund = {
            "type": "object",
            "properties": {
                "quality":   {"type": "string"},
                "growth":    {"type": "string"},
                "profit":    {"type": "string"},
                "leverage":  {"type": "string"},
                "valuation": {"type": "string"},
                "summary":   {"type": "string"},
                "evidence":  {"type": "array", "items": {"type": "string"}},
            },
            "required": ["quality", "growth", "profit", "leverage", "valuation", "summary", "evidence"],
            "additionalProperties": False,
        }   

        sys_text = SEARCHER_PROMPTS["fundamental"]["system"]
        user_text = SEARCHER_PROMPTS["fundamental"]["user_template"].format(
            ticker=ticker, 
            current_price=last_price, 
            currency=currency
        )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            schema_fund
        )
        
        # ML 모듈 사용 여부에 따른 데이터 수집
        if self.use_ml_modules and self.ml_manager:
            # ML 모듈을 사용한 향상된 펀더멘털 분석 데이터 수집
            ml_fundamental_data = self.ml_manager.get_enhanced_fundamental_data(ticker, last_price)
            
            # ML 결과를 펀더멘털 데이터에 추가
            parsed["ml_signals"] = ml_fundamental_data.get('signals', {})
            parsed["ml_confidence"] = ml_fundamental_data.get('confidence', 0.0)
            parsed["ml_fundamental_data"] = ml_fundamental_data.get('fundamental_data', {})
            
            # ML 결과를 GPT 프롬프트에 포함하여 재분석
            ml_context = f"""
ML 모델 분석 결과:
- 펀더멘털 신호: {ml_fundamental_data.get('signals', {})}
- 신뢰도: {ml_fundamental_data.get('confidence', 0.0):.2f}
- 분기 보고서: {ml_fundamental_data.get('fundamental_data', {}).get('period', 'N/A')}
- 시장 데이터: VIX, S&P500, NASDAQ 등 수집 완료
"""

            # ML 컨텍스트를 포함한 재분석
            user_text_with_ml = SEARCHER_PROMPTS["fundamental"]["user_template"].format(
                ticker=ticker, 
                current_price=last_price, 
                currency=currency
            ) + f"\n\n{ml_context}"

            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text_with_ml),
                schema_fund
            )

        self.stockdata = StockData(
            fundamental=parsed, 
            sentimental={}, 
            technical={},
            last_price=last_price,
            currency=currency
        )
        self.current_ticker = ticker  # 현재 티커 저장
        return self.stockdata

    # ------------------------------------------------------------------
    # 2) 1차 예측 
    # ------------------------------------------------------------------
    def predicter(self, stock_data: StockData) -> Target:
        # 현재 가격 정보 가져오기
        ticker = getattr(self, 'current_ticker', 'UNKNOWN')
        df = yf.download(ticker, period="1d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()
        
        # 펀더멘털 분석가 특성: 보수적, 현재가 대비 ±5% 범위
        min_price = last_price * 0.95
        max_price = last_price * 1.05
        
        ctx = {
            "fundamental_summary": stock_data.fundamental,
            "current_price": last_price,
            "currency": currency,
            "prediction_range": f"{min_price:.2f} - {max_price:.2f} {currency}",
            "agent_character": "보수적인 펀더멘털 분석가로서 장기 가치에 기반한 안정적인 예측을 제공합니다."
            }
        
        sys_text = PREDICTER_PROMPTS["fundamental"]["system"]
        user_text = PREDICTER_PROMPTS["fundamental"]["user_template"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            self.schema_obj_opinion
        )
        return Target(next_close=float(parsed.get("next_close", 0.0)))

    # ------------------------------------------------------------------
    # 3) Opinion 메시지 빌드 (가치 관점)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()
        last = float(stock_data.last_price or 0.0)

        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_price": last,
            "fundamental_summary": stock_data.fundamental or {},
            "our_prediction": float(target.next_close),
        }

        system_text = OPINION_PROMPTS["fundamental"]["system"]
        user_text   = OPINION_PROMPTS["fundamental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text


    # ------------------------------------------------------------------
    # 4) Rebuttal/Revision (가치 관점 문구)
    # ------------------------------------------------------------------
    def _build_messages_rebuttal(self,
                                my_opinion: Opinion,
                                target_agent: str,
                                target_opinion: Opinion,
                                stock_data: StockData) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()

        ctx = {
            "ticker": t,
            "currency": ccy,
            "fundamental_summary": stock_data.fundamental or {},
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:2000],
            },
            "other": {
                "agent_id": target_agent,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
            }
        }

        system_text = REBUTTAL_PROMPTS["fundamental"]["system"]
        user_text   = REBUTTAL_PROMPTS["fundamental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text


    def _build_messages_revision(self,
                                my_lastest: Opinion,
                                others_latest: Dict[str, Opinion],
                                received_rebuttals: List[Rebuttal],
                                stock_data: StockData) -> tuple[str, str]:
        ccy = (stock_data.currency or "USD").upper()

        me = {
            "agent_id": my_lastest.agent_id,
            "next_close": float(my_lastest.target.next_close),
            "reason": str(my_lastest.reason)[:2000],
        }
        peers = [{
            "agent_id": aid,
            "next_close": float(op.target.next_close),
            "reason": str(op.reason)[:2000],
        } for aid, op in (others_latest or {}).items()]
        feedback = [{
            "from": r.from_agent_id,
            "to":   r.to_agent_id,
            "stance": r.stance,
            "message": str(r.message)[:500],
        } for r in (received_rebuttals or [])]

        ctx = {
            "me": me,
            "peers": peers,
            "feedback": feedback,
            "fundamental_summary": stock_data.fundamental or {},
            "currency": ccy
        }

        system_text = REVISION_PROMPTS["fundamental"]["system"]
        user_text   = REVISION_PROMPTS["fundamental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text
    
    def _update_prompts(self, prompt_configs: Dict[str, str]) -> None:
        """프롬프트 설정 업데이트 (main.py에서 호출)"""
        global PREDICTER_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
        
        # predicter 프롬프트 업데이트
        if "predicter_system" in prompt_configs:
            PREDICTER_PROMPTS["fundamental"]["system"] = prompt_configs["predicter_system"]
        
        # rebuttal 프롬프트 업데이트
        if "rebuttal_system" in prompt_configs:
            REBUTTAL_PROMPTS["fundamental"]["system"] = prompt_configs["rebuttal_system"]
        
        # revision 프롬프트 업데이트
        if "revision_system" in prompt_configs:
            REVISION_PROMPTS["fundamental"]["system"] = prompt_configs["revision_system"]
    
    # ======================= ML 기능 =======================
    
    def search_data(self, ticker: str) -> str:
        """펀더멘털 데이터 수집"""
        if self.verbose:
            print(f"🔍 {ticker} 펀더멘털 데이터 수집 시작...")
        
        try:
            # ML 모듈을 사용한 데이터 수집
            if self.ml_manager:
                result = self.ml_manager.search_data(ticker)
                if result and self.verbose:
                    print(f"✅ {ticker} 펀더멘털 데이터 수집 완료")
                return result
            else:
                # ML 모듈이 없으면 기본 데이터 수집
                return self._generate_simulated_fundamental_data(ticker)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {ticker} 펀더멘털 데이터 수집 실패: {str(e)}")
            return None
    
    def _generate_simulated_fundamental_data(self, ticker: str) -> str:
        """시뮬레이션 펀더멘털 데이터 생성"""
        if self.verbose:
            print(f"🎲 {ticker} 시뮬레이션 펀더멘털 데이터 생성...")
        
        # 간단한 시뮬레이션 데이터 생성
        import pandas as pd
        from datetime import datetime, timedelta
        
        data = []
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2025, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'market_cap': 1000000000,
                'pe_ratio': 20.0,
                'pb_ratio': 2.0,
                'debt_to_equity': 0.5,
                'revenue_growth': 0.1,
                'profit_margin': 0.15,
                'roe': 0.12,
                'current_ratio': 2.0,
                'dividend_yield': 0.02
            })
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(data)
        
        # CSV 저장
        self.ensure_data_dir()
        filename = f"{ticker}_fundamental_data.csv"
        filepath = os.path.join("data", filename)
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"✅ 시뮬레이션 펀더멘털 데이터 저장 완료: {filepath}")
        return filepath
    
    def train_model(self, ticker: str) -> bool:
        """모델 학습 (ML 모듈 사용)"""
        if not self.use_ml_modules or not self.ml_manager:
            return False
        
        if self.verbose:
            print(f"🎯 {ticker} 펀더멘털 모델 학습 시작...")
        
        try:
            # ML 모듈을 사용한 모델 학습
            result = self.ml_manager.train_model(ticker)
            
            if result and self.verbose:
                print(f"✅ {ticker} 펀더멘털 모델 학습 완료")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {ticker} 펀더멘털 모델 학습 실패: {str(e)}")
            return False
    
    def predict_price(self, ticker: str) -> tuple:
        """가격 예측 (ML 모듈 사용)"""
        if not self.use_ml_modules or not self.ml_manager:
            return 0.0, 1.0
        
        try:
            # ML 모듈을 사용한 가격 예측
            result = self.ml_manager.predict_price(ticker)
            
            if result:
                prediction, uncertainty = result
                return prediction, uncertainty
            else:
                return 0.0, 1.0
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {ticker} 펀더멘털 예측 실패: {str(e)}")
            return 0.0, 1.0

