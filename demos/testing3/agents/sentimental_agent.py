import json
import numpy as np
import pandas as pd
import yfinance as yf
import os
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple
from agents.sentimental_modules import SentimentalModuleManager
from prompts import SEARCHER_PROMPTS, PREDICTER_PROMPTS, OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
# from agents.sentimental_modules import SentimentalModuleManager

class SentimentalAgent(BaseAgent):
    def __init__(self, 
                 agent_id: str = "SentimentalAgent",
                 use_ml_modules: bool = False,
                 finnhub_api_key: Optional[str] = None,
                 model_path: Optional[str] = None,
                 **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        
        # ML 모듈 설정
        if self.use_ml_modules:
            self.ml_manager = SentimentalModuleManager(
                use_ml_searcher=True,
                use_ml_predictor=True,
                finnhub_api_key=finnhub_api_key or os.getenv('FINNHUB_API_KEY'),
                model_path=model_path or "mlp_stock_model.pt"
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

        # ML 모듈 사용 여부에 따른 데이터 수집
        if self.use_ml_modules and self.ml_manager:
            # ML 모듈을 사용한 향상된 센티멘탈 데이터 수집
            ml_sentimental_data = self.ml_manager.get_enhanced_sentimental_data(ticker, last_price)
            
            # 기존 GPT 기반 분석과 ML 결과를 결합
            schema_sent = {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "positives": {"type": "array", "items": {"type": "string"}},
                    "negatives": {"type": "array", "items": {"type": "string"}},
                    "evidence":  {"type": "array", "items": {"type": "string"}},
                    "summary":   {"type": "string"},
                },
                "required": ["sentiment", "positives", "negatives", "evidence", "summary"],
                "additionalProperties": False,
            }

            # ML 결과를 GPT 프롬프트에 포함
            ml_context = f"""
ML 모델 분석 결과:
- 예측 종가: {ml_sentimental_data.get('ml_prediction', 'N/A')}
- 신뢰도: {ml_sentimental_data.get('ml_confidence', 0.0):.2f}
- 수집된 뉴스: {len(ml_sentimental_data.get('evidence', []))}개
- ML 센티멘탈: {ml_sentimental_data.get('sentiment', 'neutral')}
"""

            sys_text = SEARCHER_PROMPTS["sentimental"]["system"]
            user_text = SEARCHER_PROMPTS["sentimental"]["user_template"].format(
                ticker=ticker, 
                current_price=last_price, 
                currency=currency
            ) + f"\n\n{ml_context}"

            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                schema_sent
            )
            
            # ML 결과를 센티멘탈 데이터에 추가
            parsed["ml_prediction"] = ml_sentimental_data.get('ml_prediction')
            parsed["ml_confidence"] = ml_sentimental_data.get('ml_confidence')
            parsed["ml_evidence"] = ml_sentimental_data.get('evidence', [])
            
        else:
            # 기존 GPT 기반 분석만 사용
            schema_sent = {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "positives": {"type": "array", "items": {"type": "string"}},
                    "negatives": {"type": "array", "items": {"type": "string"}},
                    "evidence":  {"type": "array", "items": {"type": "string"}},
                    "summary":   {"type": "string"},
                },
                "required": ["sentiment", "positives", "negatives", "evidence", "summary"],
                "additionalProperties": False,
            }

            sys_text = SEARCHER_PROMPTS["sentimental"]["system"]
            user_text = SEARCHER_PROMPTS["sentimental"]["user_template"].format(
                ticker=ticker, 
                current_price=last_price, 
                currency=currency
            )

            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                schema_sent
            )

        self.stockdata = StockData(
            sentimental=parsed,
            fundamental={},
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
        
        # 센티멘탈 분석가 특성: 중립적, 현재가 대비 ±10% 범위
        min_price = last_price * 0.90
        max_price = last_price * 1.10
        
        # ML 모듈 사용 여부에 따른 예측
        if self.use_ml_modules and self.ml_manager and stock_data.sentimental.get('ml_prediction'):
            # ML 예측값이 있는 경우, GPT와 ML 결과를 결합
            ml_prediction = stock_data.sentimental.get('ml_prediction')
            ml_confidence = stock_data.sentimental.get('ml_confidence', 0.0)
            
            # ML 예측값이 범위 내에 있는지 확인하고 조정
            if ml_prediction < min_price:
                ml_prediction = min_price
            elif ml_prediction > max_price:
                ml_prediction = max_price
            
            # ML 신뢰도에 따른 가중치 적용
            ml_weight = min(ml_confidence * 0.5, 0.3)  # 최대 30% 가중치
            gpt_weight = 1.0 - ml_weight
            
            ctx = {
                "sentimental_summary": stock_data.sentimental,
                "current_price": last_price,
                "currency": currency,
                "prediction_range": f"{min_price:.2f} - {max_price:.2f} {currency}",
                "agent_character": "중립적인 센티멘탈 분석가로서 시장 심리와 여론에 기반한 균형 잡힌 예측을 제공합니다.",
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "ml_weight": ml_weight,
                "gpt_weight": gpt_weight
            }
            
            sys_text = PREDICTER_PROMPTS["sentimental"]["system"] + """
            
            추가로 ML 모델의 예측값과 신뢰도가 제공됩니다. 
            ML 예측값을 참고하되, 최종 예측은 당신의 전문가적 판단을 우선시하세요.
            """
            
            user_text = PREDICTER_PROMPTS["sentimental"]["user_template"].format(
                context=json.dumps(ctx, ensure_ascii=False)
            )
            
            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                self.schema_obj_opinion
            )
            
            gpt_prediction = float(parsed.get("next_close", last_price))
            
            # ML과 GPT 예측을 가중 평균으로 결합
            final_prediction = (gpt_prediction * gpt_weight) + (ml_prediction * ml_weight)
            
            # 범위 내로 클리핑
            final_prediction = max(min_price, min(max_price, final_prediction))
            
            return Target(next_close=final_prediction)
            
        else:
            # 기존 GPT 기반 예측만 사용
            ctx = {
                "sentimental_summary": stock_data.sentimental,
                "current_price": last_price,
                "currency": currency,
                "prediction_range": f"{min_price:.2f} - {max_price:.2f} {currency}",
                "agent_character": "중립적인 센티멘탈 분석가로서 시장 심리와 여론에 기반한 균형 잡힌 예측을 제공합니다."
            }
            
            sys_text = PREDICTER_PROMPTS["sentimental"]["system"]
            user_text = PREDICTER_PROMPTS["sentimental"]["user_template"].format(
                context=json.dumps(ctx, ensure_ascii=False)
            )
            
            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                self.schema_obj_opinion
            )
            
            return Target(next_close=float(parsed.get("next_close", 0.0)))
    
    # ------------------------------------------------------------------
    # 3) LLM 메시지 빌드(Opinion): 다음날 종가와 근거를 JSON으로 요구
    #    - 시스템: 역할/출력형식 고정
    #    - 사용자: 컨텍스트(JSON 직렬화 가능 타입만)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_close": float(stock_data.last_price or 0.0),
            "signals": {k: float(v) for k, v in (stock_data.technical or {}).items()},
            "our_prediction": float(target.next_close),
            "format_rule": f"숫자는 소수 {decimals}자리, 통화 {ccy}"
        }

        system_text = OPINION_PROMPTS["sentimental"]["system"]
        user_text   = OPINION_PROMPTS["sentimental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text


    # ------------------------------------------------------------------
    # 4) LLM 메시지 빌드(Rebuttal): 내/상대 의견 비교 → REBUT/SUPPORT + message
    #    - 시스템: 출력키는 'stance'와 'message' (스키마와 일치)
    #    - 사용자: 숫자는 float로, 텍스트는 문자열로 제한
    # ------------------------------------------------------------------
    def _build_messages_rebuttal(self,
                                my_opinion: Opinion,
                                target_agent: str,
                                target_opinion: Opinion,
                                stock_data: StockData) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        ctx = {
            "ticker": t,
            "me": {
                "agent_id": self.agent_id,
                "next_close": round(float(my_opinion.target.next_close), decimals),
                "reason": str(my_opinion.reason)[:2000],
            },
            "other": {
                "agent_id": target_agent,
                "next_close": round(float(target_opinion.target.next_close), decimals),
                "reason": str(target_opinion.reason)[:2000],
            },
            "snapshot": {
                "last_price": float(stock_data.last_price or 0.0),
                "currency": ccy,
                "signals": {
                    "technical":   {k: float(v) for k, v in (stock_data.technical   or {}).items()},
                    "sentimental": (stock_data.sentimental or {}),
                    "fundamental": (stock_data.fundamental or {}),
                },
            },
        }

        system_text = REBUTTAL_PROMPTS["sentimental"]["system"]
        user_text   = REBUTTAL_PROMPTS["sentimental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text

    
    
    def _build_messages_revision(self,
                                my_lastest: Opinion,
                                others_latest: Dict[str, Opinion],
                                received_rebuttals: List[Rebuttal],
                                stock_data: StockData) -> tuple[str, str]:
        ccy = (stock_data.currency or "KRW").upper()
        decimals = 0 if ccy in ("KRW", "JPY") else 2

        me = {
            "agent_id": my_lastest.agent_id,
            "next_close": float(my_lastest.target.next_close),
            "reason": str(my_lastest.reason)[:2000],
        }
        peers = [{
            "agent_id": str(aid),
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
            "snapshot": {
                "last_price": float(stock_data.last_price or 0.0),
                "currency": ccy,
                "signals": {
                    "technical":   {k: float(v) for k, v in (stock_data.technical   or {}).items()},
                    "sentimental": (stock_data.sentimental or {}),
                    "fundamental": (stock_data.fundamental or {}),
                },
            },
        }

        system_text = REVISION_PROMPTS["sentimental"]["system"]
        user_text   = REVISION_PROMPTS["sentimental"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text

    
    # ------------------------------------------------------------------
    # RSI 계산: 단순 이동평균 버전(EMA 아님)
    # ------------------------------------------------------------------
    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(period).mean()
        avg_loss = down.rolling(period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        return 100 - (100 / (1 + rs))
    
    def _update_prompts(self, prompt_configs: Dict[str, str]) -> None:
        """프롬프트 설정 업데이트 (main.py에서 호출)"""
        global PREDICTER_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
        
        # predicter 프롬프트 업데이트
        if "predicter_system" in prompt_configs:
            PREDICTER_PROMPTS["sentimental"]["system"] = prompt_configs["predicter_system"]
        
        # rebuttal 프롬프트 업데이트
        if "rebuttal_system" in prompt_configs:
            REBUTTAL_PROMPTS["sentimental"]["system"] = prompt_configs["rebuttal_system"]
        
        # revision 프롬프트 업데이트
        if "revision_system" in prompt_configs:
            REVISION_PROMPTS["sentimental"]["system"] = prompt_configs["revision_system"]
    
    # ======================= ML 기능 =======================
    
    def search_data(self, ticker: str) -> str:
        """감정 분석 데이터 수집"""
        if self.verbose:
            print(f"🔍 {ticker} 감정 분석 데이터 수집 시작...")
        
        try:
            # ML 모듈을 사용한 데이터 수집
            if self.ml_manager:
                result = self.ml_manager.search_data(ticker)
                if result and self.verbose:
                    print(f"✅ {ticker} 감정 분석 데이터 수집 완료")
                return result
            else:
                # ML 모듈이 없으면 기본 데이터 수집
                return self._generate_simulated_sentimental_data(ticker)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {ticker} 감정 분석 데이터 수집 실패: {str(e)}")
            return None
    
    def _generate_simulated_sentimental_data(self, ticker: str) -> str:
        """시뮬레이션 감정 분석 데이터 생성"""
        if self.verbose:
            print(f"🎲 {ticker} 시뮬레이션 감정 분석 데이터 생성...")
        
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
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'analyst_rating': 3.0,
                'price_target': 100.0,
                'earnings_surprise': 0.0,
                'insider_trading': 0.0,
                'institutional_flow': 0.0,
                'options_sentiment': 0.0,
                'fear_greed_index': 50.0
            })
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(data)
        
        # CSV 저장
        self.ensure_data_dir()
        filename = f"{ticker}_sentimental_data.csv"
        filepath = os.path.join("data", filename)
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"✅ 시뮬레이션 감정 분석 데이터 저장 완료: {filepath}")
        return filepath
    
    def train_model(self, ticker: str) -> bool:
        """모델 학습 (ML 모듈 사용)"""
        if not self.use_ml_modules or not self.ml_manager:
            return False
        
        if self.verbose:
            print(f"🎯 {ticker} 감정 분석 모델 학습 시작...")
        
        try:
            # ML 모듈을 사용한 모델 학습
            result = self.ml_manager.train_model(ticker)
            
            if result and self.verbose:
                print(f"✅ {ticker} 감정 분석 모델 학습 완료")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {ticker} 감정 분석 모델 학습 실패: {str(e)}")
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
                print(f"❌ {ticker} 감정 분석 예측 실패: {str(e)}")
            return 0.0, 1.0
