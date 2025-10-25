import json
import numpy as np
import pandas as pd
import yfinance as yf
import os
import torch
import torch.nn as nn
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from typing import Dict, List, Optional, Literal, Tuple
from agents.technical_modules import TechnicalModuleManager
from prompts import SEARCHER_PROMPTS, PREDICTER_PROMPTS, OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
# from agents.technical_modules import TechnicalModuleManager

class TechnicalMLModel(nn.Module):
    """기술적 분석 에이전트 (TCN 기반)"""
    
    def __init__(self, input_size: int = 14, hidden_size: int = 64, output_size: int = 1):
        super(TechnicalMLModel, self).__init__()
        
        # Temporal Convolutional Network
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size//2, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        
        # Global average pooling
        x = torch.mean(x, dim=2)  # (batch_size, hidden_size//2)
        
        x = self.fc(x)
        return x

class TechnicalAgent(BaseAgent):
    def __init__(self, 
                 agent_id: str = "TechnicalAgent",
                 use_ml_modules: bool = False,
                 fred_api_key: Optional[str] = None,
                 model_path: Optional[str] = None,
                 **kwargs):
        super().__init__(agent_id=agent_id, use_ml_modules=use_ml_modules, model_path=model_path, **kwargs)
        
        # ML 모듈 설정
        if self.use_ml_modules:
            self.ml_manager = TechnicalModuleManager(
                use_ml_searcher=True,
                use_ml_predictor=True,
                fred_api_key=fred_api_key or os.getenv('FRED_API_KEY'),
                model_path=model_path or "model_artifacts/final_best.keras"
            )
        else:
            self.ml_manager = None
    
    # ------------------------------------------------------------------
    # 1) 데이터 수집 
    # ------------------------------------------------------------------
    def searcher(self, ticker: str) -> StockData:
        df = yf.download(ticker, period="5d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()

        schema_tech = {
            "type": "object",
            "properties": {
                "trend":    {"type": "string", "enum": ["UP", "DOWN", "SIDEWAYS"]},
                "strength": {"type": "number"},
                "signals":  {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "summary":  {"type": "string"},
            },
            "required": ["trend", "strength", "signals", "evidence", "summary"],
            "additionalProperties": False,
        }

        sys_text = SEARCHER_PROMPTS["technical"]["system"]
        user_text = SEARCHER_PROMPTS["technical"]["user_template"].format(
            ticker=ticker, 
            current_price=last_price, 
            currency=currency
        )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            schema_tech
        )

        # ML 모듈 사용 여부에 따른 데이터 수집
        if self.use_ml_modules and self.ml_manager:
            # ML 모듈을 사용한 향상된 기술적 분석 데이터 수집
            ml_technical_data = self.ml_manager.get_enhanced_technical_data(ticker, last_price)
            
            # ML 결과를 기술적 데이터에 추가
            parsed["ml_signals"] = ml_technical_data.get('signals', {})
            parsed["ml_confidence"] = ml_technical_data.get('confidence', 0.0)
            parsed["ml_indicators"] = ml_technical_data.get('indicators', {})
            
            # ML 결과를 GPT 프롬프트에 포함하여 재분석
            ml_context = f"""
ML 모델 분석 결과:
- 기술적 신호: {ml_technical_data.get('signals', {})}
- 신뢰도: {ml_technical_data.get('confidence', 0.0):.2f}
- 수집된 뉴스: {ml_technical_data.get('news_count', 0)}개
- 기술적 지표: RSI, MA, 볼린저밴드 등 계산 완료
"""

            # ML 컨텍스트를 포함한 재분석
            user_text_with_ml = SEARCHER_PROMPTS["technical"]["user_template"].format(
                ticker=ticker, 
                current_price=last_price, 
                currency=currency
            ) + f"\n\n{ml_context}"

            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text_with_ml),
                schema_tech
            )

        self.stockdata = StockData(
            sentimental={},
            fundamental={},
            technical=parsed,
            last_price=last_price,
            currency=currency
        )
        self.current_ticker = ticker  # 현재 티커 저장
        return self.stockdata
    # ------------------------------------------------------------------
    # 2) 1차 예측 (LLM-only)
    #    - 기술 요약(트렌드/강도/신호) + 현재가 앵커로 next_close 산출
    # ------------------------------------------------------------------
    def predicter(self, stock_data: StockData) -> Target:
        # 현재 가격 정보 가져오기
        ticker = getattr(self, 'current_ticker', 'UNKNOWN')
        df = yf.download(ticker, period="1d", interval="1d")
        last_price = df["Close"].dropna().iloc[-1].item()
        info = yf.Ticker(ticker).info
        currency = (info.get("currency") or "USD").upper()
        
        # 기술적 분석가 특성: 공격적, 현재가 대비 ±15% 범위
        min_price = last_price * 0.85
        max_price = last_price * 1.15
        
        ctx = {
            "technical_summary": stock_data.technical,
            "current_price": last_price,
            "currency": currency,
            "prediction_range": f"{min_price:.2f} - {max_price:.2f} {currency}",
            "agent_character": "공격적인 기술적 분석가로서 차트 패턴과 모멘텀에 기반한 적극적인 예측을 제공합니다."
            }
        sys_text = PREDICTER_PROMPTS["technical"]["system"]
        user_text = PREDICTER_PROMPTS["technical"]["user_template"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            self.schema_obj_opinion
        )
        return Target(next_close=float(parsed.get("next_close", 0.0)))
    
    # ------------------------------------------------------------------
    # 3) Opinion 메시지 빌드 (기술 관점)
    # ------------------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> tuple[str, str]:
        t = getattr(self, "_last_ticker", "UNKNOWN")
        ccy = (stock_data.currency or "USD").upper()
        last = float(stock_data.last_price or 0.0)

        ctx = {
            "ticker": t,
            "currency": ccy,
            "last_price": last,
            "technical_summary": stock_data.technical or {},
            "our_prediction": float(target.next_close),
        }

        system_text = OPINION_PROMPTS["technical"]["system"]
        user_text   = OPINION_PROMPTS["technical"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text

    # ------------------------------------------------------------------
    # 4) Rebuttal/Revision (기술 관점 문구)
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
            "technical_summary": stock_data.technical or {},
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

        system_text = REBUTTAL_PROMPTS["technical"]["system"]
        user_text   = REBUTTAL_PROMPTS["technical"]["user"].format(
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
            "technical_summary": stock_data.technical or {},
            "currency": ccy
        }

        system_text = REVISION_PROMPTS["technical"]["system"]
        user_text   = REVISION_PROMPTS["technical"]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
        return system_text, user_text
    
    def _update_prompts(self, prompt_configs: Dict[str, str]) -> None:
        """프롬프트 설정 업데이트 (main.py에서 호출)"""
        global PREDICTER_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
        
        # predicter 프롬프트 업데이트
        if "predicter_system" in prompt_configs:
            PREDICTER_PROMPTS["technical"]["system"] = prompt_configs["predicter_system"]
        
        # rebuttal 프롬프트 업데이트
        if "rebuttal_system" in prompt_configs:
            REBUTTAL_PROMPTS["technical"]["system"] = prompt_configs["rebuttal_system"]
        
        # revision 프롬프트 업데이트
        if "revision_system" in prompt_configs:
            REVISION_PROMPTS["technical"]["system"] = prompt_configs["revision_system"]
    
    # ======================= ML 기능 =======================
    
    def create_ml_model(self):
        """ML 모델 생성"""
        return TechnicalMLModel()
    
    def search_data(self, ticker: str) -> str:
        """기술적 데이터 수집"""
        if self.verbose:
            print(f"🔍 {ticker} 기술적 데이터 수집 시작...")
        
        try:
            # ML 모듈을 사용한 데이터 수집
            if self.ml_manager:
                # 현재 가격 가져오기
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                current_price = hist["Close"].iloc[-1] if not hist.empty else 100.0
                
                # ML 모듈을 사용한 향상된 데이터 수집
                result = self.ml_manager.get_enhanced_technical_data(ticker, current_price)
                
                # CSV 파일로 저장
                self.ensure_data_dir()
                filename = f"{ticker}_technical_data.csv"
                filepath = os.path.join("data", filename)
                
                # 결과를 DataFrame으로 변환하여 저장
                import pandas as pd
                df = pd.DataFrame([result])
                df.to_csv(filepath, index=False)
                
                if self.verbose:
                    print(f"✅ {ticker} 기술적 데이터 수집 완료")
                return filepath
            else:
                # ML 모듈이 없으면 기본 데이터 수집
                return self._generate_simulated_technical_data(ticker)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {ticker} 기술적 데이터 수집 실패: {str(e)}")
            return self._generate_simulated_technical_data(ticker)
    
    def _generate_simulated_technical_data(self, ticker: str) -> str:
        """시뮬레이션 기술적 데이터 생성"""
        if self.verbose:
            print(f"🎲 {ticker} 시뮬레이션 기술적 데이터 생성...")
        
        technical_data = []
        start_date = pd.Timestamp('2022-01-01')
        end_date = pd.Timestamp('2025-12-31')
        
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
                'ticker': ticker,
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
            current_date += pd.Timedelta(days=1)
        
        # DataFrame 생성
        df = pd.DataFrame(technical_data)
        
        # CSV 저장
        self.ensure_data_dir()
        filename = f"{ticker}_technical_data.csv"
        filepath = os.path.join("data", filename)
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"✅ 시뮬레이션 기술적 데이터 저장 완료: {filepath}")
        return filepath
    
    def train_model(self, ticker: str) -> bool:
        """모델 학습 (ML 모듈 사용)"""
        if not self.use_ml_modules or not self.ml_manager:
            return False
        
        if self.verbose:
            print(f"🎯 {ticker} 기술적 모델 학습 시작...")
        
        try:
            # ML 모듈을 사용한 모델 학습
            result = self.ml_manager.train_model(ticker)
            
            if result and self.verbose:
                print(f"✅ {ticker} 기술적 모델 학습 완료")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ {ticker} 기술적 모델 학습 실패: {str(e)}")
            return False
    
    def _prepare_technical_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """기술적 데이터 전처리"""
        try:
            # 특성 선택
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'rsi', 'macd',
                'bollinger_upper', 'bollinger_lower', 'atr', 'volume_sma'
            ]
            
            # 결측값 처리
            df = df.dropna()
            
            # 특성과 타겟 분리
            X = df[feature_columns].values
            y = df['close'].values
            
            # 정규화
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 시퀀스 데이터 생성 (30일 윈도우)
            sequence_length = 30
            X_seq, y_seq = [], []
            
            for i in range(sequence_length, len(X_scaled)):
                X_seq.append(X_scaled[i-sequence_length:i])
                y_seq.append(y[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # 타겟 정규화
            y_scaler = MinMaxScaler()
            y_seq_scaled = y_scaler.fit_transform(y_seq.reshape(-1, 1)).flatten()
            
            # 스케일러 저장
            self.scaler = {'X': scaler, 'y': y_scaler}
            
            return X_seq, y_seq_scaled
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 기술적 데이터 전처리 실패: {str(e)}")
            return None, None
    
    def _train_technical_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """기술적 모델 학습"""
        try:
            from sklearn.model_selection import train_test_split
            import torch.optim as optim
            
            # 데이터 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # PyTorch 텐서로 변환
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            
            # 데이터로더 생성
            from torch.utils.data import DataLoader, TensorDataset
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 손실함수, 옵티마이저 설정
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.ml_model.parameters(), lr=0.001)
            
            # 학습
            num_epochs = 50
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                # 훈련
                self.ml_model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.ml_model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 검증
                self.ml_model.eval()
                with torch.no_grad():
                    val_outputs = self.ml_model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if epoch % 10 == 0 and self.verbose:
                    print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 기술적 모델 학습 실패: {str(e)}")
            return False
    
    def predict_price(self, ticker: str) -> Tuple[float, float]:
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
                print(f"❌ {ticker} 기술적 예측 실패: {str(e)}")
            return 0.0, 1.0
