# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
from collections import defaultdict
import os, json, requests
from datetime import datetime
from dotenv import load_dotenv

from agents.macro_agent import MacroPredictor
from agents.macro_agent_with_shap_llm import FundamentalForecastAgent
from debate_ver3_tmp.agents.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from debate_ver3_tmp.config.agents import agents_info, dir_info
from debate_ver3_tmp.core.data_set import build_dataset, load_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib

# -----------------------------
# 데이터 구조 정의
# -----------------------------
@dataclass
class Target:
    """예측 목표값 + 불확실성 정보 포함
    - next_close: 다음 거래일 종가 예측치
    - uncertainty: Monte Carlo Dropout 기반 예측 표준편차(σ)
    - confidence: 모델 신뢰도 β (정규화된 신뢰도; 선택적)
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    feature_cols: Optional[List[str]] = None
    importances: Optional[List[float]] = None

@dataclass
class Opinion:
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str

@dataclass
class RoundLog:
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    agent_id: str = ""
    ticker: str = ""
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    feature_cols: Optional[List[str]] = None
    last_price: Optional[float] = None
    technical: Optional[Dict] = None
    
    def __post_init__(self):
        if self.last_price is None:
            self.last_price = 100.0

# ===============================================================
# BaseAgent 클래스
# ===============================================================
class BaseAgent:
    """LLM 기반 Multi-Agent Debate 공통 클래스"""

    OPENAI_URL = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        agent_id: str,
        model: Optional[str] = None,
        preferred_models: Optional[List[str]] = None,
        temperature: float = 0.2,
        verbose: bool = False,
        need_training: bool = True,
        data_dir: str = dir_info["data_dir"],
        model_dir: str = dir_info["model_dir"],
        ticker: str = "TSLA",

    ):

        load_dotenv()
        self.agent_id = agent_id # 에이전트 식별자
        self.model = model # 모델 이름
        self.temperature = temperature # Temperature 설정 
        self.verbose = verbose            # 디버깅 모드
        self.need_training = need_training # 모델 학습 필요 여부
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.ticker = ticker
        self.scaler = DataScaler(agent_id)
        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [
                m for m in self.preferred_models if m != model
            ]

        # API 키 로드
        self.api_key = os.getenv("CAPSTONE_OPENAI_API")
        if not self.api_key:
            raise RuntimeError("환경변수 CAPSTONE_OPENAI_API가 설정되지 않았습니다.")

        # 공통 헤더
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 상태값
        self.stockdata: Optional[StockData] = None
        self.opinions: List[Opinion] = []
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)

        # JSON Schema
        self.schema_obj_opinion = {
            "type": "object",
            "properties": {
                "next_close": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["next_close", "reason"],
            "additionalProperties": False,
        }
        self.schema_obj_rebuttal = {
            "type": "object",
            "properties": {
                "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                "message": {"type": "string"},
            },
            "required": ["stance", "message"],
            "additionalProperties": False,
        }

    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """
        preprocessing.py 기반 데이터 검색기
        - CSV가 없을 경우 build_dataset()으로 자동 생성
        - 마지막 window 시퀀스를 torch.tensor로 반환
        """

        if ticker is None:
            ticker = self.ticker
            
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_id}_dataset.csv")

        # 데이터셋이 존재하지 않으면 생성
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {self.agent_id} dataset not found. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir)

        # CSV에서 데이터셋 로드
        X, y, feature_cols = load_dataset(ticker, agent_id=self.agent_id, save_dir=self.data_dir)

        # StockData 인스턴스 생성해서 self.stockdata에 저장 (Load_csv_dataset 결과 반영)
        self.stockdata = StockData()
        self.stockdata.agent_id = self.agent_id
        self.stockdata.ticker = ticker
        self.stockdata.X = X
        self.stockdata.y = y
        self.stockdata.feature_cols = feature_cols
        
        # 가장 최근 window 데이터만 사용
        X_latest = X[-1:]  # shape: (1, window_size, n_features)
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)
        
        # 실제 현재 가격 저장 (yfinance로 최신 Close 가격 가져오기)
        import yfinance as yf
        try:
            data = yf.download(ticker, period="1d", interval="1d")
            self.stockdata.last_price = float(data['Close'].iloc[-1])
        except:
            self.stockdata.last_price = 100.0  # 기본값

        return X_tensor

    def predict(self, X, n_samples: int = 30, current_price: float = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성 계산 (공통)
        모든 Agent에서 사용 가능
        """

        if self.agent_id == 'MacroAgent':
            macro_predictor = MacroPredictor(
                base_date=datetime.today(),
                window=40,
                ticker = self.ticker  # ✅ 리스트 형태로 단일 티커 지정
            )
            pred_prices, target, _ = macro_predictor.run_prediction()
            return target

        # 1. 스케일러 로드
        self.scaler.load(self.ticker)

        # 2. 입력 텐서 변환
        if isinstance(X, np.ndarray):
            # transform()은 (X_t, y_t) 형태로 반환 → X_t만 사용
            X_scaled, _ = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        # 3. 모델 선택 (self.model 또는 self)
        model = getattr(self, "model", None)
        if model is None:
            model = self

        if not hasattr(model, "parameters"):
            raise AttributeError(f"{model} has no parameters()")

        # 4. 디바이스 설정
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        X_tensor = X_tensor.to(device)

        # 5. Dropout 활성화 (Monte Carlo Dropout)
        model.train()

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor).cpu().numpy().flatten()
                preds.append(y_pred)

        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        # 6. 정규화 복원 (상승/하락율 복원)
        if hasattr(self.scaler, 'y_scaler') and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)
        else:
            # 스케일러가 없으면 원본 값 사용
            pass

        # 7. 상승/하락율을 실제 가격으로 변환
        if current_price is None:
            # self.stockdata에서 실제 현재 가격 가져오기
            current_price = getattr(self.stockdata, 'last_price', 100.0)
        
        # 기존: 절대 종가 예측
        # return Target(
        #     next_close=float(mean_pred[-1]),
        #     uncertainty=float(std_pred[-1]),
        #     confidence=float(confidence[-1])
        # )
        
        # 새로운: 상승/하락율 예측
        return_rate = float(mean_pred[-1])
        predicted_price = self.scaler.convert_return_to_price(return_rate, current_price) if hasattr(self.scaler, 'convert_return_to_price') else current_price * (1 + return_rate)
        
        # confidence 계산
        confidence = 1 / (std_pred + 1e-8)

        return Target(
            next_close=float(predicted_price),
            uncertainty=float(std_pred[-1]),
            confidence=float(confidence[-1])
        )



    # -----------------------------
    # 메인 워크플로
    # -----------------------------
    def reviewer_draft(self, stock_data, target: Target) -> Opinion:
        """(1) searcher → (2) predicter → (3) LLM(JSON Schema)로 reason 생성 → Opinion 반환"""

        # 1) 데이터 수집
        X = self.searcher(self.ticker)

        # 2) 예측값 생성
        target = self.predict(X)

        # 3) LLM 호출(reason 생성)
        if self.agent_id == 'MacroAgent':
            m_agent = FundamentalForecastAgent(agent_id = 'MacroAgent', ticker=self.ticker)
            total_json, opinions = m_agent.run()
            return opinions

        else:
            sys_text, user_text = self._build_messages_opinion(self.stockdata, target)
            msg_sys = self._msg("system", sys_text)
            msg_user = self._msg("user",   user_text)

            parsed = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj_opinion)
            reason = parsed.get("reason") or "(사유 생성 실패: 미입력)"

            prompt_set = OPINION_PROMPTS.get(self.agent_id, OPINION_PROMPTS[self.agent_id])

            #ctx
            context = json.dumps({
                "agent_id": self.agent_id,
                "predicted_next_close": round(target.next_close, 3),
                "uncertainty_sigma": round(target.uncertainty or 0.0, 4),
                "confidence_beta": round(target.confidence or 0.0, 4),
                "latest_data": str(stock_data)
            }, ensure_ascii=False, indent=2)

            sys_text = prompt_set["system"]
            user_text = prompt_set["user"].format(context=context)

            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
            )

            reason = parsed.get("reason", "(사유 생성 실패)")

            # 4) Opinion 기록/반환 (항상 최신 값 append)
            self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason))

            # 최신 오피니언 반환
            return self.opinions[-1]

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""
        prompt_set = REBUTTAL_PROMPTS.get(self.agent_id)

        context = json.dumps({
            "self_agent": my_opinion.agent_id,
            "self_next_close": my_opinion.target.next_close,
            "self_confidence": my_opinion.target.confidence,
            "self_uncertainty": my_opinion.target.uncertainty,
            "self_reason": my_opinion.reason,
            "other_agent": other_opinion.agent_id,
            "other_next_close": other_opinion.target.next_close,
            "other_confidence": other_opinion.target.confidence,
            "other_uncertainty": other_opinion.target.uncertainty,
            "other_reason": other_opinion.reason
        }, ensure_ascii=False, indent=2)

        sys_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=context)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"}
                },
                "required": ["stance", "message"]
            }
        )

        self.rebuttals.append(Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        ))

        return self.rebuttals[-1]

    def reviewer_revise(
        self,
        revised_target: Target,
        old_opinion: Opinion,
        rebuttals: list,
        others: list,
        X_input=None,
        fine_tune: bool = True,
        lr: float = 1e-4,
        epochs: int = 3,
    ):
        """
        Monte Carlo 기반 β-weighted revised_target을 받아
        - 모델 파라미터 업데이트(fine-tuning)
        - 수정된 수치에 대한 LLM reasoning 생성
        - Opinion 업데이트까지 한 번에 수행
        """

        # --------------------------------------------
        # 1️⃣ Fine-tuning 단계 (모델 파라미터 업데이트)
        # --------------------------------------------
        if fine_tune and hasattr(self, "model") and X_input is not None:
            try:
                device = next(self.model.parameters()).device
                X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
                y_tensor = torch.tensor([[revised_target.next_close]], dtype=torch.float32).to(device)

                self.model.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()

                for _ in range(epochs):
                    optimizer.zero_grad()
                    pred = self.model(X_tensor)
                    loss = criterion(pred, y_tensor)
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
            except Exception as e:
                loss_value = None
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")
        else:
            loss_value = None

        # --------------------------------------------
        # 2️⃣ LLM Reasoning 생성 단계
        # --------------------------------------------
        prompt_set = REVISION_PROMPTS.get(self.agent_id, REVISION_PROMPTS.get("default"))

        context = json.dumps({
            "agent_id": self.agent_id,
            "new_next_close": revised_target.next_close,
            "my_reason": old_opinion.reason if old_opinion else "(이전 의견 없음)",
            "my_confidence": getattr(old_opinion.target, "confidence", None) if old_opinion else None,
            "my_uncertainty": getattr(old_opinion.target, "uncertainty", None) if old_opinion else None,
            "others": [
                {
                    "agent": o.agent_id,
                    "reason": o.reason,
                    "confidence": getattr(o.target, "confidence", None),
                    "uncertainty": getattr(o.target, "uncertainty", None),
                }
                for o in others
            ],
            "rebuttals": [
                {"from": r.from_agent_id, "stance": r.stance, "message": r.message}
                for r in rebuttals
            ],
            "fine_tune_loss": loss_value,
        }, ensure_ascii=False, indent=2)

        sys_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=context)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False
            },
        )

        # --------------------------------------------
        # 3️⃣ 수정된 Opinion 기록
        # --------------------------------------------
        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=revised_target,
            reason=revised_reason,
        )
        self.opinions.append(revised_opinion)

        # --------------------------------------------
        # 4️⃣ 디버깅 / 로깅용 출력
        # --------------------------------------------
        if self.verbose:
            print(
                f"[{self.agent_id}] revise 완료 → "
                f"new_target={revised_target.next_close:.4f}, "
                f"σ={getattr(revised_target, 'uncertainty', None)}, "
                f"β={getattr(revised_target, 'confidence', None)}, "
                f"loss={loss_value:.6f if loss_value else 'N/A'}"
            )

        return self.opinions[-1]

    # -----------------------------
    # 공통 유틸
    # -----------------------------
    def _p(self, msg: str):
        if self.verbose:
            print(f"[{self.agent_id}] {msg}")

    @staticmethod
    def _msg(role: str, text: str) -> dict:
        return {"role": role, "content": [{"type": "input_text", "text": text}]}

    # -----------------------------
    # 구현 필요 함수 (추상)
    # -----------------------------

    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """LLM(system/user) 메시지 생성(구현 필요)"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_opinion method")
    
    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        """LLM(system/user) 메시지 생성(구현 필요)"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_rebuttal method")
    
    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치 로드"""

        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        
        if not os.path.exists(model_path):
            print(f"⚠️ 모델 파일 없음: {model_path}")
            return False

        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ {self.agent_id} 모델 로드 완료 ({model_path})")

    def pretrain(self):
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]

        X, y, cols = load_dataset(self.ticker, self.agent_id, save_dir=self.data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.scaler.fit_scalers(X_train, y_train)
        self.scaler.save(self.ticker)

        X_train, y_train = map(torch.tensor, self.scaler.transform(X_train, y_train))
        X_train, y_train = X_train.float(), y_train.float()

        model = self if hasattr(self, 'forward') else self.model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")

        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"✅ {self.agent_id} model saved.\n✅ pretraining finished.\n")

    # -----------------------------
    # OpenAI API 호출
    # -----------------------------
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
        payload_base = {
            "input": [msg_sys, msg_user],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Response",
                    "strict": True,
                    "schema": schema_obj,
                }
            },
            "temperature": self.temperature,
        }
        last_err = None
        for model in self.preferred_models:
            payload = dict(payload_base, model=model)
            try:
                r = requests.post(self.OPENAI_URL, headers=self.headers, json=payload, timeout=120)
                if r.ok:
                    data = r.json()
                    # 1) output_text 우선 사용
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}  # JSON 실패 시 원문 텍스트 보존
                    # 2) output 배열에서 텍스트 모으기
                    out = data.get("output")
                    if isinstance(out, list) and out:
                        texts = []
                        for blk in out:
                            for c in blk.get("content", []):
                                if "text" in c:
                                    texts.append(c["text"])
                        joined = "\n".join(t for t in texts if t)
                        if joined.strip():
                            try:
                                return json.loads(joined)
                            except Exception:
                                return {"reason": joined}
                    # 비정상 응답
                    return {}
                # 400/404는 다음 모델로 폴백
                if r.status_code in (400, 404):
                    last_err = (r.status_code, r.text)
                    continue
                # 기타 에러는 즉시 예외
                r.raise_for_status()
            except Exception as e:
                self._p(f"⚠️ 모델 {model} 실패: {e}")
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")
    # -----------------------------------------
    # 🔹 추가: Monte Carlo Dropout 기반 불확실성 추정
    # -----------------------------------------
    def evaluate(self, ticker: str = None):
        """검증 데이터로 성능 평가"""
        if ticker is None:
            ticker = self.ticker
            
        # 데이터 로드
        X, y, feature_cols = load_dataset(ticker, agent_id=self.agent_id, save_dir=self.data_dir)
        
        # 시계열 분할 (80% 훈련, 20% 검증)
        split_idx = int(len(X) * 0.8)
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        # 스케일러 로드
        self.scaler.load(ticker)
        
        # 검증 데이터 예측
        predictions = []
        actual_returns = []
        
        for i in range(len(X_val)):
            X_input = X_val[i:i+1]
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            
            # 예측
            with torch.no_grad():
                pred_return = self(X_tensor).item()
                predictions.append(pred_return)
                actual_returns.append(y_val[i, 0])
        
        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)
        
        # 성능 지표 계산
        mae = np.mean(np.abs(predictions - actual_returns))
        rmse = np.sqrt(np.mean((predictions - actual_returns) ** 2))
        correlation = np.corrcoef(predictions, actual_returns)[0, 1]
        
        # 방향 정확도
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual_returns)
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'direction_accuracy': direction_accuracy,
            'n_samples': len(predictions)
        }

class DataScaler:
    """학습/추론용 정규화 유틸리티 (BaseAgent 내부용)"""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.save_dir = dir_info["scaler_dir"]
        self.x_scaler = agents_info[self.agent_id]["x_scaler"]
        self.y_scaler = agents_info[self.agent_id]["y_scaler"]

    def fit_scalers(self, X_train, y_train):
        ScalerMap = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "None": None,
        }
        Sx = ScalerMap[self.x_scaler]
        Sy = ScalerMap[self.y_scaler]

        # ✅ 3D 입력 (samples, seq_len, features) → 2D로 변환
        n_samples, seq_len, n_feats = X_train.shape
        X_2d = X_train.reshape(-1, n_feats)
        self.x_scaler = Sx().fit(X_2d) if Sx else None
        self.y_scaler = Sy().fit(y_train.reshape(-1, 1)) if Sy else None

    def transform(self, X, y=None):
        # ✅ 3D 입력 (samples, seq_len, features) → 2D로 변환
        if X.ndim == 3:
            n_samples, seq_len, n_feats = X.shape
            X_2d = X.reshape(-1, n_feats)
            X_t = self.x_scaler.transform(X_2d).reshape(n_samples, seq_len, n_feats)
        else:
            X_t = self.x_scaler.transform(X) if self.x_scaler else X

        y_t = (
            self.y_scaler.transform(y.reshape(-1, 1)).flatten()
            if (self.y_scaler and y is not None)
            else y
        )
        return X_t, y_t


    def inverse_y(self, y_pred):
        if self.y_scaler:
            # numpy 배열로 변환
            if isinstance(y_pred, (list, tuple)):
                y_pred = np.array(y_pred)
            return self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred
    
    def convert_return_to_price(self, return_rate, current_price):
        """상승/하락율을 실제 가격으로 변환"""
        return current_price * (1 + return_rate)
    
    def evaluate(self, ticker: str = None):
        """검증 데이터로 성능 평가"""
        if ticker is None:
            ticker = self.ticker
            
        # 데이터 로드
        X, y, feature_cols = load_dataset(ticker, agent_id=self.agent_id, save_dir=self.data_dir)
        
        # 시계열 분할 (80% 훈련, 20% 검증)
        split_idx = int(len(X) * 0.8)
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        # 스케일러 로드
        self.scaler.load(ticker)
        
        # 검증 데이터 예측
        predictions = []
        actual_returns = []
        
        for i in range(len(X_val)):
            X_input = X_val[i:i+1]
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            
            # 예측
            with torch.no_grad():
                pred_return = self(X_tensor).item()
                predictions.append(pred_return)
                actual_returns.append(y_val[i, 0])
        
        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)
        
        # 성능 지표 계산
        mae = np.mean(np.abs(predictions - actual_returns))
        rmse = np.sqrt(np.mean((predictions - actual_returns) ** 2))
        correlation = np.corrcoef(predictions, actual_returns)[0, 1]
        
        # 방향 정확도
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual_returns)
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'direction_accuracy': direction_accuracy,
            'n_samples': len(predictions)
        }

    def save(self, ticker):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.x_scaler:
            joblib.dump(
                self.x_scaler,
                os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_xscaler.pkl"),
            )
        if self.y_scaler:
            joblib.dump(
                self.y_scaler,
                os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_yscaler.pkl"),
            )

    def load(self, ticker):
        x_path = os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_xscaler.pkl")
        y_path = os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_yscaler.pkl")
        if os.path.exists(x_path):
            self.x_scaler = joblib.load(x_path)
        if os.path.exists(y_path):
            self.y_scaler = joblib.load(y_path)
