# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple, Any
from dataclasses import field
from collections import defaultdict
import os, json, time, requests, yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from config.agents import agents_info, dir_info
from core.data_set import build_dataset, load_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import pandas as pd

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
    """에이전트 입력 원천 데이터(필요 시 자유 확장)
    - sentimental: 심리/커뮤니티/뉴스 스냅샷
    - fundamental: 재무/밸류에이션 요약
    - technical  : 가격/지표 스냅샷
    - last_price : 최신 종가
    - currency   : 통화코드
    """
    SentimentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    FundamentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    TechnicalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: Optional[str] = None

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
        ticker: str=None,
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
        self.window_size = agents_info[agent_id]["window_size"]
        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [
                m for m in self.preferred_models if m != model
            ]

        # API 키 로드
        self.api_key = os.getenv("CAPSTONE_OPENAI_API")
        if not self.api_key:
            self.api_key = ""

        # 공통 헤더
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 상태값
        self.stockdata: Optional[StockData] = None
        self.targets: List[Target] = []
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
        import yfinance as yf
        import pandas as pd

        agent_id = self.agent_id

        if ticker is None:
            ticker = self.ticker

        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")

        # 데이터셋이 없으면 자동 생성
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {agent_id} dataset not found. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir)

        # CSV 로드
        X, y, feature_cols = load_dataset(ticker, agent_id=agent_id, save_dir=self.data_dir)

        # StockData 초기화
        self.stockdata = StockData()

        # 최근 window
        X_latest = X[-1:]
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # DataFrame 변환
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)

        # 👉 컬럼별 리스트 저장
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}

        setattr(self.stockdata, agent_id, feature_dict)

        # 종가 및 통화 정보
        self.stockdata.ticker = ticker
        
        try:
            data = yf.download(ticker, period="1d", interval="1d")
            self.stockdata.last_price = float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"yfinance 오류 발생")


        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception as e:
            print(f"yfinance 오류 발생, 통화 기본값 사용: {e}")
            self.stockdata.currency = "USD"

        print(f"✅ {agent_id} StockData 생성 완료 ({ticker}, {self.stockdata.currency})")

        return X_tensor

    def predict(self, X, n_samples: int = 30, current_price: float = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성 계산 (공통)
        - 모델이 없으면 자동 로드, 그래도 없으면 pretrain 수행
        """
        # -----------------------------
        # 🔹 모델 준비 단계
        # -----------------------------
        if self.model is None or not hasattr(self.model, "parameters"):
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
            if os.path.exists(model_path):
                print(f"⚙️ {self.agent_id} 모델 자동 로드 시도...")
                try:
                    self.load_model(model_path)
                except Exception as e:
                    print(f"❌ 모델 로드 실패 → {e}")
            else:
                print(f"⚙️ {self.agent_id} 모델 없음 → pretrain 수행...")
                self.pretrain()  # 사전학습 자동 수행

            # 재확인
            if self.model is None or not hasattr(self.model, "parameters"):
                raise RuntimeError(f"{self.agent_id} 모델이 초기화되지 않았습니다.")

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

        self.targets.append(Target(
            next_close=float(predicted_price),
            uncertainty=float(std_pred[-1]),
            confidence=float(confidence[-1])
        ))
        
        return self.targets[-1]


    # -----------------------------
    # 메인 워크플로
    # -----------------------------
    def reviewer_draft(self, stock_data: StockData = None, target: Target = None) -> Opinion:
        """(1) searcher → (2) predicter → (3) LLM(JSON Schema)로 reason 생성 → Opinion 반환"""

        # 1) 데이터 수집
        if stock_data is None:
            stock_data = getattr(self.stockdata, self.agent_id)

        # 2) 예측값 생성
        if target is None:
            target = self.predict(stock_data)

        # 3) LLM 호출(reason 생성) - 전달받은 stock_data 사용
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")

        # 4) Opinion 기록/반환 (항상 최신 값 append)
        self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason)) 

        # 최신 오피니언 반환
        return self.opinions[-1]

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""
        
        # 1️⃣ 메시지 생성 (context 구성은 별도 헬퍼에서)
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )

        # 2️⃣ LLM 호출
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"}
                },
                "required": ["stance", "message"],
                "additionalProperties": False
            }
        )

        # 3️⃣ 결과 정리 및 기록
        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )

        # 4️⃣ 저장
        self.rebuttals[round].append(result)
        
        # 5️⃣ 디버깅 로그
        if self.verbose:
            print(
                f"[{self.agent_id}] rebuttal 생성 → {result.stance} "
                f"({my_opinion.agent_id} → {other_opinion.agent_id})"
            )

        return result

    def reviewer_revise(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: List[Rebuttal],
        stock_data: StockData,
        fine_tune: bool = True,
        lr: float = 1e-4,
        epochs: int = 20,
    ):
        """
        Revision 단계:
        - 내 의견, 다른 의견, rebuttal들을 바탕으로 변경된 가격(target) 계산
        - fine-tuning 수행
        - 수정된 reasoning을 LLM으로 생성
        - 새로운 Opinion 반환
        """

        # ======================================================
        # 1️⃣ 신뢰도 기반 β-weighted 평균으로 target 보정
        # ======================================================
        try:
            my_price = my_opinion.target.next_close
            my_conf = my_opinion.target.confidence or 1e-6

            other_prices = [o.target.next_close for o in others]
            other_confs = [o.target.confidence or 1e-6 for o in others]

            all_prices = [my_price] + other_prices
            all_confs = [my_conf] + other_confs

            betas = np.array(all_confs) / np.sum(all_confs)
            weighted_avg = float(np.sum(np.array(all_prices) * betas))

            revised_price = 0.8 * weighted_avg + 0.2 * my_price
        except Exception as e:
            print(f"[{self.agent_id}] revised_target 계산 실패: {e}")
            revised_price = my_opinion.target.next_close

        # ======================================================
        # 2️⃣ Fine-tuning 단계 (선택)
        # ======================================================
        loss_value = None
        if fine_tune and hasattr(self, "model"):
            try:
                X_input = self.searcher(self.ticker)
                device = next(self.model.parameters()).device
                X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
                y_tensor = torch.tensor([[revised_price]], dtype=torch.float32).to(device)

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
                print(f"[{self.agent_id}] fine-tuning 완료: loss={loss_value}")
            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")

        # ======================================================
        # 3️⃣ Fine-tuning 이후 새 예측값 생성
        # ======================================================
        try:
            X_latest = self.searcher(self.ticker)
            new_target = self.predict(X_latest)
        except Exception as e:
            print(f"[{self.agent_id}] predict 실패: {e}")
            new_target = my_opinion.target

        # ======================================================
        # 4️⃣ LLM Reasoning 생성
        # ======================================================
        try:
            sys_text, user_text = self._build_messages_revision(
                my_opinion=my_opinion,
                others=others,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )
        except Exception as e:
            print(f"[{self.agent_id}] _build_messages_revision 실패: {e}")
            sys_text, user_text = (
                "너는 금융 분석가다. 간단히 reason만 생성하라.",
                json.dumps({"reason": "기본 메시지 생성 실패"}),
            )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        )

        

        # ======================================================
        # 5️⃣ Opinion 업데이트 및 반환
        # ======================================================
        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=new_target,
            reason=revised_reason,
        )

        self.opinions.append(revised_opinion)
        print(f"[{self.agent_id}] revise 완료 → new_close={new_target.next_close:.2f}, loss={loss_value}")
        return self.opinions[-1]
        
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
        """저장된 모델 가중치 로드 (객체/딕셔너리/state_dict 자동 인식 + model 자동 생성)"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            print(f"⚠️ 모델 파일 없음: {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

            # 1️⃣ 모델 인스턴스가 없으면 새로 생성
            if getattr(self, "model", None) is None:
                if hasattr(self, "_build_model"):
                    self.model = self._build_model()
                    print(f"🧠 {self.agent_id} 모델 새로 생성됨 (로드 전 초기화).")
                elif hasattr(self, "forward"):
                    # Agent 자체가 nn.Module인 경우
                    self.model = self
                    print(f"🧠 {self.agent_id} 모델 직접 self로 설정됨.")
                else:
                    raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되어 있지 않음.")

            model = self.model

            # 2️⃣ 다양한 저장 포맷 처리
            if isinstance(checkpoint, torch.nn.Module):
                model.load_state_dict(checkpoint.state_dict())
                print(f"✅ {self.agent_id} 모델(객체) 로드 완료 ({model_path})")

            elif isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
                model.load_state_dict(state_dict)
                print(f"✅ {self.agent_id} 모델(state_dict) 로드 완료 ({model_path})")

            else:
                print(f"⚠️ 알 수 없는 체크포인트 포맷: {type(checkpoint)}")
                return False

            # ✅ 항상 메모리 연결 보장
            self.model = model
            model.eval()

            # 🔧 모델이 여전히 None이라면 self 자체를 모델로 설정
            if self.model is None and hasattr(self, "forward"):
                self.model = self
                print(f"🔄 {self.agent_id} 모델 self로 대체됨.")

            return True

        except Exception as e:
            print(f"❌ 모델 로드 실패: {model_path}")
            print(f"오류 내용: {e}")
            return False

    def pretrain(self):
        """Agent별 사전학습 루틴 (모델 생성, 학습, 저장, self.model 연결까지 포함)"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]

        # --------------------------
        # 1️⃣ 데이터 로드
        # --------------------------
        X, y, cols = load_dataset(self.ticker, self.agent_id, save_dir=self.data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.scaler.fit_scalers(X_train, y_train)
        self.scaler.save(self.ticker)

        X_train, y_train = map(torch.tensor, self.scaler.transform(X_train, y_train))
        X_train, y_train = X_train.float(), y_train.float()

        # --------------------------
        # 2️⃣ 모델 생성 및 초기화
        # --------------------------
        if getattr(self, "model", None) is None:
            # BaseAgent에 _build_model()이 있다면 호출
            if hasattr(self, "_build_model"):
                self.model = self._build_model()
                print(f"🧠 {self.agent_id} 모델 새로 생성됨.")
            else:
                raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되지 않음")

        model = self.model
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        # --------------------------
        # 3️⃣ 학습 루프
        # --------------------------
        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")

        # --------------------------
        # 4️⃣ 모델 저장 및 연결
        # --------------------------
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        # ✅ 메모리 연결 유지
        self.model = model

        print(f"✅ {self.agent_id} 모델 학습 및 저장 완료: {model_path}")


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

    def _msg(self, role: str, content: str) -> dict:
        """OpenAI ChatCompletion용 메시지 구조 생성"""
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}


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
