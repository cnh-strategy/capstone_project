# debate_ver3/agents/base_agent.py
# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
from collections import defaultdict
import os, json, requests, yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import pandas as pd  # ✅ (2) SentimentalAgent 스냅샷 주입에 필요

from debate_ver3.config.agents import agents_info, dir_info
from debate_ver3.core.data_set import build_dataset, load_dataset


# -----------------------------
# 데이터 구조 정의
# -----------------------------
@dataclass
class Target:
    """예측 목표값 + 불확실성 정보 포함
    - next_close: 다음 거래일 종가 예측치 (또는 수익률 기반 변환값)
    - uncertainty: Monte Carlo Dropout 기반 예측 표준편차(σ)
    - confidence: 모델 신뢰도 (정규화된 신뢰도; 선택적)
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
    # 최소 공통 필드 (SentimentalAgent 등은 searcher에서 동적 필드 추가)
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
        self.agent_id = agent_id        # 에이전트 식별자
        self.model = model              # (LLM) 모델 이름
        self.temperature = temperature  # Temperature 설정
        self.verbose = verbose          # 디버깅 모드
        self.need_training = need_training  # 모델 학습 필요 여부
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.ticker = ticker

        # 🔹 스케일러 유틸
        self.scaler = DataScaler(agent_id)

        # 모델 폴백 우선순위 (LLM)
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [m for m in self.preferred_models if m != model]

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

        # JSON Schema (참고용)
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

    # -----------------------------
    # 데이터 수집/스냅샷
    # -----------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """
        데이터 검색기
        - CSV가 없을 경우 build_dataset()으로 자동 생성
        - 마지막 window 시퀀스를 torch.tensor로 반환
        - (2) SentimentalAgent 스냅샷 주입
        - feature_cols / asof_date 메타 저장
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

        # StockData 인스턴스 생성/기록
        self.stockdata = StockData()
        self.stockdata.agent_id = self.agent_id
        self.stockdata.ticker = ticker
        self.stockdata.X = X
        self.stockdata.y = y
        self.stockdata.feature_cols = feature_cols

        # 가장 최근 window 데이터만 사용
        X_latest = X[-1:]  # shape: (1, window_size, n_features)
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # 🔹 SentimentalAgent 스냅샷(마지막 시점 값으로 키:값 매핑)
        #    ctx에서 stockdata.SentimentalAgent['news_sentiment'] 형태로 접근 가능하게 한다.
        try:
            df_last = pd.DataFrame(X_latest[0], columns=feature_cols)  # (T, F)
            last_row_dict = df_last.iloc[-1].to_dict()                 # {feature: value}
        except Exception:
            last_row_dict = {}
        setattr(self.stockdata, "SentimentalAgent", last_row_dict)

        # 🔹 메타 정보 저장
        setattr(self.stockdata, "feature_cols", feature_cols)
        setattr(self.stockdata, "asof_date", str(pd.Timestamp.today().date()))

        # 🔹 (3) yfinance 가드 포함: 최신 종가
        try:
            data = yf.download(ticker, period="1d", interval="1d", progress=False)
            if data is not None and not data.empty:
                close_last = data["Close"].iloc[-1]
                # numpy 타입이면 .item()으로 파이썬 float로
                if hasattr(close_last, "item"):
                    close_last = close_last.item()
                self.stockdata.last_price = float(close_last)
            else:
                # 빈 데이터면 기존 값 유지 또는 기본값
                self.stockdata.last_price = self.stockdata.last_price or 100.0
        except Exception:
            self.stockdata.last_price = self.stockdata.last_price or 100.0

        # 🔹 (4) 최신 입력 캐시
        self._last_X = X_tensor

        return X_tensor

    # -----------------------------
    # 예측 (MC Dropout)
    # -----------------------------
    def predict(self, X, n_samples: int = 30, current_price: float = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성 계산 (공통)
        모든 Agent에서 사용 가능
        """
        # (옵션) 저장된 가중치가 있으면 한번 로드 시도 (없으면 그냥 진행)
        try:
            self.load_model()
        except Exception:
            pass

        # 1. 스케일러 로드
        self.scaler.load(self.ticker)

        # 2. 입력 텐서 변환
        if isinstance(X, np.ndarray):
            X_scaled, _ = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        # 3. 모델 선택 (self.model 또는 self)
        model = getattr(self, "model", None)
        if model is None:
            model = self  # 자식이 nn.Module 상속 시

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
        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        # 7. 상승/하락율을 실제 가격으로 변환
        if current_price is None:
            current_price = getattr(self.stockdata, "last_price", 100.0)

        return_rate = float(mean_pred[-1])
        predicted_price = (
            self.scaler.convert_return_to_price(return_rate, current_price)
            if hasattr(self.scaler, "convert_return_to_price")
            else current_price * (1 + return_rate)
        )

        # 간단 confidence (σ의 역수)
        confidence = 1.0 / (std_pred + 1e-8)

        return Target(
            next_close=float(predicted_price),
            uncertainty=float(std_pred[-1]),
            confidence=float(confidence[-1]),
        )

    # -----------------------------
    # 메인 워크플로 (Debate 호환 시그니처)
    # -----------------------------
    def reviewer_draft(self, ticker: str) -> Opinion:
        """
        (4) 캐시 재사용: _last_X와 stockdata가 있으면 재다운로드 없이 사용
        """
        # 1) 데이터 확보
        if getattr(self, "_last_X", None) is None or self.stockdata is None or self.stockdata.ticker != ticker:
            X = self.searcher(ticker)
        else:
            X = self._last_X

        # 2) 예측값 생성
        target = self.predict(X)

        # 3) LLM 호출(reason 생성)
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)

        schema_reason_only = {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,  # ✅ 필수
        }

        try:
            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                schema_reason_only,
            )
        except Exception:
            # API 실패 시에도 토론이 계속 돌도록 안전 폴백
            parsed = {
                "reason": f"(LLM 실패로 기본 사유 사용) 예측가={target.next_close:.2f}, σ={target.uncertainty:.4f}"
            }

        reason = parsed.get("reason", "(사유 생성 실패)")
        op = Opinion(agent_id=self.agent_id, target=target, reason=reason)
        self.opinions.append(op)
        return op

    def reviewer_rebut(
        self,
        round_num: int,
        my_lastest: Opinion,
        others_latest: Dict[str, Opinion],
        stock_data: Optional[StockData],
    ) -> List[Rebuttal]:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성 (기본: 침묵)"""
        # 필요 시 자식에서 _build_messages_rebuttal 구현 후 여기서 호출
        # 기본 구현은 토론을 방해하지 않도록 '반환 없음'
        return []

    def reviewer_revise(
        self,
        my_lastest: Opinion,
        others_latest: Dict[str, Opinion],
        received_rebuttals: List[Rebuttal],
        stock_data: Optional[StockData],
    ) -> Opinion:
        """
        기본 수정 로직:
        - 반박 수가 많으면 불확실성 소폭 증가, 지지가 많으면 감소
        - 수치를 크게 바꾸지 않고 형식만 맞춰서 최신 Opinion으로 append
        """
        t = my_lastest.target
        delta = 1.0
        for r in received_rebuttals:
            if r.to_agent_id != self.agent_id:
                continue
            if r.stance == "REBUT":
                delta *= 1.02
            elif r.stance == "SUPPORT":
                delta *= 0.98

        revised = Target(
            next_close=t.next_close,
            uncertainty=min(5.0, max(0.0, (t.uncertainty or 0.0) * delta)),
            confidence=min(1.0, (t.confidence or 0.0) / delta if delta > 0 else (t.confidence or 0.0)),
            feature_cols=t.feature_cols,
            importances=t.importances,
        )

        # 간단한 수정 사유 (LLM 없이 기본값)
        revised_reason = my_lastest.reason or "(이전 의견 없음)"
        op = Opinion(agent_id=self.agent_id, target=revised, reason=revised_reason)
        self.opinions.append(op)
        return op

    # -----------------------------
    # 공통 유틸
    # -----------------------------
    def _p(self, msg: str):
        if self.verbose:
            print(f"[{self.agent_id}] {msg}")

    @staticmethod
    def _msg(role: str, text: str) -> dict:
        # OpenAI Responses API용 포맷
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

    # -----------------------------
    # 모델 저장/로딩
    # -----------------------------
    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치 로드"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            # 모델이 없으면 조용히 패스
            return False

        # self는 nn.Module이 아닐 수 있으므로 self.model 우선
        try:
            if hasattr(self, "model") and hasattr(self.model, "load_state_dict"):
                state = torch.load(model_path, map_location=torch.device("cpu"))
                if isinstance(state, dict) and "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                else:
                    self.model.load_state_dict(state)
            elif hasattr(self, "load_state_dict"):
                state = torch.load(model_path, map_location=torch.device("cpu"))
                self.load_state_dict(state if not isinstance(state, dict) else state.get("model_state_dict", state))
            else:
                return False
            if self.verbose:
                print(f"✅ {self.agent_id} 모델 로드 완료 ({model_path})")
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ {self.agent_id} 모델 로드 실패: {e}")
            return False

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

        model = self if hasattr(self, "forward") else self.model
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
        # 통일: state_dict만 저장
        torch.save(model.state_dict(), model_path)
        print(f"✅ {self.agent_id} model saved.\n✅ pretraining finished.\n")

    # -----------------------------
    # OpenAI API 호출
    # -----------------------------
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
        # ✅ 스키마 방어: 루트에 additionalProperties: False 강제 주입
        if isinstance(schema_obj, dict) and "additionalProperties" not in schema_obj:
            schema_obj = dict(schema_obj)
            schema_obj["additionalProperties"] = False

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
    # 🔹 추가: 간단한 성능 평가 (참고용)
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
            X_input = X_val[i:i + 1]
            X_tensor = torch.tensor(X_input, dtype=torch.float32)

            with torch.no_grad():
                pred_target = self.predict(X_tensor)
                predictions.append(pred_target.next_close)
                actual_returns.append(y_val[i, 0])

        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)

        # 성능 지표 계산 (참고: price vs return 혼합이므로 방향 정확도 위주 해석 권장)
        mae = np.mean(np.abs(predictions - actual_returns))
        rmse = np.sqrt(np.mean((predictions - actual_returns) ** 2))
        correlation = np.corrcoef(predictions, actual_returns)[0, 1]

        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual_returns)
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100

        return {
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "direction_accuracy": direction_accuracy,
            "n_samples": len(predictions),
        }


# ===============================================================
# DataScaler: 학습/추론용 정규화 유틸리티 (BaseAgent 내부용)
# ===============================================================
class DataScaler:
    """학습/추론용 정규화 유틸리티 (BaseAgent 내부용)"""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.save_dir = dir_info["scaler_dir"]
        # 이름 보존
        self.x_scaler_name = agents_info[self.agent_id]["x_scaler"]
        self.y_scaler_name = agents_info[self.agent_id]["y_scaler"]
        # 실제 인스턴스
        self.x_scaler = None
        self.y_scaler = None

    def fit_scalers(self, X_train, y_train):
        ScalerMap = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "None": None,
        }
        Sx = ScalerMap[self.x_scaler_name]
        Sy = ScalerMap[self.y_scaler_name]

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
            X_t = self.x_scaler.transform(X_2d).reshape(n_samples, seq_len, n_feats) if self.x_scaler else X
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
            if isinstance(y_pred, (list, tuple)):
                y_pred = np.array(y_pred)
            return self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred

    def convert_return_to_price(self, return_rate, current_price):
        """상승/하락율을 실제 가격으로 변환"""
        return current_price * (1 + return_rate)

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
        else:
            self.x_scaler = None
        if os.path.exists(y_path):
            self.y_scaler = joblib.load(y_path)
        else:
            self.y_scaler = None
