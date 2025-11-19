# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple, Any
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


# ===============================================================
# 데이터 구조 정의
# ===============================================================

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
    - SentimentalAgent: 심리/커뮤니티/뉴스 스냅샷
    - MacroSentiAgent : 거시 + 심리 스냅샷
    - TechnicalAgent  : 가격/지표 스냅샷
    - last_price      : 최신 종가
    - currency        : 통화코드
    """
    SentimentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    MacroSentiAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    TechnicalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: Optional[str] = None
    ticker: Optional[str] = None
    snapshot: Optional[Dict[str, Any]] = field(default_factory=dict)
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)
    raw_df: Optional[Any] = None 


# ===============================================================
# DataScaler: 학습/추론용 스케일러 유틸리티
# ===============================================================
class DataScaler:
    """학습/추론용 정규화 유틸리티 (BaseAgent / TechnicalAgent 등에서 공통 사용)"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.save_dir = dir_info["scaler_dir"]

        info = agents_info.get(self.agent_id, {})
        # config에 없으면 "None" 기본값
        self.x_scaler = info.get("x_scaler", "None")
        self.y_scaler = info.get("y_scaler", "None")

    # --------- 학습용 ---------
    def fit_scalers(self, X_train, y_train):
        ScalerMap = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "None": None,
        }

        Sx = ScalerMap[self.x_scaler] if isinstance(self.x_scaler, str) else self.x_scaler
        Sy = ScalerMap[self.y_scaler] if isinstance(self.y_scaler, str) else self.y_scaler

        # 3D 입력 (samples, seq_len, features) → 2D로 변환
        n_samples, seq_len, n_feats = X_train.shape
        X_2d = X_train.reshape(-1, n_feats)

        self.x_scaler = (Sx().fit(X_2d) if isinstance(Sx, type) else Sx.fit(X_2d)) if Sx else None
        self.y_scaler = (
            Sy().fit(y_train.reshape(-1, 1)) if isinstance(Sy, type) else Sy.fit(y_train.reshape(-1, 1))
        ) if Sy else None

    def transform(self, X, y=None):
        # 3D 입력 (samples, seq_len, features) → 2D로 변환
        if X.ndim == 3:
            n_samples, seq_len, n_feats = X.shape
            X_2d = X.reshape(-1, n_feats)
            X_t = (
                self.x_scaler.transform(X_2d).reshape(n_samples, seq_len, n_feats)
                if self.x_scaler
                else X
            )
        else:
            X_t = self.x_scaler.transform(X) if self.x_scaler else X

        y_t = (
            self.y_scaler.transform(y.reshape(-1, 1)).flatten()
            if (self.y_scaler and y is not None)
            else y
        )
        return X_t, y_t
    
    # 🔹 스케일러 저장
    def save(self, ticker: str, agent_id: str = "SentimentalAgent"):
        """
        현재 스케일러 객체(self)를 models/scalers/{ticker}_{agent_id}.pkl 로 저장
        """
        model_dir = Path("models/scalers")
        model_dir.mkdir(parents=True, exist_ok=True)

        path = model_dir / f"{ticker}_{agent_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[DataScaler.save] scaler saved to {path}")

    # 🔹 클래스 메서드 형태 로드
    @classmethod
    def load(cls, ticker: str, agent_id: str = "SentimentalAgent"):
        """
        저장된 스케일러를 로드해서 반환.
        없으면 None 반환.
        """
        model_dir = Path("models/scalers")
        path = model_dir / f"{ticker}_{agent_id}.pkl"

        if not path.exists():
            print(f"[DataScaler.load] no scaler file found: {path}")
            return None

        with open(path, "rb") as f:
            scaler = pickle.load(f)

        # 타입 체크 (선택)
        if not isinstance(scaler, cls):
            print(f"[DataScaler.load] warning: loaded object is {type(scaler)}, expected {cls}")
        else:
            print(f"[DataScaler.load] scaler loaded from {path}")

        return scaler

    # 🔹 인스턴스 메서드 형태의 load (지금 코드와 호환용)
    def load_for_agent(self, ticker: str, agent_id: str = "SentimentalAgent"):
        """
        self.load(...) 대신 type(self).load(...) 를 부르는 helper.
        """
        return type(self).load(ticker, agent_id)
        
    # --------- 역변환/저장 ---------
    def inverse_y(self, y_pred):
        if self.y_scaler and self.y_scaler != "None" and hasattr(self.y_scaler, "inverse_transform"):
            if isinstance(y_pred, (list, tuple)):
                y_pred = np.array(y_pred)
            return self.y_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).flatten()
        return y_pred

    def _convert_uncertainty_to_confidence(self, sigma: float) -> float:
        """
        std(σ)를 0~1 사이 confidence로 바꿔주는 헬퍼.
        σ가 작을수록 confidence ↑
        """
        import numpy as np

        sigma = float(abs(sigma) or 1e-6)
        return float(1.0 / (1.0 + np.log1p(sigma)))

    def save(self, ticker: str):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.x_scaler and self.x_scaler != "None":
            joblib.dump(
                self.x_scaler,
                os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_xscaler.pkl"),
            )
        if self.y_scaler and self.y_scaler != "None":
            joblib.dump(
                self.y_scaler,
                os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_yscaler.pkl"),
            )

    def load(self, ticker: str):
        x_path = os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_xscaler.pkl")
        y_path = os.path.join(self.save_dir, f"{ticker}_{self.agent_id}_yscaler.pkl")
        if os.path.exists(x_path):
            self.x_scaler = joblib.load(x_path)
        if os.path.exists(y_path):
            self.y_scaler = joblib.load(y_path)


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
        ticker: str | None = None,
        gamma: float = 0.3,
        delta_limit: float = 0.05,
    ):
        load_dotenv()

        self.agent_id = agent_id
        self.model = None  # torch.nn.Module 또는 에이전트별 모델
        self.temperature = temperature
        self.verbose = verbose
        self.need_training = need_training
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.ticker = ticker

        # 스케일러 (agent별로 config에서 x_scaler / y_scaler 지정)
        self.scaler: DataScaler | None = DataScaler(agent_id)

        # 윈도우/수렴율/이동한계는 config 우선
        info = agents_info.get(agent_id, {})
        self.window_size = info.get("window_size", 40)
        self.gamma = info.get("gamma", gamma)
        self.delta_limit = info.get("delta_limit", delta_limit)

        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [m for m in self.preferred_models if m != model]

        # API 키
        self.api_key = os.getenv("CAPSTONE_OPENAI_API") or ""
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 상태값
        self.stockdata: Optional[StockData] = None
        self.targets: List[Target] = []
        self.opinions: List[Opinion] = []
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)

        # JSON Schema (Opinion/Rebuttal)
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

    # -----------------------------------------------------------
    # 데이터 검색 (데이터셋 로드 + StockData 스냅샷 구성)
    # -----------------------------------------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        import pandas as pd

        agent_id = self.agent_id
        if ticker is None:
            ticker = self.ticker
        self.ticker = ticker

        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")

        # 데이터셋이 없으면 자동 생성
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {agent_id} dataset not found. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir)

        # CSV 로드
        X, y, feature_cols = load_dataset(ticker, agent_id=agent_id, save_dir=self.data_dir)

        # StockData 초기화
        if self.stockdata is None:
            self.stockdata = StockData()
        sd = self.stockdata

        # 최근 window
        X_latest = X[-1:]
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # DataFrame 변환
        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(sd, agent_id, feature_dict)

        # 종가 및 통화
        sd.ticker = ticker
        try:
            data = yf.download(ticker, period="1d", interval="1d", auto_adjust=False, progress=False)
            val = data["Close"].iloc[-1]
            sd.last_price = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            print("yfinance 오류 발생 (last_price)")

        try:
            sd.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception as e:
            print(f"yfinance 오류 발생, 통화 기본값 사용: {e}")
            sd.currency = "USD"

        print(f"■ {agent_id} StockData 생성 완료 ({ticker}, {sd.currency})")
        return X_tensor

    # -----------------------------------------------------------
    # current_price 추론 유틸
    # -----------------------------------------------------------
    def _infer_current_price(self, X, X_arr, explicit_current_price=None) -> float:
        """
        current_price가 None일 때, StockData / snapshot / 배열에서 최대한 추론.
        실패하면 RuntimeError 던짐.
        """
        if explicit_current_price is not None:
            return float(explicit_current_price)

        sd = None
        try:
            from agents.base_agent import StockData as _SD
        except Exception:
            _SD = object

        if isinstance(X, _SD):
            sd = X
        elif hasattr(self, "stockdata"):
            sd = getattr(self, "stockdata", None)

        # snapshot/meta 사용
        if sd is not None:
            snap = getattr(sd, "snapshot", None) or getattr(sd, "meta", None) or {}
            if isinstance(snap, dict):
                for key in ("last_price", "current_price", "close", "adj_close"):
                    v = snap.get(key)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass

        # 배열에서 추론
        import numpy as _np

        if X_arr is not None and _np.ndim(X_arr) >= 2:
            last_step = X_arr[-1]
            if _np.ndim(last_step) == 2:
                last_step = last_step[-1]
            try:
                return float(last_step[-1])
            except Exception:
                pass

        raise RuntimeError(
            "[BaseAgent.predict] current_price를 자동으로 찾지 못했습니다.\n"
            "- StockData.snapshot 또는 StockData.meta에 'last_price'/'current_price'를 넣거나,\n"
            "- predict(X, current_price=...) 로 직접 전달해 주세요."
        )

    # -----------------------------------------------------------
    # Monte Carlo Dropout 기반 예측 (공통)
    # -----------------------------------------------------------
    def predict(self, X, n_samples: int = 30, current_price: float | None = None) -> Target:
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)

        기본값:
        - 모델이 "다음날 수익률(return)"을 예측한다고 가정하고
          current_price * (1 + return) 으로 종가를 복원.
        - 에이전트가 decode_prediction(y_pred_raw, stock_data, current_price)
          를 구현했다면 그 로직을 우선 사용.
        """
        import numpy as _np
        import torch as _torch

        X_original = X

        # StockData → 내부 배열로 변환
        try:
            from agents.base_agent import StockData as _SD
        except Exception:
            _SD = None

        if _SD is not None and isinstance(X, _SD):
            X_arr = None
            for name in ["X", "x", "X_seq", "data", "inputs"]:
                if hasattr(X, name):
                    X_arr = getattr(X, name)
                    break
            if X_arr is None and hasattr(X, "__dict__"):
                for name, val in X.__dict__.items():
                    if isinstance(val, (np.ndarray, torch.Tensor)):
                        X_arr = val
                        break
            if X_arr is None:
                raise AttributeError(
                    "StockData 안에서 입력 배열(np.ndarray/torch.Tensor) 필드를 찾지 못했습니다."
                )
            X = X_arr

        # numpy / tensor 정규화
        if isinstance(X, _torch.Tensor):
            X_tensor = X.float()
        else:
            X_np = _np.asarray(X, dtype=_np.float32)
            X_tensor = _torch.from_numpy(X_np)

        # [T, F] → [1, T, F]
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(0)

        # 모델 준비
        if not hasattr(self, "model") or self.model is None:
            # 필요한 경우 여기서 self.pretrain() 또는 load_model() 호출해도 됨
            raise RuntimeError(f"{self.agent_id} 모델이 초기화되지 않았습니다.")

        device = next(self.model.parameters()).device
        X_tensor = X_tensor.to(device)

        # Monte Carlo Dropout
        self.model.train()
        preds = []
        with _torch.no_grad():
            for _ in range(n_samples):
                out = self.model(X_tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                preds.append(out.detach().cpu().numpy())

        preds_arr = _np.stack(preds, axis=0)  # [S, B, D]
        mean_pred = preds_arr.mean(axis=0).squeeze()
        std_pred = preds_arr.std(axis=0).squeeze()

        # σ, confidence
        if _np.ndim(std_pred) > 0:
            sigma = float(std_pred[-1])
        else:
            sigma = float(std_pred)
        confidence = float(1.0 / (1.0 + sigma))

        # current_price 추론
        X_arr_for_price = X_tensor.detach().cpu().numpy()
        current_price_val = self._infer_current_price(
            X_original,
            X_arr_for_price,
            explicit_current_price=current_price,
        )

        mean_pred = _np.asarray(mean_pred)

        # decode_prediction이 있으면 사용
        if hasattr(self, "decode_prediction"):
            next_close = float(
                self.decode_prediction(
                    mean_pred,
                    stock_data=getattr(self, "stockdata", None),
                    current_price=current_price_val,
                )
            )
        else:
            # 기본: return → price
            if mean_pred.ndim == 0:
                predicted_return = float(mean_pred)
            else:
                predicted_return = float(mean_pred[-1])
            next_close = float(current_price_val * (1.0 + predicted_return))

        # uncertainty 정리
        if std_pred is not None:
            std_pred = _np.asarray(std_pred)
            if std_pred.ndim == 0:
                uncertainty = float(std_pred)
            else:
                uncertainty = float(std_pred[-1])
        else:
            uncertainty = None

        target = Target(
            next_close=next_close,
            uncertainty=uncertainty,
            confidence=confidence,
        )
        self.targets.append(target)
        return target

    # -----------------------------------------------------------
    # 메인 워크플로 (Opinion / Rebuttal / Revision)
    # -----------------------------------------------------------
    def reviewer_draft(self, stock_data=None, target: Target | None = None) -> Opinion:
        # 1) StockData 확보
        if stock_data is None:
            sd = getattr(self, "stockdata", None)
            if sd is None:
                raise RuntimeError(
                    f"[{self.agent_id}] stockdata가 None 입니다. "
                    "먼저 run_dataset()/searcher() 등을 호출하세요."
                )
            if isinstance(sd, dict):
                stock_data = sd.get(self.agent_id, None)
            else:
                stock_data = sd

            if stock_data is None:
                raise RuntimeError(
                    f"[{self.agent_id}] stockdata에서 유효한 StockData를 찾지 못했습니다."
                )

        # 2) 예측값 생성
        if target is None:
            target = self.predict(stock_data)

        # 3) LLM 호출(reason 생성)
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)
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
        reason = parsed.get("reason", "(사유 생성 실패)")

        op = Opinion(agent_id=self.agent_id, target=target, reason=reason)
        self.opinions.append(op)
        return op

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata,
        )
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            self.schema_obj_rebuttal,
        )

        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)"),
        )
        self.rebuttals[round].append(result)

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
    ) -> Opinion:
        """
        Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - fine-tuning (return 단위, 선택)
        - reasoning 생성
        """
        gamma = getattr(self, "gamma", 0.3)
        delta_limit = getattr(self, "delta_limit", 0.05)

        # ① 수렴 업데이트
        try:
            my_price = my_opinion.target.next_close
            my_sigma = abs(my_opinion.target.uncertainty or 1e-6)

            other_prices = np.array([o.target.next_close for o in others])
            other_sigmas = np.array([abs(o.target.uncertainty or 1e-6) for o in others])

            all_sigmas = np.concatenate([[my_sigma], other_sigmas])
            all_prices = np.concatenate([[my_price], other_prices])

            inv_sigmas = 1 / (all_sigmas + 1e-6)
            betas = inv_sigmas / inv_sigmas.sum()

            delta = np.sum(betas[1:] * (other_prices - my_price))
            revised_price = my_price + gamma * delta

            current_price = getattr(self.stockdata, "last_price", 100.0)
            up = current_price * (1 + delta_limit)
            down = current_price * (1 - delta_limit)
            revised_price = float(np.clip(revised_price, down, up))
        except Exception as e:
            print(f"[{self.agent_id}] revised_target 계산 실패: {e}")
            revised_price = my_opinion.target.next_close

        # ② Fine-tuning (선택)
        loss_value = None
        if fine_tune and hasattr(self, "model") and self.model is not None:
            try:
                current_price = getattr(self.stockdata, "last_price", 100.0)
                revised_return = (revised_price / current_price) - 1

                X_input = self.searcher(self.ticker)
                device = next(self.model.parameters()).device
                X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
                y_tensor = torch.tensor([[revised_return]], dtype=torch.float32).to(device)

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
                print(f"[{self.agent_id}] fine-tuning 완료: loss={loss_value:.6f}")
            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")

        # ③ fine-tuning 이후 새 예측
        try:
            X_latest = self.searcher(self.ticker)
            new_target = self.predict(X_latest)
        except Exception as e:
            print(f"[{self.agent_id}] predict 실패: {e}")
            new_target = my_opinion.target

        # ④ reasoning 생성
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
        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")

        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=new_target,
            reason=revised_reason,
        )
        self.opinions.append(revised_opinion)
        print(
            f"[{self.agent_id}] revise 완료 → new_close={new_target.next_close:.2f}, "
            f"loss={loss_value}"
        )
        return revised_opinion

    # -----------------------------------------------------------
    # 에이전트별 구현이 필요한 메서드 (프롬프트 빌더)
    # -----------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_messages_opinion method"
        )

    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_messages_rebuttal method"
        )

    def _build_messages_revision(self, *args, **kwargs) -> Tuple[str, str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_messages_revision method"
        )

    # -----------------------------------------------------------
    # 모델 로드 / pretrain
    # -----------------------------------------------------------
    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치 로드 (객체/딕셔너리/state_dict 자동 인식 + model 자동 생성)"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            print(f"■ 모델 파일 없음: {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

            if getattr(self, "model", None) is None:
                if hasattr(self, "_build_model"):
                    self.model = self._build_model()
                    print(f"■ {self.agent_id} 모델 새로 생성됨 (로드 전 초기화).")
                elif hasattr(self, "forward"):
                    self.model = self
                    print(f"■ {self.agent_id} 모델 직접 self로 설정됨.")
                else:
                    raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되어 있지 않음.")

            model = self.model

            if isinstance(checkpoint, torch.nn.Module):
                model.load_state_dict(checkpoint.state_dict())
                print(f" {self.agent_id} 모델(객체) 로드 완료 ({model_path})")
            elif isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
                model.load_state_dict(state_dict)
                print(f" {self.agent_id} 모델(state_dict) 로드 완료 ({model_path})")
            else:
                print(f" 알 수 없는 체크포인트 포맷: {type(checkpoint)}")
                return False

            self.model = model
            model.eval()
            return True

        except Exception as e:
            print(f"■ 모델 로드 실패: {model_path}")
            print(f"오류 내용: {e}")
            return False

    def pretrain(self):
        """Agent별 사전학습 루틴 (모델 생성, 학습, 저장, self.model 연결까지 포함)"""
        info = agents_info[self.agent_id]
        epochs = info["epochs"]
        lr = info["learning_rate"]
        batch_size = info["batch_size"]

        # --------------------------
        # 데이터 로드
        # --------------------------
        X, y, cols = load_dataset(self.ticker, self.agent_id, save_dir=self.data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 타깃 스케일 조정 (수익률 × 100)
        y_train *= 100.0
        y_val *= 100.0

        # --------------------------
        # 스케일링 (있으면 사용)
        # --------------------------
        use_scaler = (
            getattr(self, "scaler", None) is not None
            and hasattr(self.scaler, "fit_scalers")
            and hasattr(self.scaler, "transform")
        )

        if use_scaler:
            self.scaler.fit_scalers(X_train, y_train)
            if hasattr(self.scaler, "save"):
                self.scaler.save(self.ticker)
            X_train, y_train = self.scaler.transform(X_train, y_train)
        else:
            print(f"[WARN] {self.agent_id}: scaler 없음 → 비스케일링 데이터로 pretrain 진행")

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # --------------------------
        # 모델 생성
        # --------------------------
        if getattr(self, "model", None) is None:
            if hasattr(self, "_build_model"):
                self.model = self._build_model()
                print(f"■ {self.agent_id} 모델 새로 생성됨.")
            else:
                raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되지 않음")

        model = self.model
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.HuberLoss(delta=1.0)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        # --------------------------
        # 학습 루프
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
        # 모델 저장 및 연결
        # --------------------------
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        self.model = model

        print(f" {self.agent_id} 모델 학습 및 저장 완료: {model_path}")

    # -----------------------------------------------------------
    # OpenAI API 호출
    # -----------------------------------------------------------
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
        if not msg_sys or not msg_user:
            raise ValueError("Invalid messages: system or user message is None.")

        if schema_obj and isinstance(schema_obj, dict):
            schema_obj.setdefault("additionalProperties", False)
            if "type" not in schema_obj:
                schema_obj["type"] = "object"

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
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}

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
                    return {}

                if r.status_code in (400, 404):
                    last_err = (r.status_code, r.text)
                    continue
                r.raise_for_status()
            except Exception as e:
                print(f"■ {self.agent_id} - 모델 {model} 실패: {e}")
                last_err = str(e)
                continue

        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    # -----------------------------------------------------------
    # 평가 유틸
    # -----------------------------------------------------------
    def evaluate(self, ticker: str | None = None):
        """검증 데이터로 성능 평가"""
        if ticker is None:
            ticker = self.ticker

        X, y, feature_cols = load_dataset(ticker, agent_id=self.agent_id, save_dir=self.data_dir)

        split_idx = int(len(X) * 0.8)
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        if self.scaler:
            self.scaler.load(ticker)

        predictions = []
        actual_returns = []

        for i in range(len(X_val)):
            X_input = X_val[i:i+1]
            X_tensor = torch.tensor(X_input, dtype=torch.float32)

            with torch.no_grad():
                pred_return = self.model(X_tensor).item()
                predictions.append(pred_return)
                actual_returns.append(y_val[i, 0])

        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)

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

    # -----------------------------------------------------------
    # 기타
    # -----------------------------------------------------------
    def _msg(self, role: str, content: str) -> dict:
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}
