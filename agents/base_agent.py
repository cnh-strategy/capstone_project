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
    - sentimental: 심리/커뮤니티/뉴스 스냅샷
    - fundamental: 재무/밸류에이션 요약
    - technical  : 가격/지표 스냅샷
    - last_price : 최신 종가
    - currency   : 통화코드
    - feature_cols: 피처 컬럼 목록
    """
    SentimentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    MacroAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    TechnicalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: Optional[str] = None
    ticker: Optional[str] = None
    feature_cols: Optional[List[str]] = field(default_factory=list)


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
            gamma: float = 0.3,
            delta_limit: float = 0.05,
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
            raise RuntimeError("환경변수 CAPSTONE_OPENAI_API가 설정되지 않았습니다.")

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

        # 수렴율 및 이동 한계
        self.gamma = agents_info[agent_id]["gamma"]
        self.delta_limit = agents_info[agent_id]["delta_limit"]

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

        self.ticker = ticker

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

        # 컬럼별 리스트 저장
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}

        setattr(self.stockdata, agent_id, feature_dict)

        # 종가 및 통화 정보
        self.stockdata.ticker = ticker

        try:
            data = yf.download(ticker, period="1d", interval="1d", auto_adjust=False, progress=False)
            val = data["Close"].iloc[-1]
            self.stockdata.last_price = float(val.item() if hasattr(val, "item") else val)
        except Exception as e:
            print(f"yfinance 오류 발생")


        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception as e:
            print(f"yfinance 오류 발생, 통화 기본값 사용: {e}")
            self.stockdata.currency = "USD"

        # StockData 생성 완료 (로그는 DebateAgent에서 처리)

        return X_tensor

    def _infer_current_price(self, X, X_arr, explicit_current_price=None) -> float:
        """
        current_price가 None일 때, StockData / snapshot / 배열에서 최대한 추론.
        실패하면 RuntimeError를 던져서 문제를 명확히 알 수 있게 한다.
        """
        # 0) 이미 함수 인자로 들어온 값이 있으면 그대로 사용
        if explicit_current_price is not None:
            return float(explicit_current_price)

        sd = None
        # 1) X가 StockData면 우선 사용
        try:
            from agents.base_agent import StockData  # 순환 import 방지용 (이미 있다면 생략 가능)
        except Exception:
            StockData = object  # 타입 체크만 우회

        if isinstance(X, StockData):
            sd = X
        # 2) 아니면 self.stockdata 사용 (run_dataset() 결과를 보통 여기 저장)
        elif hasattr(self, "stockdata"):
            sd = getattr(self, "stockdata", None)

        # 3) snapshot / meta 딕셔너리에서 찾기
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

        # 4) 배열(X_arr)의 마지막 행에서 종가 비슷한 값 추론
        import numpy as np

        if X_arr is not None and np.ndim(X_arr) >= 2:
            # 시계열이라면 보통 (T, F) or (N, T, F) 구조
            last_step = X_arr[-1]
            # (T, F) 구조일 때: 마지막 타임스텝
            if np.ndim(last_step) == 2:
                last_step = last_step[-1]
            try:
                # 마지막 컬럼을 현재가로 가정
                return float(last_step[-1])
            except Exception:
                pass

        # 5) 여기까지 왔으면 자동 추론 실패 → 명시적으로 에러
        raise RuntimeError(
            "[BaseAgent.predict] current_price를 자동으로 찾지 못했습니다.\n"
            "- StockData.snapshot 또는 StockData.meta에 'last_price' 또는 'current_price' 키를 넣어주시거나,\n"
            "- predict(X, current_price=...) 형태로 현재가를 직접 전달해 주세요."
        )


    def predict(self, X, n_samples: int = 30, current_price: float | None = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)
        """
        import numpy as np
        import torch

        # 원본 X는 나중에 current_price 추론에 사용
        X_original = X

        # 0) StockData가 들어온 경우 내부 X로 변환
        try:
            from agents.base_agent import StockData as _StockData
        except Exception:
            _StockData = None

        if _StockData is not None and isinstance(X, _StockData):
            # 우선 자주 쓰는 이름들부터 시도
            for name in ["X", "x", "X_seq", "data", "inputs"]:
                if hasattr(X, name):
                    X_arr = getattr(X, name)
                    break
            else:
                # 그래도 못 찾으면 __dict__ 안에서 np.ndarray / torch.Tensor 중 첫 번째를 사용
                X_arr = None
                if hasattr(X, "__dict__"):
                    for name, val in X.__dict__.items():
                        if isinstance(val, (np.ndarray, torch.Tensor)):
                            X_arr = val
                            # print(f"[debug] StockData.{name} 를 입력으로 사용합니다.")
                            break

                if X_arr is None:
                    raise AttributeError(
                        "StockData 안에서 입력 배열(np.ndarray 또는 torch.Tensor) 필드를 찾을 수 없습니다. "
                        "__dict__ 안에 어떤 필드들이 있는지 확인해보세요."
                    )

            # 이후 로직은 순수 배열만 다루도록 X를 바꿔줌
            X = X_arr

        # 1) numpy / tensor 정규화
        if isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            X_np = np.asarray(X, dtype=np.float32)
            X_tensor = torch.from_numpy(X_np)

        # 입력 차원: [T, F] -> [1, T, F] 로 맞춰주기
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(0)

        # device 정렬
        if hasattr(self, "device"):
            device = self.device
        else:
            device = next(self.model.parameters()).device  # type: ignore[attr-defined]

        X_tensor = X_tensor.to(device)

        # 2) Monte Carlo Dropout 실행
        self.model.train()  # dropout 활성화 (eval()이면 dropout 꺼짐)

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.model(X_tensor)
                # 모델이 (pred, hidden) 같은 튜플을 리턴하는 경우 대비
                if isinstance(out, (tuple, list)):
                    out = out[0]
                # [B, D] 가정
                preds.append(out.detach().cpu().numpy())

        preds_arr = np.stack(preds, axis=0)  # [S, B, D]
        mean_pred = preds_arr.mean(axis=0).squeeze()   # [D]
        std_pred = preds_arr.std(axis=0).squeeze()     # [D]

        # 불확실성(σ) 및 confidence 계산
        if np.ndim(std_pred) > 0:
            sigma = float(std_pred[-1])   # 마지막 출력 차원을 타겟으로 가정
        else:
            sigma = float(std_pred)

        # 예: σ가 작을수록 confidence ↑ (0~1 사이 값으로 squash)
        confidence = float(1.0 / (1.0 + sigma))

        # current_price 추론용 배열 (모델 입력과 동일 스케일)
        X_arr_for_price = X_tensor.detach().cpu().numpy()

        # 3) 현재가(current_price) 결정
        current_price_val = self._infer_current_price(
            X_original,
            X_arr_for_price,
            explicit_current_price=current_price,
        )

        # 4) 수익률 → 종가로 변환
        # 현재 모델은 "다음날 수익률(return)"을 예측한다고 가정

        # mean_pred가 스칼라인 경우와 벡터인 경우를 모두 처리
        mean_pred = np.asarray(mean_pred)

        if mean_pred.ndim == 0:
            # 모델 출력이 스칼라 1개인 경우 (바로 다음날 수익률)
            predicted_return = float(mean_pred)
        else:
            # 시계열 전체를 내보내는 모델인 경우, 마지막 시점 값 사용
            predicted_return = float(mean_pred[-1])  # 예: 0.012 = +1.2%

        predicted_price = current_price_val * (1.0 + predicted_return)

        # 5) Target 생성 및 반환
        if std_pred is not None:
            std_pred = np.asarray(std_pred)
            if std_pred.ndim == 0:
                uncertainty = float(std_pred)
            else:
                uncertainty = float(std_pred[-1])
        else:
            uncertainty = None

        target = Target(
            next_close=predicted_price,
            uncertainty=uncertainty,
            confidence=confidence,
        )
        return target



        # -----------------------------
        # 모델 준비 및 스케일러 로드
        # -----------------------------
        already_loaded = bool(getattr(self, "model_loaded", False))

        if (self.model is None or not hasattr(self.model, "parameters")) and not already_loaded:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
            if os.path.exists(model_path):
                ok = self.load_model(model_path)
                # 일부 에이전트는 model_loaded 속성이 없을 수 있어서 getattr/hasattr로 방어
                if hasattr(self, "model_loaded"):
                    self.model_loaded = bool(ok)
            else:
                if not self.ticker and getattr(self, "stockdata", None):
                    self.ticker = getattr(self.stockdata, "ticker", None)
                if not self.ticker:
                    raise ValueError("ticker가 설정되지 않았습니다. 먼저 searcher(ticker)를 호출하세요.")
                self.pretrain()

        if self.model is None:
            raise RuntimeError(f"{self.agent_id} 모델이 초기화되지 않음")

        self.scaler.load(self.ticker)

        # -----------------------------
        # 입력 변환
        # -----------------------------
        if isinstance(X, np.ndarray):
            X_scaled, _ = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        model = self.model
        device = next(model.parameters()).device
        X_tensor = X_tensor.to(device)

        # -----------------------------
        # Monte Carlo Dropout 추론
        # -----------------------------
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor).cpu().numpy().flatten()
                preds.append(y_pred)

        preds = np.stack(preds)  # (samples, seq)
        mean_pred = preds.mean(axis=0)
        std_pred = np.abs(preds.std(axis=0))  # ✅ 항상 양수

        # -----------------------------
        # σ 기반 confidence 계산
        # -----------------------------
        sigma = float(std_pred[-1])
        sigma = max(sigma, 1e-6)

        # 신뢰도: 불확실성 작을수록 1에 가까움
        confidence = 1 / (1 + np.log1p(sigma))

        # -----------------------------
        # 역변환 및 가격 계산
        # -----------------------------
        if hasattr(self.scaler, 'y_scaler') and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        if current_price is None:
            current_price = getattr(self.stockdata, 'last_price', 100.0)

        # ✅ 현재 모델은 "다음날 수익률(return)"을 예측하므로, 종가로 변환 시 (1 + return)
        predicted_return = float(mean_pred[-1])  # 예측된 상승률 (%)
        predicted_price = current_price * (1 + predicted_return)

        # -----------------------------
        # Target 생성 및 반환
        # -----------------------------
        target = Target(
            next_close=float(predicted_price),
            uncertainty=sigma,
            confidence=float(confidence),
        )
        self.targets.append(target)
        return target


    # -----------------------------
    # 메인 워크플로
    # -----------------------------
    def reviewer_draft(self, stock_data=None, target=None):
        # 1) 데이터 수집
        if stock_data is None:
            sd = getattr(self, "stockdata", None)
            if sd is None:
                raise RuntimeError(
                    f"[{self.agent_id}] stockdata가 None 입니다. 먼저 run_dataset() 또는 StockData 생성 로직을 호출하세요."
                )
            # dict / namespace / 단일 객체 모두 대응
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

        # 메시지 생성 (context 구성은 별도 헬퍼에서)
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )

        # LLM 호출
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

        # 결과 정리 및 기록
        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )

        # 저장
        self.rebuttals[round].append(result)

        # 디버깅 로그
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
        Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - fine-tuning (수익률 단위)
        - reasoning 생성
        """
        gamma = getattr(self, "gamma", 0.3)               # 수렴율 (0~1)
        delta_limit = getattr(self, "delta_limit", 0.05)  # fine-tuning 보정 한계

        try:
            # ===================================
            # ① β 계산 (불확실성 작을수록 신뢰 높음)
            # ===================================
            my_price = my_opinion.target.next_close
            my_sigma = abs(my_opinion.target.uncertainty or 1e-6)

            other_prices = np.array([o.target.next_close for o in others])
            other_sigmas = np.array([abs(o.target.uncertainty or 1e-6) for o in others])

            all_sigmas = np.concatenate([[my_sigma], other_sigmas])
            all_prices = np.concatenate([[my_price], other_prices])

            inv_sigmas = 1 / (all_sigmas + 1e-6)
            betas = inv_sigmas / inv_sigmas.sum()

            # ===================================
            # ② 논문식 수렴 업데이트
            #     y_i_rev = y_i + γ Σ β_j (y_j - y_i)
            # ===================================
            delta = np.sum(betas[1:] * (other_prices - my_price))
            revised_price = my_price + gamma * delta

        except Exception as e:
            print(f"[{self.agent_id}] revised_target 계산 실패: {e}")
            revised_price = my_opinion.target.next_close
            current_price = getattr(self.stockdata, "last_price", 100.0)
            price_uplimit = current_price * (1 + delta_limit)
            price_downlimit = current_price * (1 - delta_limit)
            revised_price = min(max(revised_price, price_downlimit), price_uplimit)

        # ===================================
        # ③ Fine-tuning (return 단위)
        # ===================================
        loss_value = None
        if fine_tune and hasattr(self, "model"):
            try:
                current_price = getattr(self.stockdata, "last_price", 100.0)
                revised_return = (revised_price / current_price) - 1  # 🔹수익률 변환

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
                    delta_loss = pred - y_tensor
                    loss = criterion(pred - delta_loss, y_tensor)
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
                print(f"[{self.agent_id}] fine-tuning 완료: loss={loss_value:.6f}")

            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")

        # ===================================
        # ④ fine-tuning 이후 새 예측 생성
        # ===================================
        try:
            X_latest = self.searcher(self.ticker)
            new_target = self.predict(X_latest)
        except Exception as e:
            print(f"[{self.agent_id}] predict 실패: {e}")
            new_target = my_opinion.target

        # ===================================
        # ⑤ reasoning 생성
        # ===================================
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
        print(f"[{self.agent_id}] revise 완료 → new_close={new_target.next_close:.2f}, loss={loss_value}")
        return self.opinions[-1]



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
            return False

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

            # 모델 인스턴스가 없으면 새로 생성
            if getattr(self, "model", None) is None:
                if hasattr(self, "_build_model"):
                    self.model = self._build_model()
                elif hasattr(self, "forward"):
                    # Agent 자체가 nn.Module인 경우
                    self.model = self
                else:
                    raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되어 있지 않음.")

            model = self.model

            # 다양한 저장 포맷 처리
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

            # 항상 메모리 연결 보장
            self.model = model
            model.eval()

            # 모델이 여전히 None이라면 self 자체를 모델로 설정
            if self.model is None and hasattr(self, "forward"):
                self.model = self

            return True

        except Exception as e:
            return False

    def pretrain(self):
        """Agent별 사전학습 루틴 (모델 생성, 학습, 저장, self.model 연결까지 포함)"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]

        # --------------------------
        # 데이터 로드
        # --------------------------
        X, y, cols = load_dataset(self.ticker, self.agent_id, save_dir=self.data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 🔹 타깃 스케일 조정 복원 - 상승/하락율을 100배로 스케일링
        # 기존: 원본 상승/하락율 그대로 사용 (문제: 너무 작은 값으로 과적합)
        # 수정: ±0.04 → ±4.0으로 스케일링하여 적절한 학습 범위 확보
        y_train *= 100.0
        y_val   *= 100.0

        self.scaler.fit_scalers(X_train, y_train)
        self.scaler.save(self.ticker)

        X_train, y_train = map(torch.tensor, self.scaler.transform(X_train, y_train))
        X_train, y_train = X_train.float(), y_train.float()

        # --------------------------
        # 모델 생성 및 초기화
        # --------------------------
        if getattr(self, "model", None) is None:
            # BaseAgent에 _build_model()이 있다면 호출
            if hasattr(self, "_build_model"):
                self.model = self._build_model()
            else:
                raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되지 않음")

        model = self.model
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # 기존: MSE Loss 사용
        # loss_fn = torch.nn.MSELoss()
        # 수정: Huber Loss 사용 - 이상치에 덜 민감하고 더 안정적인 학습
        # delta=1.0으로 조정 (타겟 스케일링 후 적절한 값)
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

        # 메모리 연결 유지
        self.model = model

        print(f" {self.agent_id} 모델 학습 및 저장 완료: {model_path}")

    # OpenAI API 호출
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
        if not msg_sys or not msg_user:
            raise ValueError("Invalid messages: system or user message is None.")

        # ✅ JSON Schema 보완
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
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    # 추가: Monte Carlo Dropout 기반 불확싱 추정
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

        # 3D 입력 (samples, seq_len, features) → 2D로 변환
        n_samples, seq_len, n_feats = X_train.shape
        X_2d = X_train.reshape(-1, n_feats)
        self.x_scaler = Sx().fit(X_2d) if Sx else None
        self.y_scaler = Sy().fit(y_train.reshape(-1, 1)) if Sy else None

    def transform(self, X, y=None):
        # 3D 입력 (samples, seq_len, features) → 2D로 변환
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
        # 실제 스케일러 객체인지 확인
        if self.y_scaler and self.y_scaler != "None" and hasattr(self.y_scaler, 'inverse_transform'):
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

    def _build_messages_opinion(self, stock_data, target) -> tuple[str, str]:
        """
        BaseAgent.reviewer_draft 가 호출하는 시그니처와 호환.
        stock_data/target으로 ctx를 만들고 기존 ctx-주입 빌더를 호출.
        """
        win = int(agents_info.get(self.agent_id, {}).get("window_size", 40))
        last = getattr(stock_data, "last_price", None)
        snap = {
            "asof_date": datetime.now().strftime("%Y-%m-%d"),
            "last_price": last,
            "currency": getattr(stock_data, "currency", None),
            "window_size": win,
            "feature_cols_preview": (self.feature_cols or [])[:12],
        }

        next_close = getattr(target, "next_close", None)
        pred_return = None
        if last is not None and next_close is not None:
            try:
                pred_return = float(next_close) / float(last) - 1.0
            except Exception:
                pred_return = None

        pred = {
            "pred_close": next_close,
            "pred_return": pred_return,
            "uncertainty": {
                "std": getattr(target, "uncertainty", None),
                "ci95": None,
            },
            "confidence": getattr(target, "confidence", None),
        }

        ctx = {
            "agent_id": self.agent_id,
            "ticker": getattr(stock_data, "ticker", self.ticker),
            "snapshot": snap,
            "prediction": pred,
        }

        # pred_next_close 보장
        self._ensure_pred_next_close(ctx)

        # 최종 프롬프트
        return self._messages_from_prompts_opinion(ctx)