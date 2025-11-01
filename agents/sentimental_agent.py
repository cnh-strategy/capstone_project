# agents/sentimental_agent.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import yfinance as yf

from agents.base_agent import BaseAgent
try:
    from config.agents import agents_info, dir_info
except Exception:
    agents_info = {
        "SentimentalAgent": {
            "window_size": 40,
            "hidden_dim": 128,
            "dropout": 0.2,
            "epochs": 30,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "x_scaler": "StandardScaler",
            "y_scaler": "StandardScaler",
            "gamma": 0.3,
            "delta_limit": 0.05,
        }
    }
    dir_info = {
        "data_dir": "data",
        "model_dir": "models",
        "scaler_dir": os.path.join("models", "scalers"),
    }

from core.data_set import build_dataset, load_dataset


# ---------------------------
# Lazy LSTM (stub 파라미터 포함)
# ---------------------------
class _LazyLSTMWithStub(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self._inited = False
        self.lstm: Optional[nn.LSTM] = None
        self.fc: Optional[nn.Linear] = None
        # ✔️ 파라미터 이터레이터가 비지 않도록 보장
        self._stub = nn.Parameter(torch.zeros(1))

    def _lazy_build(self, in_dim: int):
        self.lstm = nn.LSTM(in_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self._inited = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        if not self._inited:
            in_dim = int(x.size(-1))
            self._lazy_build(in_dim)
        out, _ = self.lstm(x)      # (B, T, H)
        out = out[:, -1, :]        # (B, H)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.fc(out)         # (B, 1) → 다음날 수익률 예측
        return out


class SentimentalAgent(BaseAgent):
    """BaseAgent(풀스택)와 호환되는 감성 에이전트"""

    def __init__(self, **kwargs):
        super().__init__(agent_id="SentimentalAgent", **kwargs)
        cfg = agents_info.get(self.agent_id, {})
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        # demo에서 참고하는 필드가 반드시 존재하도록 초기화
        self.feature_cols: List[str] = []

    # ---------------------------
    # BaseAgent가 호출하는 훅
    # ---------------------------
    def _build_model(self) -> nn.Module:
        return _LazyLSTMWithStub(hidden_dim=self.hidden_dim, dropout=self.dropout)

    # ---------------------------
    # 옛 체크포인트(Transformer 등)도 느슨하게 로드
    # ---------------------------
    def load_model(self, model_path: Optional[str] = None) -> bool:
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            print(f"■ 모델 파일 없음: {model_path}")
            if getattr(self, "model", None) is None:
                self.model = self._build_model()
            self.model.eval()
            return False

        try:
            ckpt = torch.load(model_path, map_location="cpu")

            if getattr(self, "model", None) is None:
                self.model = self._build_model()
                print(f"■ {self.agent_id} 모델 새로 생성됨 (로드 전 초기화).")

            state_dict = None
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
                if state_dict is None and all(isinstance(k, str) for k in ckpt.keys()):
                    state_dict = ckpt
            elif isinstance(ckpt, nn.Module):
                state_dict = ckpt.state_dict()

            if state_dict is not None:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if unexpected:
                    print(f"⚠️ 무시된 키(예전 구조): {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
                if missing:
                    print(f"⚠️ 새 구조 전용 키(체크포인트에 없음): {missing[:8]}{'...' if len(missing)>8 else ''}")
                print(f"✅(loose)) 모델 로드 시도 완료: {model_path}")
            else:
                print("⚠️ 알 수 없는 체크포맷 → 새 모델 그대로 사용")

            self.model.eval()
            return True

        except Exception as e:
            print(f"■ 모델 로드 실패: {model_path}")
            print(f"오류 내용: {e}")
            if getattr(self, "model", None) is None:
                self.model = self._build_model()
            self.model.eval()
            return False

    # ---------------------------
    # demo 코드가 ag.feature_cols를 참조하므로
    # BaseAgent.searcher()를 오버라이드하여 반드시 세팅
    # ---------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        agent_id = self.agent_id
        if ticker is None:
            ticker = self.ticker
        self.ticker = ticker

        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {agent_id} dataset not found. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir)

        # X, y, feature_cols 로드 및 보관
        X, y, feature_cols = load_dataset(ticker, agent_id=agent_id, save_dir=self.data_dir)
        self.feature_cols = feature_cols[:]  # ★ 반드시 세팅

        # 최신 윈도우 텐서 반환 (BaseAgent.predict와 호환)
        X_latest = X[-1:]
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # StockData 구성 + 현재가/통화
        self.stockdata = self.stockdata or type(self).StockData if hasattr(type(self), "StockData") else None
        # stockdata는 base_agent의 dataclass를 사용해야 하므로 아래처럼 생성
        from agents.base_agent import StockData as _StockData
        self.stockdata = _StockData(ticker=ticker)

        try:
            data = yf.download(ticker, period="1d", interval="1d")
            # pandas 경고 회피
            self.stockdata.last_price = float(data["Close"].iloc[-1].item())
        except Exception:
            print("yfinance 오류 발생")

        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception as e:
            print(f"yfinance 오류 발생, 통화 기본값 사용: {e}")
            self.stockdata.currency = "USD"

        print(f"■ {agent_id} StockData 생성 완료 ({ticker}, {self.stockdata.currency})")
        return X_tensor

    # ---------------------------
    # LLM 메시지 빌더 3종
    # ---------------------------
    def _build_messages_opinion(self, stock_data, target) -> Tuple[str, str]:
        last = getattr(self.stockdata, "last_price", None)
        try:
            pred_ret = float(target.next_close / float(last) - 1.0) if (last and target and target.next_close) else None
        except Exception:
            pred_ret = None

        sys = (
            "너는 감성/뉴스 중심의 단기 주가 분석가다. "
            "예측 수익률, 불확실성(표준편차), 신뢰도를 근거로 2~3문장 reason만 생성하라."
        )
        user = (
            "컨텍스트:\n"
            f"- ticker: {self.ticker}\n"
            f"- last_price: {last}\n"
            f"- pred_next_close: {getattr(target, 'next_close', None)}\n"
            f"- pred_return: {pred_ret}\n"
            f"- uncertainty_std: {getattr(target, 'uncertainty', None)}\n"
            f"- confidence: {getattr(target, 'confidence', None)}\n"
            "→ reason만 출력"
        )
        return sys, user

    def _build_messages_rebuttal(self, my_opinion, target_opinion, stock_data) -> Tuple[str, str]:
        sys = (
            "너는 금융 토론 보조자다. 상대 의견의 수치·근거를 검토해 한 문단 이내로 "
            "REBUT 또는 SUPPORT 메시지를 생성하라."
        )
        user = (
            "컨텍스트:\n"
            f"- 내 의견: {my_opinion}\n"
            f"- 상대 의견: {target_opinion}\n"
            f"- 현재가: {getattr(self.stockdata, 'last_price', None)}\n"
            "→ stance와 message를 일관되게 구성"
        )
        return sys, user

    def _build_messages_revision(self, my_opinion, others, rebuttals, stock_data) -> Tuple[str, str]:
        sys = "너는 금융 분석가다. 토론 결과를 반영하여 2~3문장으로 수정 사유(reason)를 간결히 작성하라."
        user = (
            "컨텍스트:\n"
            f"- 내 의견: {my_opinion}\n"
            f"- 타 의견 수: {len(others)}\n"
            f"- 반박/지지 수: {len(rebuttals)}\n"
            f"- 현재가: {getattr(self.stockdata, 'last_price', None)}\n"
            "→ reason만 출력"
        )
        return sys, user
