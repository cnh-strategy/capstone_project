# agents/sentimental_agent.py
from __future__ import annotations
import os
import math
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
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
# 수익률 -> 예측 종가
# ---------------------------
def _compute_pred_fields(last_close: float, yhat_scaled: float, y_scaler, target_mode: str = "log_return"):
    yhat = y_scaler.inverse_transform([[yhat_scaled]])[0, 0] if y_scaler else yhat_scaled

    if target_mode == "log_return":
        pred_return = math.exp(yhat) - 1.0
        pred_next_close = last_close * (1.0 + pred_return)
    elif target_mode == "return":
        pred_return = yhat
        pred_next_close = last_close * (1.0 + pred_return)
    elif target_mode == "price":
        pred_next_close = yhat
        pred_return = (pred_next_close / last_close) - 1.0
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    assert abs((pred_next_close / last_close - 1.0) - pred_return) < 1e-6
    return float(pred_return), float(pred_next_close)


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
        self._stub = nn.Parameter(torch.zeros(1))

    def _lazy_build(self, in_dim: int):
        self.lstm = nn.LSTM(in_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self._inited = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._inited:
            in_dim = int(x.size(-1))
            self._lazy_build(in_dim)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.fc(out)
        return out


# ---------------------------
# SentimentalAgent
# ---------------------------
class SentimentalAgent(BaseAgent):
    """BaseAgent(풀스택)와 호환되는 감성 에이전트"""

    def __init__(self, **kwargs):
        super().__init__(agent_id="SentimentalAgent", **kwargs)
        cfg = agents_info.get(self.agent_id, {})
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.feature_cols: List[str] = []

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
                    print(f"⚠️ 무시된 키: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
                if missing:
                    print(f"⚠️ 누락된 키: {missing[:8]}{'...' if len(missing) > 8 else ''}")
                print(f"✅ 모델 로드 완료: {model_path}")
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
    # 데이터 검색/로딩 (최신 윈도우 반환)
    # ---------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        from agents.base_agent import StockData as _StockData

        ticker = ticker or self.ticker
        self.ticker = ticker
        agent_id = self.agent_id
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")

        if rebuild or not os.path.exists(dataset_path):
            print(f"⚙️ {ticker} {agent_id} dataset {'rebuild requested' if rebuild else 'not found'}. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir, agent_id=agent_id)

        X, y, feature_cols = load_dataset(ticker, agent_id=agent_id, save_dir=self.data_dir)
        self.feature_cols = feature_cols[:]

        X_latest = X[-1:]
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        self.stockdata = _StockData(ticker=ticker)

        try:
            data = yf.download(ticker, period="1d", interval="1d", progress=False)
            self.stockdata.last_price = float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"yfinance 오류(가격): {e}")

        try:
            info = yf.Ticker(ticker).info
            self.stockdata.currency = info.get("currency", "USD")
        except Exception as e:
            print(f"yfinance 오류(통화), 기본값 USD 사용: {e}")
            self.stockdata.currency = "USD"

        print(f"■ {agent_id} StockData 생성 완료 ({ticker}, {self.stockdata.currency})")
        return X_tensor

    # ===========================================================
    # ctx 생성: snapshot + prediction(preview)
    # ===========================================================
    def preview_opinion_ctx(self, ticker: Optional[str] = None, mc_passes: int = 30) -> Dict[str, Any]:
        X_tensor = self.searcher(ticker or self.ticker)

        if getattr(self, "model", None) is None:
            self.model = self._build_model()
            self.model.eval()

        self.load_model()

        snap = self._ctx_snapshot()
        pred = self._ctx_prediction(X_tensor, mc_passes=mc_passes)

        ctx: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "ticker": self.ticker,
            "snapshot": snap,
            "prediction": pred,
        }
        return ctx

    def _ctx_snapshot(self) -> Dict[str, Any]:
        win = int(agents_info.get(self.agent_id, {}).get("window_size", 40))
        feat_prev = (self.feature_cols or [])[:12]
        snap = {
            "asof_date": datetime.now().strftime("%Y-%m-%d"),
            "last_price": getattr(self.stockdata, "last_price", None),
            "currency": getattr(self.stockdata, "currency", None),
            "window_size": win,
            "feature_cols_preview": feat_prev,
        }
        return snap

    def _mc_predict_return(self, X: torch.Tensor, n: int = 30) -> Tuple[float, float]:
        assert X.ndim == 3, "X must be (B,T,F)"
        if getattr(self, "model", None) is None:
            self.model = self._build_model()

        self.model.train()
        preds: List[float] = []
        with torch.no_grad():
            for _ in range(int(max(1, n))):
                y = self.model(X).squeeze(-1)
                preds.append(float(y.cpu().numpy().reshape(-1)[0]))

        self.model.eval()
        mu = float(np.mean(preds)) if preds else 0.0
        std = float(np.std(preds)) if preds else 0.0
        return mu, std

    def _ctx_prediction(self, X_tensor: torch.Tensor, mc_passes: int = 30) -> Dict[str, Any]:
        last = getattr(self.stockdata, "last_price", None)
        mu_ret, std_ret = self._mc_predict_return(X_tensor, n=mc_passes)

        pred_close = None
        if last is not None:
            try:
                pred_close = float(last * (1.0 + float(mu_ret)))
            except Exception:
                pred_close = None

        alpha = 5.0
        confidence = float(1.0 / (1.0 + alpha * max(0.0, std_ret)))

        pred = {
            "pred_close": pred_close,
            "pred_return": float(mu_ret),
            "uncertainty": {
                "std": float(std_ret),
                "ci95": float(1.96 * std_ret),
            },
            "confidence": confidence,
        }
        return pred

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
            "주어진 ctx를 근거로 다음 거래일 종가(next_close)에 대한 해석과 근거(reason)를 작성한다. "
            "1~5번까지 번호를 매겨가며 작성하며 충분하고 구체적인 증거를 포함한다. "
            "반드시 포함할 요소:\n"
            "- 현재가 대비 예상 변화(비율 또는 방향)와 그 근거\n"
            "- 긍정/부정 기사 비율(또는 감정 점수)과 기간(예: 최근 7일/30일) 비교\n"
            "- 핵심 이벤트/뉴스(가능하면 날짜·출처·키워드)와 그 영향 해석\n"
            "- 여론 추세 변화(개선/악화)와 강도, 노이즈/한계에 대한 주의점\n"
            "- 모델 신호(예: 신뢰도/불확실성)가 있다면 수치로 간단히 해석\n"
            "전문용어(예: attention, embedding, 회귀계수 등) 대신 일반 투자자가 이해하기 쉬운 표현을 사용하라. "
            "출력은 반드시 하나의 JSON 객체로만 반환하며, 키는 "
            "{\"next_close\": number, \"reason\": string} 만 허용한다."
        )
        # pred_return, pred_next_close = _compute_pred_fields(
        #     last_close=last,
        #     yhat_scaled=target.yhat_scaled,  # 모델 원시 출력 (target 안에 yhat_scaled 있다고 추정)
        #     y_scaler=self.scaler.y_scaler if getattr(self.scaler, "y_scaler", None) else None,
        #     target_mode="log_return"  # 실제 타깃에 맞게 설정
        # )
        user = (
            "컨텍스트:\n"
            f"- ticker: {self.ticker}\n"
            f"- last_price: {last:.4f}\n"
            f"- pred_next_close: {pred_next_close:.4f}\n"
            f"- pred_return: {pred_return:.6f}\n"
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