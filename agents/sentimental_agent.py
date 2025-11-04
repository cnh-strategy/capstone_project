# agents/sentimental_agent.py
from __future__ import annotations
import os
import math
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yfinance as yf

from agents.base_agent import BaseAgent
# 타입 힌트용 (런타임 의존 최소화)
try:
    from agents.base_agent import StockData, Target, Opinion, Rebuttal
except Exception:
    StockData = Any  # type: ignore
    Target = Any     # type: ignore
    Opinion = Any    # type: ignore
    Rebuttal = Any   # type: ignore

# 프롬프트 세트
try:
    from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
except Exception:
    OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS = {}, {}, {}

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

# 수익률 -> 예측 종가(검증용 유틸)
def _compute_pred_fields(
    last_close: float,
    yhat_scaled: float,
    y_scaler,
    target_mode: str = "log_return",
) -> Tuple[float, float]:
    """
    모델 출력(yhat_scaled)을 원복/해석해 (pred_return, pred_next_close) 반환
    """
    # yhat = y_scaler.inverse_transform([[yhat_scaled]])[0, 0] if y_scaler else yhat_scaled
    yhat = model(x)              # (B, 1)
    loss = F.mse_loss(yhat.squeeze(-1), y)   # y shape -> (B,)

    if target_mode == "log_return":
        pred_return = math.exp(float(yhat)) - 1.0
        pred_next_close = float(last_close) * (1.0 + pred_return)
    elif target_mode == "return":
        pred_return = float(yhat)
        pred_next_close = float(last_close) * (1.0 + pred_return)
    elif target_mode == "price":
        pred_next_close = float(yhat)
        pred_return = (pred_next_close / float(last_close)) - 1.0
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    # 일관성 체크
    assert abs((pred_next_close / float(last_close) - 1.0) - pred_return) < 1e-6
    return float(pred_return), float(pred_next_close)

# Lazy LSTM (stub 파라미터 포함)
class _LazyLSTMWithStub(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self._inited = False
        self.lstm: Optional[nn.LSTM] = None
        self.fc: Optional[nn.Linear] = None
        self._stub = nn.Parameter(torch.zeros(1))  # 로딩 호환용 더미 파라미터

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
        out = F.dropout(out, p=self.dropout_p, training=self.training)  # MC Dropout 용
        out = self.fc(out)
        return out


# SentimentalAgent
class SentimentalAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(agent_id="SentimentalAgent", **kwargs)
        cfg = agents_info.get(self.agent_id, {})
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.feature_cols: List[str] = []

    # BaseAgent 호환 모델 생성
    def _build_model(self) -> nn.Module:
        return _LazyLSTMWithStub(hidden_dim=self.hidden_dim, dropout=self.dropout)

    # PT 체크포인트 로드 (유연 로더)
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
                print(f"■ {self.agent_id} 모델 새로 생성됨 (로드 전 초기화)")

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

    # 데이터 검색/로딩 (최신 윈도우 반환)
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        # 순환 import 회피
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

        # 가격/통화 정보 보강
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

    # --- 컨텍스트 스냅샷 ---
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

    # --- MC Dropout 예측 ---
    def _mc_predict_return(self, X: torch.Tensor, n: int = 30) -> Tuple[float, float]:
        assert X.ndim == 3, "X must be (B,T,F)"
        if getattr(self, "model", None) is None:
            self.model = self._build_model()

        self.model.train()  # MC dropout
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

        # 간단 confidence (표준편차 역수)
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

    # ctx 내 pred_next_close 보장(상호 호환)
    def _ensure_pred_next_close(self, ctx: Dict[str, Any]) -> None:
        snap = ctx.get("snapshot", {}) or {}
        pred = ctx.get("prediction", {}) or {}
        last = snap.get("last_price")
        pred_close = pred.get("pred_close")
        pred_ret = pred.get("pred_return")

        if pred_close is None and last is not None and pred_ret is not None:
            try:
                pred_close = float(last * (1.0 + float(pred_ret)))
            except Exception:
                pred_close = None

        pred["pred_next_close"] = pred_close  # 호환 키
        ctx["prediction"] = pred

    # --- 미리보기용 ctx 생성 ---
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
        # 호환 보장
        self._ensure_pred_next_close(ctx)
        return ctx

    # 프롬프트 빌더 (ctx 직접 주입 버전)
    def _messages_from_prompts_opinion(self, ctx: dict) -> tuple[str, str]:
        prompt_set = OPINION_PROMPTS.get(self.agent_id, OPINION_PROMPTS.get("SentimentalAgent", {}))
        system_text = prompt_set.get("system", "")
        payload = json.dumps(ctx, ensure_ascii=False)
        user_tpl = prompt_set.get("user", "{context_json}")
        user_text = user_tpl.replace("{context_json}", payload).replace("{context}", payload)
        return system_text, user_text

    def _schema_opinion_json(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "our_prediction": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["our_prediction", "reason"],
            "additionalProperties": False,
        }

    # LLM 호출 헬퍼
    def run_opinion_with_prompts(self, ctx: dict) -> dict:
        # 방어적으로 호환 키 채움
        self._ensure_pred_next_close(ctx)
        sys_text, user_text = self._messages_from_prompts_opinion(ctx)
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user",   user_text),
            self._schema_opinion_json(),
        )
        # parsed: {"our_prediction": float, "reason": str}
        return parsed

    # BaseAgent.reviewer_* 호환 시그니처 버전
    def _build_messages_opinion(self, stock_data, target) -> tuple[str, str]:
        """
        BaseAgent.reviewer_draft()에서 호출되는 시그니처와 호환
        stock_data/target로 ctx를 만들고, 기존 ctx 기반 빌더를 재사용해 (system, user)를 반환
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

        # prediction (BaseAgent.Target: next_close/uncertainty/confidence 가정)
        next_close = getattr(target, "next_close", None)
        pred_return = None
        if last is not None and next_close is not None:
            try:
                pred_return = float(next_close / float(last) - 1.0)
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
        self._ensure_pred_next_close(ctx)
        return self._messages_from_prompts_opinion(ctx)

    # Rebuttal / Revision 메시지 빌더
    def _build_messages_rebuttal(
        self,
        my_opinion: Opinion,
        target_opinion: Opinion,
        stock_data: StockData,
    ) -> tuple[str, str]:
        """
        준비된 opinion 객체 2개와 stock_data로 ctx를 만들고,
        REBUTTAL_PROMPTS에서 system/user 메시지를 구성한다.
        """
        assert hasattr(stock_data, self.agent_id), f"{self.agent_id} 데이터 누락"

        agent_data = getattr(stock_data, self.agent_id, {})
        ctx = {
            "ticker": getattr(stock_data, "ticker", "UNKNOWN"),
            "currency": getattr(stock_data, "currency", "USD"),
            "me": {
                "agent_id": self.agent_id,
                "our_prediction": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:1500],
                "uncertainty": float(my_opinion.target.uncertainty),
                "confidence": float(my_opinion.target.confidence),
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "our_prediction": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:1500],
                "uncertainty": float(target_opinion.target.uncertainty),
                "confidence": float(target_opinion.target.confidence),
            },
            "feature_cols": agent_data.get("feature_cols", []),
        }

        # 필요시 최근 14일 시계열 데이터 추가
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-14:]
            else:
                ctx[col] = [values]

        prompt_set = REBUTTAL_PROMPTS.get(self.agent_id, REBUTTAL_PROMPTS.get("SentimentalAgent", {}))
        system_text = prompt_set.get("system", "")
        payload = json.dumps(ctx, ensure_ascii=False, indent=2)
        user_tpl = prompt_set.get("user", "{context_json}")
        user_text = user_tpl.replace("{context_json}", payload).replace("{context}", payload)

        return system_text, user_text

    def _build_messages_revision(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: Optional[List[Rebuttal]] = None,
        stock_data: Optional[StockData] = None,
    ) -> tuple[str, str]:
        """
        REVISION_PROMPTS를 사용해 Revision 메시지 구성.
        """
        agent_data = getattr(stock_data, self.agent_id, {}) if stock_data is not None else {}
        others_summary = []
        for o in others:
            entry = {
                "agent_id": getattr(o, "agent_id", "UNKNOWN"),
                "predicted_price": float(o.target.next_close),
                "confidence": float(o.target.confidence),
                "uncertainty": float(o.target.uncertainty),
                "reason": str(o.reason)[:600],
            }
            # 내게 들어온 rebuttal만 추가
            if rebuttals:
                related = [
                    {"stance": r.stance, "message": r.message}
                    for r in rebuttals
                    if getattr(r, "to_agent_id", None) == self.agent_id
                    and getattr(r, "from_agent_id", None) == getattr(o, "agent_id", None)
                ]
                if related:
                    entry["rebuttals_to_me"] = related
            others_summary.append(entry)

        ctx = {
            "ticker": getattr(stock_data, "ticker", "UNKNOWN") if stock_data is not None else "UNKNOWN",
            "currency": getattr(stock_data, "currency", "USD") if stock_data is not None else "USD",
            "agent_type": self.agent_id,
            "my_opinion": {
                "predicted_price": float(my_opinion.target.next_close),
                "confidence": float(my_opinion.target.confidence),
                "uncertainty": float(my_opinion.target.uncertainty),
                "reason": str(my_opinion.reason)[:1000],
            },
            "others_summary": others_summary,
        }

        # 최근 14일치 시계열 추가
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-14:]
            else:
                ctx[col] = [values]

        prompt_set = REVISION_PROMPTS.get(self.agent_id, REVISION_PROMPTS.get("SentimentalAgent", {}))
        system_text = prompt_set.get("system", "")
        payload = json.dumps(ctx, ensure_ascii=False, indent=2)
        user_tpl = prompt_set.get("user", "{context_json}")
        user_text = user_tpl.replace("{context_json}", payload).replace("{context}", payload)

        return system_text, user_text

    def get_opinion(self, idx: int, ticker: str):
        df = self.searcher(ticker)                 # 학습과 동일 규칙으로 만든 df
        last_close = float(df["Close"].iloc[-2])   # ← 중요: y = shift(-1) 사용했으므로 X는 마지막-1까지
                                                  # df.iloc[-1]는 타깃 없는 미래행일 수 있어 -2 사용
        x_last = self._make_last_window(df)        # (1, W, F)

        pred_return, pred_close = self.predict_next(x_last, last_close)

        return Opinion(
            agent_id=self.agent_id,
            target=Target(
                next_close=pred_close,
                uncertainty=self._estimate_uncertainty(x_last),  # 기존 로직
                confidence=self._calibrate_confidence(pred_return)
            ),
            reason=self._compose_reason(
                last_close=last_close,
                pred_close=pred_close,
                pred_return=pred_return
            )
        )