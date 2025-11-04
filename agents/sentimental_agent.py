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
# 수익률 -> 예측 종가(확인 필요)
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
    def __init__(self, **kwargs):
        super().__init__(agent_id="SentimentalAgent", **kwargs)
        cfg = agents_info.get(self.agent_id, {})
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.feature_cols: List[str] = []

    def _build_model(self) -> nn.Module:
        return _LazyLSTMWithStub(hidden_dim=self.hidden_dim, dropout=self.dropout)

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

    # ctx 생성: snapshot + prediction(preview)
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

    # 1) prompts.py에서 system/user 가져와 ctx 주입
    def _messages_from_prompts_opinion(self, ctx: dict) -> tuple[str, str]:
        from prompts import OPINION_PROMPTS
        prompt_set = OPINION_PROMPTS.get(self.agent_id, OPINION_PROMPTS["SentimentalAgent"])
        system_text = prompt_set["system"]
        payload = json.dumps(ctx, ensure_ascii=False)
        user_tpl = prompt_set["user"]
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
            sys_text, user_text = self._messages_from_prompts_opinion(ctx)

            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user",   user_text),
                self._schema_opinion_json()
            )
            # parsed: {"our_prediction": float, "reason": str}
            return parsed


    def _build_messages_opinion(self, ctx: dict) -> tuple[str, str]:
        """
        준비된 ctx(dict)를 prompts.py의 SentimentalAgent opinion 프롬프트에 주입해
        (system_text, user_text)를 반환한다.
        - ctx 즉석 생성/보정 없음 (호출부에서 완성된 ctx를 넘겨줄 것)
        - 중괄호 충돌 방지를 위해 .format() 대신 {context}/{context_json}만 replace
        """
        assert isinstance(ctx, dict), "ctx는 dict로 미리 준비되어 있어야 합니다."

        # 1) 프롬프트 로드
        from prompts import OPINION_PROMPTS
        prompt_set = OPINION_PROMPTS.get(self.agent_id, OPINION_PROMPTS["SentimentalAgent"])
        system_text = prompt_set["system"]

        # 2) ctx 주입 (format 금지, 안전한 치환)
        payload = json.dumps(ctx, ensure_ascii=False)
        user_tpl = prompt_set["user"]
        user_text = user_tpl.replace("{context_json}", payload).replace("{context}", payload)

        return system_text, user_text

    from prompts import REBUTTAL_PROMPTS

    def _build_messages_rebuttal(
        self,
        my_opinion: Opinion,
        target_opinion: Opinion,
        stock_data: StockData,
    ) -> tuple[str, str]:
        """
        준비된 opinion 객체 2개와 stock_data로 ctx를 만들고,
        prompts.py의 REBUTTAL_PROMPTS를 불러 system/user 메시지를 구성한다.
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

        # prompts.py에서 system/user 불러오기 (format → replace)
        prompt_set = REBUTTAL_PROMPTS.get(self.agent_id, REBUTTAL_PROMPTS["SentimentalAgent"])
        system_text = prompt_set["system"]
        payload = json.dumps(ctx, ensure_ascii=False, indent=2)
        user_tpl = prompt_set["user"]
        user_text = user_tpl.replace("{context_json}", payload).replace("{context}", payload)

        return system_text, user_text

    import json
    from prompts import REVISION_PROMPTS

    def _build_messages_revision(
        self,
        my_opinion: Opinion,
        others: list[Opinion],
        rebuttals: Optional[list[Rebuttal]] = None,
        stock_data: Optional[StockData] = None,
    ) -> tuple[str, str]:
        """
        prompts.py의 REVISION_PROMPTS를 사용해 Revision 메시지 구성.
        - my_opinion: 내 의견
        - others: 타 에이전트들의 의견 리스트
        - rebuttals: (선택) 내게 들어온 반박 리스트
        - stock_data: 종목/데이터 정보
        """
        agent_data = getattr(stock_data, self.agent_id, {})
        others_summary = []
        for o in others:
            entry = {
                "agent_id": o.agent_id,
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
                    if r.to_agent_id == self.agent_id and r.from_agent_id == o.agent_id
                ]
                if related:
                    entry["rebuttals_to_me"] = related
            others_summary.append(entry)

        ctx = {
            "ticker": getattr(stock_data, "ticker", "UNKNOWN"),
            "currency": getattr(stock_data, "currency", "USD"),
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

        # prompts.py에서 system/user 구성 (replace로 주입)
        prompt_set = REVISION_PROMPTS.get(self.agent_id, REVISION_PROMPTS["SentimentalAgent"])
        system_text = prompt_set["system"]
        payload = json.dumps(ctx, ensure_ascii=False, indent=2)
        user_tpl = prompt_set["user"]
        user_text = user_tpl.replace("{context_json}", payload).replace("{context}", payload)

        return system_text, user_text
