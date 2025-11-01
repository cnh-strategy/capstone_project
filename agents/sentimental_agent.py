# agents/sentimental_agent.py
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from agents.base_agent import BaseAgent

try:
    from config.agents import agents_info, dir_info
except Exception:
    agents_info = {"SentimentalAgent": {"hidden_dim": 64, "dropout": 0.2}}
    dir_info = {"models_dir": "models"}


class _SentimentalNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)        # (B, T, H)
        out = out[:, -1, :]          # (B, H)
        out = self.dropout(out)
        out = self.fc(out)           # (B, 1)  next-day return
        return out


class SentimentalAgent(BaseAgent):
    agent_id = "SentimentalAgent"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        **kwargs,  # absorbs ticker, agent_id, verbose, etc.
    ):
        super().__init__(data_dir=data_dir, models_dir=models_dir, **kwargs)
        cfg = agents_info.get(self.agent_id, {})
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.dropout = float(cfg.get("dropout", 0.2))

    # BaseAgent가 predict()에서 호출
    def _predict_impl(self, X: np.ndarray, current_price: Optional[float]) -> Tuple[float, float, float]:
        if self.model is None:
            input_dim = X.shape[2]
            self._build_model(input_dim)

        xt = torch.from_numpy(X[-1:]).float()  # 마지막 윈도우 하나로 추론
        self.model.eval()
        with torch.no_grad():
            yhat_ret = float(self.model(xt).cpu().numpy().reshape(-1)[0])

        price_base = current_price
        if price_base is None and self.stockdata and self.stockdata.last_price is not None:
            price_base = float(self.stockdata.last_price)

        next_close = float(price_base * (1.0 + yhat_ret)) if price_base is not None else None
        uncertainty = float(abs(yhat_ret))  # placeholder
        confidence = float(max(0.0, 1.0 - min(1.0, abs(yhat_ret))))  # placeholder
        return next_close, uncertainty, confidence

    # -----------------------------
    # Model I/O
    # -----------------------------
    def _build_model(self, input_dim: int) -> None:
        self.model = _SentimentalNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def _extract_state_dict(self, ckpt_obj) -> Optional[OrderedDict]:
        """
        다양한 저장 포맷 대응:
        - {"state_dict": OrderedDict(...)}
        - {"model_state_dict": OrderedDict(...)}
        - OrderedDict(...) 자체 (torch.save(model.state_dict()))
        """
        if isinstance(ckpt_obj, OrderedDict):
            return ckpt_obj
        if isinstance(ckpt_obj, dict):
            if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], (dict, OrderedDict)):
                return ckpt_obj["state_dict"]
            if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], (dict, OrderedDict)):
                return ckpt_obj["model_state_dict"]
            # dict인데 키들이 파라미터처럼 보이면 그대로 사용
            if all(isinstance(k, str) for k in ckpt_obj.keys()):
                # 값이 Tensor/ndarray/Parameter면 state_dict로 간주
                like_state = True
                for v in ckpt_obj.values():
                    if not (torch.is_tensor(v) or hasattr(v, "shape")):
                        like_state = False
                        break
                if like_state:
                    return OrderedDict(ckpt_obj)
        return None

    def load_model(self, model_path: Optional[str] = None) -> None:
        input_dim = len(self.feature_cols) if self.feature_cols else None
        models_dir = dir_info.get("models_dir", "models")
        os.makedirs(models_dir, exist_ok=True)
        if not model_path:
            model_path = os.path.join(models_dir, f"{self._safe_ticker()}_{self.agent_id}.pt")

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            # 1) state_dict 추출
            state_dict = self._extract_state_dict(ckpt)

            # 2) input_dim 추론
            if input_dim is None:
                meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
                if isinstance(meta, dict) and "input_dim" in meta:
                    input_dim = int(meta["input_dim"])
            if input_dim is None:
                # meta도 없고 feature_cols도 없으면 로드 불가 → 사용자에게 재학습/재저장을 유도
                raise ValueError(
                    "Cannot infer input_dim for model (no meta['input_dim'] and no feature_cols). "
                    "Delete the old model file or save a new one with meta."
                )

            # 3) 모델 빌드 후 로드/폴백
            self._build_model(input_dim)
            if state_dict is not None:
                self.model.load_state_dict(state_dict, strict=False)
                if self.verbose:
                    print(f"✅ 모델 로드(유연 포맷 지원): {model_path}")
            else:
                # 알 수 없는 포맷 → 새 모델로 폴백
                if self.verbose:
                    print(f"⚠️ 알 수 없는 체크포인트 포맷. 새 모델로 대체: {model_path}")
        else:
            if input_dim is None:
                raise ValueError("feature_cols가 없어 모델 입력 차원을 알 수 없습니다.")
            self._build_model(input_dim)
            if self.verbose:
                print(f"⚠️ 가중치 파일 없음. 새 모델 생성(경로: {model_path})")

        self.model.eval()

    def save_model(self, model_path: Optional[str] = None) -> None:
        models_dir = dir_info.get("models_dir", "models")
        os.makedirs(models_dir, exist_ok=True)
        if not model_path:
            model_path = os.path.join(models_dir, f"{self._safe_ticker()}_{self.agent_id}.pt")

        meta = {
            "agent_id": self.agent_id,
            "input_dim": self.model.lstm.input_size if hasattr(self.model, "lstm") else None,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }
        torch.save({"state_dict": self.model.state_dict(), "meta": meta}, model_path)
        if self.verbose:
            print(f"💾 모델 저장: {model_path}")
