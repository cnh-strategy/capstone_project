# agents/base_agent.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import yfinance as yf

try:
    from config.agents import dir_info
except Exception:
    dir_info = {
        "data_root": "data",
        "processed_dir": os.path.join("data", "processed"),
        "raw_dir": os.path.join("data", "raw"),
        "preview_dir": os.path.join("data", "preview"),
        "models_dir": "models",
    }

from core.data_set import load_dataset


@dataclass
class StockData:
    ticker: str
    currency: Optional[str] = None
    last_price: Optional[float] = None


@dataclass
class Target:
    next_close: Optional[float] = None
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    feature_cols: Optional[List[str]] = None
    importances: Optional[Dict[str, float]] = None


class BaseAgent:
    agent_id: str = "BaseAgent"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        **kwargs,  # absorb ticker, agent_id, verbose
    ):
        self.verbose: bool = bool(kwargs.pop("verbose", False))
        override_agent_id = kwargs.pop("agent_id", None)
        if override_agent_id:
            self.agent_id = str(override_agent_id)
        self._init_ticker: Optional[str] = kwargs.pop("ticker", None)

        self.data_dir = data_dir or dir_info.get("processed_dir", os.path.join("data", "processed"))
        self.models_dir = models_dir or dir_info.get("models_dir", "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.stockdata: Optional[StockData] = None
        self.model = None
        self.feature_cols: Optional[List[str]] = None
        self.window_size: Optional[int] = None

    # -----------------------------
    # Public API
    # -----------------------------
    def searcher(self, ticker: str) -> np.ndarray:
        """X만 반환(모델 인퍼런스용). y, feature_cols는 멤버에 저장."""
        X, y, feature_cols = load_dataset(
            ticker=ticker,
            agent_id=self.agent_id,
            save_dir=self.data_dir,
        )
        self.feature_cols = feature_cols
        self.window_size = X.shape[1]

        # 명시적으로 auto_adjust 지정하여 경고 제거
        data = yf.download(ticker, period="1d", interval="1d", progress=False, auto_adjust=True)
        last_close = float(data["Close"].iloc[-1].item())
        self.stockdata = StockData(ticker=ticker, currency="USD", last_price=last_close)
        if self.verbose:
            print(f"■ {self.agent_id} StockData 생성 완료 ({ticker}, {self.stockdata.currency})")
        return X

    def predict(self, X: np.ndarray, current_price: Optional[float] = None) -> Target:
        model_path = os.path.join(self.models_dir, f"{self._safe_ticker()}_{self.agent_id}.pt")
        self.load_model(model_path=model_path)
        if self.verbose:
            print(f"■ {self.agent_id} 모델 자동 로드 시도...")

        next_close, uncertainty, confidence = self._predict_impl(X, current_price=current_price)
        return Target(
            next_close=next_close,
            uncertainty=uncertainty,
            confidence=confidence,
            feature_cols=self.feature_cols,
            importances=None,
        )

    # -----------------------------
    # Must be implemented by child
    # -----------------------------
    def _predict_impl(self, X: np.ndarray, current_price: Optional[float]) -> Tuple[float, float, float]:
        raise NotImplementedError

    def load_model(self, model_path: Optional[str] = None) -> None:
        raise NotImplementedError

    def save_model(self, model_path: Optional[str] = None) -> None:
        raise NotImplementedError

    # -----------------------------
    # Helpers
    # -----------------------------
    def _safe_ticker(self) -> str:
        if self.stockdata and self.stockdata.ticker:
            return self.stockdata.ticker.replace("/", "_").replace(":", "_")
        if self._init_ticker:
            return self._init_ticker.replace("/", "_").replace(":", "_")
        return "UNKNOWN"
