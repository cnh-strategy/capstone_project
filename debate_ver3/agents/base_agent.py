# debate_ver3/agents/base_agent.py
# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스  (NaN/Inf Ultra-Safe v2025-10-31C)
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
import pandas as pd  # ✅ SentimentalAgent 스냅샷/파생치 주입에 필요

from debate_ver3.config.agents import agents_info, dir_info
from debate_ver3.core.data_set import build_dataset, load_dataset


# -----------------------------
# 데이터 구조 정의
# -----------------------------
@dataclass
class Target:
    """예측 목표 + 불확실성
    - next_close: 다음 거래일 종가 (항상 '가격' 단위)
    - uncertainty: 표준편차 σ (가격 단위)
    - confidence: 간이 신뢰도 (1/σ)
    - pred_return: 예측 수익률 r (옵션)
    - calibrated_prob_up: 상승확률 근사 (옵션)
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    feature_cols: Optional[List[str]] = None
    importances: Optional[List[float]] = None
    pred_return: Optional[float] = None
    calibrated_prob_up: Optional[float] = None


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
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.need_training = need_training
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.ticker = ticker

        # 🔹 스케일러 유틸
        self.scaler = DataScaler(agent_id)

        # 모델 폴백 우선순위 (LLM)
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [m for m in self.preferred_models if m != model]

        # API 키
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
    # 내부 유틸: NaN/inf 안전 처리
    # -----------------------------
    @staticmethod
    def _to_none_if_nan(v):
        try:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            if isinstance(v, (np.floating,)):
                vf = float(v)
                if np.isnan(vf) or np.isinf(vf):
                    return None
                return vf
            return v
        except Exception:
            return None

    @staticmethod
    def _nan_to_num_array(a, nan=0.0):
        return np.nan_to_num(a, nan=nan, posinf=nan, neginf=nan)

    @staticmethod
    def _finite(x, fb):
        try:
            xf = float(x)
            return fb if (np.isnan(xf) or np.isinf(xf)) else xf
        except Exception:
            return fb

    def _debug_check_model_nans(self):
        """가중치 NaN/Inf 점검(디버그용)"""
        try:
            model = getattr(self, "model", None)
            if isinstance(model, nn.Module) and (model is not self):
                pass  # 그대로 사용
            elif hasattr(self, "net") and isinstance(self.net, nn.Module):
                model = self.net
            else:
                model = self

            bad = []
            for name, p in model.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    bad.append(name)
            if bad and self.verbose:
                print("⚠️ Model has NaN/Inf in params:", bad)
        except Exception:
            pass

    # -----------------------------
    # 데이터 수집/스냅샷
    # -----------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """
        - CSV 없으면 build_dataset()으로 생성
        - 마지막 window를 torch.Tensor로 반환
        - SentimentalAgent 스냅샷(마지막 시점 값) + 파생치(trend/vol/shock) 계산
        - feature_cols / asof_date 메타 저장
        - ✅ NaN/Inf 원천 차단
        """
        if ticker is None:
            ticker = self.ticker

        dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_id}_dataset.csv")

        # 데이터셋 생성
        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {self.agent_id} dataset not found. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir)

        # 로드
        X, y, feature_cols = load_dataset(ticker, agent_id=self.agent_id, save_dir=self.data_dir)

        # ✅ 원천 차단 (전 구간)
        X = self._nan_to_num_array(X, nan=0.0)
        if y is not None:
            y = self._nan_to_num_array(y, nan=0.0)

        # StockData
        self.stockdata = StockData()
        self.stockdata.agent_id = self.agent_id
        self.stockdata.ticker = ticker
        self.stockdata.X = X
        self.stockdata.y = y
        self.stockdata.feature_cols = feature_cols

        # 마지막 윈도우 → 다시 한 번 강제 정리
        X_latest = X[-1:]  # (1, T, F)
        X_latest = self._nan_to_num_array(X_latest, nan=0.0)
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        if self.verbose:
            n_nans = int(np.isnan(X_latest).sum())
            n_infs = int(np.isinf(X_latest).sum())
            print(f"[searcher] X_latest NaN/Inf: ({n_nans}, {n_infs})")

        # --- 스냅샷(마지막 시점의 feature: value) ---
        try:
            df_last = pd.DataFrame(X_latest[0], columns=feature_cols)
            df_last = (
                df_last
                .replace([np.inf, -np.inf], np.nan)
                .ffill()            # ✅ fillna(method="ffill") → ffill()
                .fillna(0.0)
            )

            last_row_dict = {
                k: self._to_none_if_nan(float(v)) if isinstance(v, (int, float, np.floating)) else v
                for k, v in df_last.iloc[-1].to_dict().items()
            }
        except Exception:
            last_row_dict = {}

        # --- sentiment 파생치(fallback) 계산 ---
        try:
            senti_idx = None
            for cand in ["news_sentiment", "sentiment_mean"]:
                if cand in feature_cols:
                    senti_idx = feature_cols.index(cand)
                    break

            if senti_idx is not None and X_latest.shape[1] >= 1:
                s = X_latest[0, :, senti_idx].astype(float)  # 이미 nan→0
                # today
                if last_row_dict.get("news_sentiment") is None and "sentiment_mean" in last_row_dict:
                    last_row_dict["news_sentiment"] = last_row_dict["sentiment_mean"]
                # 7일 변동성
                if last_row_dict.get("sentiment_volatility_7d") is None and s.size >= 7:
                    last_row_dict["sentiment_volatility_7d"] = float(np.std(s[-7:], ddof=1))
                # 7일 추세
                if last_row_dict.get("sentiment_trend_7d") is None and s.size >= 7:
                    x = np.arange(7, dtype=float)
                    y7 = s[-7:].astype(float)
                    last_row_dict["sentiment_trend_7d"] = float(np.polyfit(x, y7, 1)[0])
                # 30일 쇼크 z-score
                if last_row_dict.get("sentiment_shock_score") is None:
                    w = s[-30:] if s.size >= 30 else s
                    if w.size >= 2:
                        mu = float(np.mean(w)); sd = float(np.std(w, ddof=1)) + 1e-6
                        last_row_dict["sentiment_shock_score"] = float((s[-1] - mu) / sd)

            # 별칭 강제 매핑
            def _alias(dst, *srcs):
                if last_row_dict.get(dst) is None:
                    for s in srcs:
                        if s in last_row_dict and last_row_dict[s] is not None:
                            last_row_dict[dst] = last_row_dict[s]; break

            _alias("news_sentiment", "sentiment_mean", "finbert_t")
            _alias("sentiment_trend_7d", "sentiment_trend", "finbert_tr30")
            _alias("sentiment_volatility_7d", "sentiment_vol")
            _alias("sentiment_shock_score", "news_shock")

            last_row_dict.setdefault("news_count_1d", 0)
            last_row_dict.setdefault("news_count_7d", 0)

            # 최종 한 번 더 정리
            for k, v in list(last_row_dict.items()):
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    last_row_dict[k] = None
        except Exception as e:
            if self.verbose:
                print("[searcher] sentiment fallback calc skipped:", repr(e))

        # 최종 스냅샷 주입
        setattr(self.stockdata, "SentimentalAgent", last_row_dict)
        setattr(self.stockdata, "feature_cols", feature_cols)
        setattr(self.stockdata, "asof_date", str(pd.Timestamp.today().date()))

        # 최신 종가(yfinance)
        try:
            data = yf.download(ticker, period="1d", interval="1d", progress=False, auto_adjust=True)
            if data is not None and not data.empty:
                close_last = data["Close"].iloc[-1]
                if hasattr(close_last, "item"):
                    close_last = close_last.item()
                self.stockdata.last_price = float(close_last)
            else:
                self.stockdata.last_price = self.stockdata.last_price or 100.0
        except Exception:
            self.stockdata.last_price = self.stockdata.last_price or 100.0

        # 캐시
        self._last_X = X_tensor

        return X_tensor

    # -----------------------------
    # 예측 (MC Dropout)
    # -----------------------------
    def predict(self, X, n_samples: int = 30, current_price: float = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성
        - output_type ∈ {'return','log_return','price'}
        - 항상 '가격' 단위 next_close/uncertainty 반환
        - pred_return, calibrated_prob_up 채움
        - ✅ NaN/Inf 방어 로직(다단계 + 반환 직전 강제-유효화) 포함
        """
        # (옵션) 가중치 로드 + NaN 파라미터 점검
        try:
            self.load_model()
        except Exception:
            pass
        self._debug_check_model_nans()

        # 설정
        cfg = agents_info.get(self.agent_id, {})
        output_type   = str(cfg.get("output_type", "return"))   # 'return' | 'log_return' | 'price'
        return_index  = int(cfg.get("return_index", -1))        # 다차원 출력 시 인덱스
        has_prob_head = bool(cfg.get("has_prob_head", False))   # 별도 확률헤드

        # 스케일러
        self.scaler.load(self.ticker)
        if self.scaler.x_scaler is None and self.verbose:
            self._p("⚠️ X scaler not loaded. Using raw features.")
        if self.scaler.y_scaler is None and self.verbose:
            self._p("ℹ️ Y scaler not loaded (inverse_y disabled).")

        # 입력 텐서 (강제 정리 → 스케일링 → 강제 정리)
        if isinstance(X, np.ndarray):
            X_np = self._nan_to_num_array(X, nan=0.0)
            X_scaled, _ = self.scaler.transform(X_np)
            X_scaled = self._nan_to_num_array(X_scaled, nan=0.0)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
            X_np = self._nan_to_num_array(X_np, nan=0.0)
            X_scaled, _ = self.scaler.transform(X_np)
            X_scaled = self._nan_to_num_array(X_scaled, nan=0.0)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        # 모델
        model = getattr(self, "model", None) or self
        if not hasattr(model, "parameters"):
            # 문자열 LLM 모델명이 self.model에 있을 수 있으므로 방지
            model = self
        # 디바이스
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        X_tensor = X_tensor.to(device)

        # MC Dropout
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y = model(X_tensor)  # (B, D) or (B, 1)
                y = y.detach().cpu().numpy()
                y = self._nan_to_num_array(y, nan=0.0)
                preds.append(y)
        preds = np.stack(preds, axis=0)  # (K, B, D)

        # 배치=1 가정
        pred_arr = preds[:, 0, :] if preds.ndim == 3 else preds[:, 0:1]  # (K, D)
        if pred_arr.ndim == 2 and pred_arr.shape[1] > 1:
            pred_main = pred_arr[:, return_index]                         # (K,)
        else:
            pred_main = pred_arr.reshape(-1)                              # (K,)

        # 평균/표준편차 (모델 출력 단위, NaN 방지)
        mean_pred = float(np.nanmean(pred_main)) if pred_main.size > 0 else 0.0
        std_pred  = float(np.nanstd(pred_main, ddof=1) if pred_main.size > 1 else 0.0)
        if not np.isfinite(mean_pred): mean_pred = 0.0
        if not np.isfinite(std_pred):  std_pred  = 0.0

        # y 스케일 역변환(필요시) → 역변환 후도 재가드
        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = float(self.scaler.inverse_y([mean_pred])[0])
            std_pred  = float(self.scaler.inverse_y([std_pred])[0])
            if not np.isfinite(mean_pred): mean_pred = 0.0
            if not np.isfinite(std_pred):  std_pred  = 0.0

        # 현재가
        try:
            if current_price is None:
                current_price = float(getattr(self.stockdata, "last_price", 100.0) or 100.0)
        except Exception:
            current_price = 100.0
        if not np.isfinite(current_price) or current_price <= 0:
            current_price = 100.0

        # 출력 타입별 가격/σ 계산
        pred_return_for_record = None
        if output_type == "price":
            predicted_price = mean_pred
            sigma_price = std_pred
        elif output_type == "log_return":
            r = np.expm1(mean_pred)                  # r = e^y - 1
            if not np.isfinite(r): r = 0.0
            predicted_price = current_price * (1.0 + float(r))
            sigma_price = current_price * float(np.expm1(max(std_pred, 0.0)))
            pred_return_for_record = float(r)
        else:
            r = mean_pred
            if not np.isfinite(r): r = 0.0
            predicted_price = current_price * (1.0 + float(r))
            sigma_price = current_price * float(max(std_pred, 0.0))
            pred_return_for_record = float(r)

        # --- 단위 가드: 실수로 r를 price로 오인한 경우 자동 보정 ---
        try:
            if (predicted_price is not None) and (current_price is not None):
                if predicted_price < 1.0 and current_price > 10.0:
                    predicted_price = current_price * (1.0 + float(mean_pred))
                    sigma_price = current_price * float(std_pred)
        except Exception:
            pass

        # === 🔐 반환 직전 "강제-유효화"(핵심) ===
        predicted_price = self._finite(predicted_price, float(current_price))
        sigma_price     = abs(self._finite(sigma_price, 0.0))
        confidence      = 1.0 / (abs(sigma_price) + 1e-8)
        confidence      = self._finite(confidence, 0.0)

        # 상승확률 근사
        calibrated_prob_up = None
        if has_prob_head:
            pass
        else:
            if output_type == "price":
                approx_r_samples = (pred_main - current_price) / max(current_price, 1e-8)
            elif output_type == "log_return":
                approx_r_samples = np.expm1(pred_main)
            else:
                approx_r_samples = pred_main
            approx_r_samples = self._nan_to_num_array(approx_r_samples, nan=0.0)
            calibrated_prob_up = float((approx_r_samples > 0).mean()) if approx_r_samples.size > 0 else None
        if calibrated_prob_up is None or not (0.0 <= calibrated_prob_up <= 1.0):
            calibrated_prob_up = 0.5


        # 항상 pred_return 채우기 (price 출력인 경우 환산)
        if pred_return_for_record is None and current_price:
            pred_return_for_record = float(predicted_price / current_price - 1.0)
        pred_return_for_record = self._finite(pred_return_for_record, 0.0)

        if self.verbose:
            print(f"[predict] mean_pred={mean_pred:.6f}, std_pred={std_pred:.6f}")
            print(f"[predict] price={predicted_price:.6f}, sigma={sigma_price:.6f}, conf={confidence:.6f}, prob_up={calibrated_prob_up}")

        return Target(
            next_close=predicted_price,
            uncertainty=sigma_price,
            confidence=confidence,
            feature_cols=getattr(self.stockdata, "feature_cols", None),
            importances=None,
            pred_return=pred_return_for_record,
            calibrated_prob_up=calibrated_prob_up,
        )

    # -----------------------------
    # 메인 워크플로 (Debate 호환 시그니처)
    # -----------------------------
    def reviewer_draft(self, ticker: str) -> Opinion:
        """캐시 재사용: _last_X/stockdata 있으면 재다운로드 없이 사용"""
        if getattr(self, "_last_X", None) is None or self.stockdata is None or self.stockdata.ticker != ticker:
            X = self.searcher(ticker)
        else:
            X = self._last_X

        target = self.predict(X)

        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)

        schema_reason_only = {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        }

        try:
            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                schema_reason_only,
            )
        except Exception:
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
        """기본: 반박/지지 생성 안 함(침묵)"""
        return []

    def reviewer_revise(
        self,
        my_lastest: Opinion,
        others_latest: Dict[str, Opinion],
        received_rebuttals: List[Rebuttal],
        stock_data: Optional[StockData],
    ) -> Opinion:
        """간단 수정(반박↑, 지지↓)"""
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
            pred_return=getattr(t, "pred_return", None),
            calibrated_prob_up=getattr(t, "calibrated_prob_up", None),
        )

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
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_opinion method")

    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_rebuttal method")

    # -----------------------------
    # 모델 저장/로딩
    # -----------------------------
    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치 로드 (state_dict 호환)"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            return False

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
        """간단 사전학습 루틴(MSE)"""
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
        torch.save(model.state_dict(), model_path)
        print(f"✅ {self.agent_id} model saved.\n✅ pretraining finished.\n")

    # -----------------------------
    # OpenAI API 호출
    # -----------------------------
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """모델 폴백 포함 OpenAI Responses API 호출"""
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
                    # 1) output_text
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}
                    # 2) output 배열
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
                self._p(f"⚠️ 모델 {model} 실패: {e}")
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    # -----------------------------------------
    # 🔹 간단 성능 평가 (참고용)
    # -----------------------------------------
    def evaluate(self, ticker: str = None):
        """검증 데이터로 성능 평가 (참고: price vs return 혼합일 수 있어 방향 정확도 위주 해석 권장)"""
        if ticker is None:
            ticker = self.ticker

        X, y, feature_cols = load_dataset(ticker, agent_id=self.agent_id, save_dir=self.data_dir)

        split_idx = int(len(X) * 0.8)
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        self.scaler.load(ticker)

        predictions = []
        actual_returns = []

        for i in range(len(X_val)):
            X_input = X_val[i:i + 1]
            # NaN 방지
            X_input = self._nan_to_num_array(X_input, nan=0.0)
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            with torch.no_grad():
                pred_target = self.predict(X_tensor)
                predictions.append(pred_target.next_close)
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


# ===============================================================
# DataScaler: 학습/추론용 정규화 유틸리티
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

        # 3D 입력 (samples, seq_len, features) → 2D로 변환 후 피팅
        n_samples, seq_len, n_feats = X_train.shape
        X_2d = X_train.reshape(-1, n_feats)
        self.x_scaler = Sx().fit(X_2d) if Sx else None
        self.y_scaler = Sy().fit(y_train.reshape(-1, 1)) if Sy else None

    def transform(self, X, y=None):
        # 3D 입력 → 2D로 변환 후 스케일링, 원형 복원
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
            out = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            # 역변환 후에도 NaN/Inf 방지
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out
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
