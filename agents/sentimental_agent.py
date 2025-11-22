# agents/sentimental_agent.py

from __future__ import annotations

import os
import json
from typing import Optional, Tuple, Dict, Any, List, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

# BaseAgent
from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal

# 뉴스 병합 / 뉴스 기반 데이터셋
from core.sentimental_classes.news import merge_price_with_news_features
from core.sentimental_classes.pretrain_dataset_builder import build_pretrain_dataset

# LSTM 모델
from core.sentimental_classes.lstm_model import SentimentalLSTM

# dataset loader
from core.data_set import load_dataset, build_dataset

# 프롬프트
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS

from config.agents import agents_info, dir_info

# config
from config.agents import agents_info, dir_info

CFG_S = agents_info["SentimentalAgent"]

FEATURE_COLS = [
    "return_1d",
    "hl_range",
    "Volume",
    "news_count_1d",
    "news_count_7d",
    "sentiment_mean_1d",
    "sentiment_mean_7d",
    "sentiment_vol_7d",
]

# config 값과 동기화
WINDOW_SIZE = CFG_S["window_size"]
HIDDEN_DIM = CFG_S.get("d_model", 64)   # d_model을 LSTM hidden_dim으로 재활용
NUM_LAYERS = CFG_S["num_layers"]
DROPOUT = CFG_S["dropout"]
# =============================================================================


class SentimentalAgent(BaseAgent):

    def __init__(self, ticker, agent_id="SentimentalAgent", **kwargs):
        super().__init__(ticker=ticker, agent_id=agent_id, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = agents_info[self.agent_id]

        # BaseAgent에서도 window_size를 세팅하지만, 여기서도 명시적으로 맞춰둠
        self.window_size = cfg["window_size"]

        # LSTM 구조 관련 하이퍼파라미터
        self.hidden_dim = HIDDEN_DIM          # = cfg.get("d_model", 64)
        self.num_layers = NUM_LAYERS          # = cfg["num_layers"]
        self.dropout = DROPOUT                # = cfg["dropout"]

        # 피처 목록 (실제는 FEATURE_COLS 기준)
        self.feature_cols = list(FEATURE_COLS)

        self.model = None
        self.model_loaded = False

        if not getattr(self, "ticker", None):
            self.ticker = ticker
        if not self.ticker:
            raise ValueError("SentimentalAgent: ticker is None/empty")
        self.ticker = str(self.ticker).upper()
        setattr(self, "symbol", self.ticker)


    # -------------------------------------------------------
    # PRETRAIN
    # -------------------------------------------------------
    def pretrain(self):
        print(f"[SentimentalAgent] Building pretrain dataset with news for {self.ticker}...")
        build_pretrain_dataset(self.ticker)

        print(f"[SentimentalAgent] Pretraining LSTM for {self.ticker}...")
        super().pretrain()

    # -------------------------------------------------------
    # _BUILD_MODEL
    # -------------------------------------------------------
    def _build_model(self) -> nn.Module:
        """BaseAgent.pretrain에서 사용할 LSTM 모델 생성"""

        # dataset 로드 또는 생성
        try:
            X, y, cols = load_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )
        except Exception:
            build_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )
            X, y, cols = load_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )

        # feature_cols 자동 업데이트
        self.feature_cols = list(cols)
        input_dim = X.shape[-1]

        model = SentimentalLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        return model

    # -------------------------------------------------------
    # RUN_DATASET
    # -------------------------------------------------------
    def run_dataset(self, days: int = 365) -> StockData:
        """
        최근 days일치 가격 + 뉴스 피처를 기반으로
        FEATURE_COLS 입력(1, T, F)을 만들고 StockData를 생성
        """
        # 0) 날짜 범위
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=days)

        # 1) 가격 데이터 (yfinance)
        df_price = yf.download(self.ticker, start=start, end=end)
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0].lower() for c in df_price.columns]
        else:
            df_price.columns = [c.lower() for c in df_price.columns]

        df_price = df_price.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

        df_price["date"] = df_price.index
        df_price = df_price.reset_index(drop=True)

        # 2) 뉴스 + 가격 병합
        df_merged = merge_price_with_news_features( 
            df_price=df_price,
            ticker=self.ticker,
            asof_kst=end.date(),
            base_dir=os.path.join("data", "raw", "news"),
        )
        if isinstance(df_merged, tuple):
            df_feat = df_merged[0]
        else:
            df_feat = df_merged

        df_feat = df_feat.sort_values("date").reset_index(drop=True)

        # ---------------------------------------
        # FEATURE_COLS 자동 보정
        # ---------------------------------------
        required = list(FEATURE_COLS)
        print("[SentimentalAgent.run_dataset] missing(before):",
              [c for c in required if c not in df_feat.columns])

        # return_1d
        if "return_1d" not in df_feat.columns:
            df_feat["return_1d"] = df_feat["close"].pct_change().fillna(0)

        # hl_range
        if "hl_range" not in df_feat.columns:
            df_feat["hl_range"] = ((df_feat["high"] - df_feat["low"]) /
                                   df_feat["close"].replace(0, np.nan)).fillna(0)

        # Volume (대문자)
        if "Volume" not in df_feat.columns:
            df_feat["Volume"] = df_feat["volume"].fillna(0)

        # 뉴스 1일 feature (없으면 0)
        for col in ["news_count_1d", "sentiment_mean_1d"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        # 마지막 검증
        missing_after = [c for c in required if c not in df_feat.columns]
        if missing_after:
            raise ValueError(
                f"[SentimentalAgent.run_dataset] FEATURE_COLS 부족: {missing_after}"
            )

        print("[SentimentalAgent.run_dataset] all FEATURE_COLS present.")

        # ---------------------------------------
        # 입력 행렬 생성
        # ---------------------------------------
        feat_values = df_feat[required].values.astype("float32")

        if len(feat_values) < self.window_size:
            raise ValueError(
                f"데이터 길이({len(feat_values)}) < 윈도우({self.window_size})"
            )

        X_last = feat_values[-self.window_size:]
        X_last = X_last[None, :, :]  # (1, T, F)
        self._last_input = X_last

        # ---------------------------------------
        # StockData 생성
        # ---------------------------------------
        last_row = df_feat.iloc[-1]
        last_price = float(last_row["close"])

        sd = StockData()
        sd.ticker = self.ticker
        sd.last_price = last_price
        sd.currency = "USD"
        sd.feature_cols = required
        sd.window_size = self.window_size
        sd.raw_df = df_feat

        sd.news_feats = {
            "news_count_7d": float(last_row.get("news_count_7d", 0)),
            "sentiment_mean_7d": float(last_row.get("sentiment_mean_7d", 0)),
            "sentiment_vol_7d": float(last_row.get("sentiment_vol_7d", 0)),
        }

        sd.snapshot = {
            "agent_id": self.agent_id,
            "feature_cols": sd.feature_cols,
            "window_size": sd.window_size,
            "news_feats": sd.news_feats,
            "raw_df": sd.raw_df,
        }

        sd.X_seq = X_last
        sd.SentimentalAgent = {
            "X_seq": X_last,
            "last_price": last_price,
        }

        self.stockdata = sd
        return sd

    # -------------------------------------------------------
    # searcher (DebateAgent에서 호출) :
    #   - 여기서는 run_dataset() 기반으로 최신 가격/뉴스 사용
    # -------------------------------------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        if ticker and ticker != self.ticker:
            self.ticker = str(ticker).upper()

        sd = self.run_dataset(days=365)
        self.stockdata = sd

        X_last = sd.X_seq  # (1, T, F)
        X_tensor = torch.tensor(X_last, dtype=torch.float32)
        return X_tensor

    # -------------------------------------------------------
    # predict
    #   - Monte Carlo Dropout + DataScaler + y*100 스케일 고려
    # -------------------------------------------------------
    def predict(self, X, n_samples: int = 30, current_price: float | None = None):
        """
        SentimentalAgent 전용 Monte Carlo Dropout 예측 함수

        - 입력: StockData 또는 (T, F) / (1, T, F) numpy/tensor
        - self.model(SentimentalLSTM) + self.scaler 로드
        - MC Dropout으로 예측 분포 샘플링
        - "수익률 * 100" → 실제 가격(next_close)로 변환
        - Target(next_close, uncertainty, confidence) 반환
        """
        # -----------------------------
        # 0) 입력 정리 (StockData 래핑)
        # -----------------------------
        if isinstance(X, StockData):
            sd = X
            X_in = getattr(sd, "X_seq", None)
            if X_in is None:
                raise ValueError("StockData에 X_seq가 없습니다. run_dataset()을 먼저 호출하세요.")
            if current_price is None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
        else:
            sd = None
            X_in = X

        if X_in is None:
            raise ValueError("predict()에 전달된 입력 X가 None 입니다.")

        # numpy / tensor 로 통일
        if isinstance(X_in, np.ndarray):
            X_raw_np = X_in.copy()
        elif isinstance(X_in, torch.Tensor):
            X_raw_np = X_in.detach().cpu().numpy().copy()
        else:
            raise TypeError(f"Unsupported input type for predict: {type(X_in)}")

        # run_dataset() 기준: X_seq.shape == (1, T, F)
        if X_raw_np.ndim == 3 and X_raw_np.shape[0] == 1:
            X_seq_np = X_raw_np[0]        # (T, F)
        elif X_raw_np.ndim == 2:
            X_seq_np = X_raw_np           # (T, F)
        else:
            raise ValueError(f"예상하지 못한 입력 shape: {X_raw_np.shape}, (T,F) 또는 (1,T,F)만 지원합니다.")

        # -----------------------------
        # 1) 모델 준비 (load_model()에 의존하지 않음)
        # -----------------------------
        model = getattr(self, "model", None)
        if model is None:
            # __init__에서 hidden_dim, num_layers, dropout 세팅해둔 상태라고 가정
            self.model = SentimentalLSTM(
                input_dim=len(FEATURE_COLS),
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            model = self.model

        # state_dict 직접 로드
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        if os.path.exists(model_path) and not getattr(self, "model_loaded", False):
            try:
                ckpt = torch.load(model_path, map_location="cpu")
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                self.model_loaded = True
                print(f" SentimentalAgent 모델(state_dict) 로드 완료 ({model_path})")
            except Exception as e:
                print(f"[SentimentalAgent] 모델 state_dict 로드 실패(무시하고 진행): {e}")

        # -----------------------------
        # 1-1) 스케일러 로드
        # -----------------------------
        if not hasattr(self, "scaler"):
            raise RuntimeError("[SentimentalAgent] self.scaler가 정의되지 않았습니다.")
        self.scaler.load(self.ticker)

        # -----------------------------
        # 2) 입력 스케일링
        # -----------------------------
        X_scaled, _ = self.scaler.transform(X_seq_np)     # (T, F)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # [T, F] → [1, T, F]
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(0)

        device = getattr(self, "device", torch.device("cpu"))
        X_tensor = X_tensor.to(device)
        model.to(device)

        # -----------------------------
        # 3) Monte Carlo Dropout 추론
        # -----------------------------
        model.train()  # dropout 활성화 (MC Dropout)
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor)   # 예: (1, seq_len) 또는 (1, 1)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                preds.append(y_pred.detach().cpu().numpy().flatten())

        preds = np.stack(preds)           # (samples, L)
        mean_pred = preds.mean(axis=0)    # (L,)
        std_pred = np.abs(preds.std(axis=0))

        # -----------------------------
        # 4) σ 기반 confidence 계산
        # -----------------------------
        sigma = float(std_pred[-1])
        sigma = max(sigma, 1e-6)
        confidence = float(1.0 / (1.0 + np.log1p(sigma)))

        # -----------------------------
        # 5) y_scaler 역변환 + 수익률 → 가격 변환
        # -----------------------------
        # 모델 출력이 "수익률 * 100" 형태라고 가정 (예: 3.5 → +3.5%)
        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        predicted_return = float(mean_pred[-1]) / 100.0  # 3.5 → 0.035

        # current_price 추론
        if current_price is None:
            if sd is not None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
            else:
                current_price = float(getattr(self, "last_price", 100.0))

        predicted_price = float(current_price * (1.0 + predicted_return))

        # -----------------------------
        # 6) Target 생성 및 저장
        # -----------------------------
        target = Target(
            next_close=predicted_price,
            uncertainty=sigma,
            confidence=confidence,
        )

        if hasattr(self, "targets"):
            self.targets.append(target)

        return target

    # -------------------------------------------------------
    # 내부 helper: _predict_next_close
    #   - run_dataset → self.predict(StockData) 조합으로 사용
    # -------------------------------------------------------
    @torch.inference_mode()
    def _predict_next_close(self) -> Tuple[float, float, float, List[str]]:
        """
        run_dataset() 결과 또는 이미 계산된 self.stockdata를 이용해
        다음날 종가 / 불확실성 / 신뢰도를 얻는다.
        """
        sd = getattr(self, "stockdata", None)
        if sd is None or getattr(sd, "X_seq", None) is None:
            sd = self.run_dataset(days=365)

        target = self.predict(sd, n_samples=30)
        cols = list(getattr(sd, "feature_cols", self.feature_cols))
        return float(target.next_close), float(target.uncertainty or 0.0), float(target.confidence or 0.0), cols

    # -------------------------------------------------------
    # ctx 구성 (run_dataset의 news_feats 사용)
    # -------------------------------------------------------
    def build_ctx(self, asof_date_kst: Optional[str] = None) -> Dict[str, Any]:
        # 0) StockData 확보
        stockdata: StockData | None = getattr(self, "stockdata", None)
        if stockdata is None or getattr(stockdata, "X_seq", None) is None:
            stockdata = self.run_dataset()

        # 1) 기준 날짜(asof_date_kst)
        if asof_date_kst is None:
            asof_date_kst = datetime.now().strftime("%Y-%m-%d")

        # 2) 예측 값
        pred_close, uncertainty_std, confidence, cols = self._predict_next_close()

        # 3) 가격 스냅샷 (raw_df 마지막 행 기준)
        price_snapshot: Dict[str, Optional[float]] = {}
        df = getattr(stockdata, "raw_df", None)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            last = df.iloc[-1]
            price_snapshot["Close"] = float(last.get("close", np.nan))
            price_snapshot["Open"] = float(last.get("open", np.nan))
            price_snapshot["High"] = float(last.get("high", np.nan))
            price_snapshot["Low"] = float(last.get("low", np.nan))
            price_snapshot["Volume"] = float(last.get("volume", np.nan))
        else:
            price_snapshot = {
                "Close": getattr(stockdata, "last_price", np.nan),
                "Open": None,
                "High": None,
                "Low": None,
                "Volume": None,
            }

        # 4) 뉴스/감성 피처: run_dataset()에서 저장한 news_feats 사용
        nf = getattr(stockdata, "news_feats", {}) or {}
        news_count_7d = float(nf.get("news_count_7d", 0.0))
        sentiment_mean_7d = float(nf.get("sentiment_mean_7d", 0.0))
        sentiment_vol_7d = float(nf.get("sentiment_vol_7d", 0.0))

        sentiment_summary = {
            "mean_7d": sentiment_mean_7d,
            "mean_30d": 0.0,          # 아직은 30일 피처 없으니 0으로
            "pos_ratio_7d": 0.0,
            "neg_ratio_7d": 0.0,
        }
        sentiment_vol = {"vol_7d": sentiment_vol_7d}
        news_count = {"count_7d": int(news_count_7d)}
        trend_7d = 0.0
        has_news = bool(news_count_7d > 0)

        # 5) snapshot / prediction 구성
        last_price = price_snapshot.get("Close", np.nan)
        if last_price and last_price == last_price:
            pred_return = float(pred_close / last_price - 1.0)
        else:
            pred_return = None

        snapshot = {
            "asof_date": asof_date_kst,
            "last_price": last_price,
            "currency": getattr(stockdata, "currency", "USD"),
            "window_size": self.window_size,
            "feature_cols_preview": [c for c in (cols or [])[:8]],
        }

        feature_importance = {
            "sentiment_score": sentiment_summary.get("mean_7d", 0.0),
            "sentiment_summary": sentiment_summary,
            "sentiment_volatility": sentiment_vol,
            "trend_7d": trend_7d,
            "news_count": news_count,
            "has_news": has_news,
            "price_snapshot": {
                "Close": price_snapshot.get("Close"),
                "Open": price_snapshot.get("Open"),
                "High": price_snapshot.get("High"),
                "Low": price_snapshot.get("Low"),
                "Volume": price_snapshot.get("Volume"),
                "ret_1d": None,
                "ret_5d": None,
                "ret_20d": None,
                "zscore_20d": None,
                "vol_change_5d": None,
            },
        }

        ctx = {
            "agent_id": self.agent_id,
            "ticker": self.ticker,
            "snapshot": snapshot,
            "prediction": {
                "pred_close": pred_close,
                "pred_return": pred_return,
                "uncertainty": {
                    "std": uncertainty_std,
                    "ci95": float(1.96 * uncertainty_std),
                },
                "confidence": confidence,
                "pred_next_close": pred_close,
            },
            "feature_importance": feature_importance,
        }
        return ctx

    def reviewer_rebuttal(
        self,
        my_opinion: Opinion,
        other_opinion: Opinion,
        round_index: int,
    ) -> Rebuttal:

        return self.reviewer_rebut(
            my_opinion=my_opinion,
            other_opinion=other_opinion,
            round_index=round_index,
        )

    # -------------------------------------------------------
    # Opinion / Rebuttal / Revision 프롬프트
    # -------------------------------------------------------
    def _build_messages_opinion(
        self,
        stock_data: StockData,
        target: Target,
    ) -> Tuple[str, str]:
        if stock_data is None:
            stock_data = self.stockdata

        # 공통 ctx 사용
        ctx = self.build_ctx()
        # DebateAgent에서 target이 업데이트됐을 수 있으므로 반영
        ctx["prediction"]["pred_next_close"] = float(getattr(target, "next_close", 0.0))
        ctx["prediction"]["pred_close"] = ctx["prediction"]["pred_next_close"]

        last_close = ctx["snapshot"].get("last_price")
        if isinstance(last_close, (int, float)) and last_close not in (0, None):
            try:
                chg = ctx["prediction"]["pred_next_close"] / float(last_close) - 1.0
            except ZeroDivisionError:
                chg = None
        else:
            chg = None
        ctx["prediction"]["pred_return"] = chg

        ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
        prompts = OPINION_PROMPTS["SentimentalAgent"]
        system_text = prompts["system"]
        user_tmpl = prompts["user"]

        try:
            user_text = user_tmpl.format(context=ctx_json)
        except KeyError:
            user_text = user_tmpl.replace("{context}", ctx_json)

        return system_text, user_text

    def _build_messages_rebuttal(
        self,
        my_opinion: Opinion,
        target_opinion: Opinion,
        stock_data: StockData,
    ) -> Tuple[str, str]:
        opp_agent = getattr(target_opinion, "agent_id", "UnknownAgent")
        opp_reason = getattr(target_opinion, "reason", "")

        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        pred_close = float(my_opinion.target.next_close)
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        system_tmpl = None
        user_tmpl = None
        if REBUTTAL_PROMPTS and "SentimentalAgent" in REBUTTAL_PROMPTS:
            pp = REBUTTAL_PROMPTS["SentimentalAgent"]
            system_tmpl = pp.get("system")
            user_tmpl = pp.get("user")

        if not system_tmpl:
            system_tmpl = (
                "당신은 감성 기반 단기 주가 분석가로서 상대 의견의 논리적/수치적 허점을 분석해 반박합니다. "
                "감성지표(평균, 추세, 변동성)와 뉴스 개수, 예측의 불확실성을 근거로 삼되, "
                "합리적 포인트는 인정하고 핵심 쟁점 위주로 간결히 반박하세요."
            )

        if not user_tmpl:
            user_tmpl = (
                "티커: {ticker}\n"
                "상대 에이전트: {opp_agent}\n"
                "상대 의견:\n{opp_reason}\n\n"
                "우리 예측:\n- next_close: {pred_close}\n- 예상 변화율(현재가 대비): {chg}\n"
                "감성 근거:\n- mean7={mean7}, mean30={mean30}, pos7={pos7}, neg7={neg7}\n"
                "- vol7={vol7}, trend7={trend7}, news7={news7}\n\n"
                "요청: 위 정보를 바탕으로 상대 의견의 약점 2~4개를 조목조목 반박하세요. "
                "특히 감성 추세/변동성, 뉴스 수의 맥락, 예측 불확실성(높/낮음)이 "
                "상대 주장과 어떻게 상충/보완되는지 구체적으로 지적하세요."
            )

        user_text = user_tmpl.format(
            ticker=self.ticker,
            opp_agent=opp_agent,
            opp_reason=opp_reason if opp_reason else "(상대 의견 내용 없음)",
            pred_close=f"{pred_close:.4f}",
            chg=("NA" if change_ratio is None else f"{change_ratio*100:.2f}%"),
            mean7=f"{sent.get("mean_7d", 0.0):.4f}",
            mean30=f"{sent.get("mean_30d", 0.0):.4f}",
            pos7=f"{sent.get("pos_ratio_7d", 0.0):.4f}",
            neg7=f"{sent.get("neg_ratio_7d", 0.0):.4f}",
            vol7=("NA" if vol7 is None else f"{vol7:.4f}"),
            trend7=("NA" if trend7 is None else f"{trend7:.4f}"),
            news7=("NA" if news7 is None else f"{news7}"),
        )
        return system_tmpl, user_text

    def _build_messages_revision(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: Optional[List[Rebuttal]] = None,
        stock_data: StockData = None,
    ) -> Tuple[str, str]:
        if stock_data is None:
            stock_data = self.stockdata

        def _op_text(x: Union[Opinion, Dict[str, Any], str, None, Any]) -> str:
            if isinstance(x, Opinion):
                return getattr(x, "reason", "")
            if isinstance(x, dict):
                return x.get("reason", "") or x.get("message", "")
            if hasattr(x, "message"):
                return getattr(x, "message", "")
            if hasattr(x, "reason"):
                return getattr(x, "reason", "")
            return str(x) if x else ""

        prev_reason = _op_text(my_opinion)

        reb_texts: List[str] = []
        if isinstance(rebuttals, list):
            for r in rebuttals:
                reb_texts.append(_op_text(r))
        elif rebuttals is not None:
            reb_texts.append(_op_text(rebuttals))

        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        pred_info = ctx.get("prediction", {}) or {}
        unc_dict = pred_info.get("uncertainty", {}) or {}
        unc_std = unc_dict.get("std", None)
        confidence = pred_info.get("confidence", None)

        pred_close = float(my_opinion.target.next_close)
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        context_parts: List[str] = []
        if last_price is not None:
            if change_ratio is not None:
                context_parts.append(
                    f"현재 주가는 {last_price:.2f}이고, 모델은 다음 거래일 종가를 {pred_close:.2f}로 예측했습니다 "
                    f"(변화율 약 {change_ratio*100:.2f}%)."
                )
            else:
                context_parts.append(
                    f"현재 주가는 {last_price:.2f}이며, 다음 거래일 종가 예측값은 {pred_close:.2f}입니다."
                )
        else:
            context_parts.append(
                f"다음 거래일 종가 예측값은 {pred_close:.2f}입니다."
            )

        mean7=f"{sent.get('mean_7d', 0.0):.4f}",
        mean30=f"{sent.get('mean_30d', 0.0):.4f}",
        pos7=f"{sent.get('pos_ratio_7d', 0.0):.4f}",
        neg7=f"{sent.get('neg_ratio_7d', 0.0):.4f}",

        if mean7 is not None and mean30 is not None:
            context_parts.append(
                f"최근 7일 평균 감성 점수는 {mean7:.3f}, 최근 30일 평균은 {mean30:.3f}입니다."
            )
        if pos7 is not None and neg7 is not None:
            context_parts.append(
                f"최근 7일 기준 긍정 기사 비율은 {pos7:.2%}, 부정 기사 비율은 {neg7:.2%}입니다."
            )
        if vol7 is not None:
            context_parts.append(
                f"최근 7일 감성 점수의 변동성(표준편차)은 {vol7:.3f}입니다."
            )
        if trend7 is not None:
            context_parts.append(
                f"최근 7일 감성 추세(회귀 기울기)는 {trend7:.4f}입니다."
            )
        if news7 is not None:
            context_parts.append(
                f"최근 7일 동안 수집된 뉴스 개수는 {news7}건입니다."
            )

        if unc_std is not None and confidence is not None:
            context_parts.append(
                f"예측 표준편차는 {unc_std:.4f}, 신뢰도는 {confidence:.3f}입니다."
            )

        context_str = " ".join(context_parts) if context_parts else (
            "최근 뉴스 감성 점수, 변동성, 긍·부정 비율, 뉴스 수, 예측 불확실성 등을 종합해 단기 주가를 해석합니다."
        )

        system_tmpl = None
        user_tmpl = None
        if REVISION_PROMPTS and "SentimentalAgent" in REVISION_PROMPTS:
            pp = REVISION_PROMPTS["SentimentalAgent"]
            system_tmpl = pp.get("system")
            user_tmpl = pp.get("user")

        if not system_tmpl:
            system_tmpl = (
                "당신은 감성 기반 단기 주가 분석가입니다. "
                "초안 의견과 반박들을 검토해 핵심만 남기고, 데이터에 근거해 결론을 더 명확히 다듬습니다. "
                "불확실성/신뢰도 해석을 포함하여 한 단계 더 견고한 최종 의견으로 수정하세요."
            )

        if not user_tmpl:
            user_tmpl = (
                "티커: {ticker}\n"
                "초안 의견:\n{prev}\n\n"
                "수신한 반박 요약:\n{rebuts}\n\n"
                "업데이트된 수치:\n- next_close: {pred_close}\n- 예상 변화율: {chg}\n"
                "감성 근거 스냅샷:\n- mean7={mean7}, mean30={mean30}, pos7={pos7}, neg7={neg7}\n"
                "- vol7={vol7}, trend7={trend7}, news7={news7}\n\n"
                "추가 컨텍스트:\n{context}\n\n"
                "요청: 초안의 과장/중복/약한 근거를 정리하고, 강한 근거(감성 추세, 변동성, 뉴스 수 변화)를 중심으로 "
                "최종 의견을 3~5문장으로 재작성하세요. 불확실성/신뢰도 해석을 포함하세요."
            )

        rebuts_joined = "- " + "\n- ".join(
            [s for s in reb_texts if s]
        ) if reb_texts else "(반박 없음)"

        user_text = user_tmpl.format(
            ticker=self.ticker,
            prev=prev_reason if prev_reason else "(초안 없음)",
            rebuts=rebuts_joined,
            pred_close=f"{pred_close:.4f}",
            chg=("NA" if change_ratio is None else f"{change_ratio*100:.2f}%"),
            mean7=("NA" if mean7 is None else f"{mean7:.4f}"),
            mean30=("NA" if mean30 is None else f"{mean30:.4f}"),
            pos7=("NA" if pos7 is None else f"{pos7:.4f}"),
            neg7=("NA" if neg7 is None else f"{neg7:.4f}"),
            vol7=("NA" if vol7 is None else f"{vol7:.4f}"),
            trend7=("NA" if trend7 is None else f"{trend7:.4f}"),
            news7=("NA" if news7 is None else f"{news7}"),
            context=context_str,
        )
        return system_tmpl, user_text

    # -------------------------------------------------------
    # 레거시 get_opinion (단독 테스트용)
    # -------------------------------------------------------
    def get_opinion(self, idx: int = 0, ticker: Optional[str] = None) -> Opinion:
        if ticker and ticker != self.ticker:
            self.ticker = str(ticker).upper()

        pred_close, uncertainty_std, confidence, _ = self._predict_next_close()
        target = Target(
            next_close=float(pred_close),
            uncertainty=float(uncertainty_std),
            confidence=float(confidence),
        )

        try:
            if hasattr(self, "reviewer_draft"):
                op = self.reviewer_draft(getattr(self, "stockdata", None), target)
                return op
        except Exception as e:
            print("[SentimentalAgent] reviewer_draft 사용 실패:", e)

        ctx = self.build_ctx()
        fi = ctx["feature_importance"]
        sent = fi["sentiment_summary"]

        reason = (
            f"{self.ticker}의 최근 7일 감성 평균은 {sent['mean_7d']:.3f}이며 "
            f"뉴스 개수(7d)는 {fi['news_count']['count_7d']}건입니다. "
            f"감성 변동성(vol_7d)={fi['sentiment_volatility']['vol_7d']:.3f}, "
            f"감성 추세(trend_7d)={fi['trend_7d']:.3f}입니다."
        )

        return Opinion(
            agent_id=self.agent_id,
            target=target,
            reason=reason,
        )
