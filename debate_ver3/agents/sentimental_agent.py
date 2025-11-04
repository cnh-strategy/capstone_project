# agents/sentimental_agent.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

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
            "target_mode" : "return"
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
    """BaseAgent(풀스택)와 호환되는 감성 에이전트 (V3 스타일 CTX 포함)"""

    def __init__(self, **kwargs):
        super().__init__(agent_id="SentimentalAgent", **kwargs)
        cfg = agents_info.get(self.agent_id, {})
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        # demo에서 참고하는 필드가 반드시 존재하도록 초기화
        self.feature_cols: List[str] = []

    # ---------------------------
    # BaseAgent 훅
    # ---------------------------
    def _build_model(self) -> nn.Module:
        return _LazyLSTMWithStub(hidden_dim=self.hidden_dim, dropout=self.dropout)

    # ---------------------------
    # 느슨 로드 (예전 체크포인트 호환)
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
    # 데이터 검색/로딩 (최신 윈도우 반환)
    # ---------------------------
    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        import os
        import torch
        import yfinance as yf
        from core.data_set import build_dataset, load_dataset
        from agents.base_agent import StockData as _StockData

        # 1) 기본 파라미터 정리
        ticker = ticker or self.ticker
        self.ticker = ticker
        agent_id = self.agent_id

        # 2) 파일 경로
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")

        # 3) 데이터 없거나 재빌드면 생성
        if rebuild or not os.path.exists(dataset_path):
            print(f"⚙️ {ticker} {agent_id} dataset {'rebuild requested' if rebuild else 'not found'}. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir, agent_id=agent_id)

        # 4) 로드 (X: (N,T,F))
        X, y, feature_cols = load_dataset(ticker, agent_id=agent_id, save_dir=self.data_dir)
        self.feature_cols = feature_cols[:]  # 보존

        # 5) 최신 윈도우만 (1,T,F)로 반환
        X_latest = X[-1:]                     # (1,T,F)
        X_tensor = torch.tensor(X_latest, dtype=torch.float32)

        # 6) StockData 채우기 (가격/통화)
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
    # ✅ CTX 생성: snapshot + prediction + (price/volume/news/regime/explainability)
    # ===========================================================
    def preview_opinion_ctx(self, ticker: Optional[str] = None, mc_passes: int = 30) -> Dict[str, Any]:
        """
        - 최신 윈도우로 예측 요약을 만들고 ctx(dict)를 반환
        - 포함: agent_id, ticker, snapshot, prediction, price_features, volume_features,
               news_features, regime_features, explainability, explain_helpers
        """
        # 데이터 윈도우 확보 (가격/통화 포함 업데이트)
        X_tensor = self.searcher(ticker or self.ticker)

        # 모델 준비
        if getattr(self, "model", None) is None:
            self.model = self._build_model()
        self.model.eval()
        # 느슨 로드 시도 (파일 없어도 계속 진행)
        self.load_model()

        # ---------- 블록 생성 ----------
        snap = self._ctx_snapshot()
        pred = self._ctx_prediction(X_tensor, mc_passes=mc_passes)

        # 가격/거래량 시계열 다운로드
        price_df = self._download_price_df(self.ticker, lookback_days=120)
        price_feats = self._ctx_price_features(price_df)
        volume_feats = self._ctx_volume_features(price_df)

        # 뉴스 집계 로드 (가능하면), 실패 시 중립값
        news_df = self._try_load_news_daily(self.ticker)
        news_feats = self._ctx_news_features(news_df)

        # 레짐 특징 (VIX/섹터모멘텀 등) 계산
        regime_feats = self._ctx_regime_features()

        ctx: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "ticker": self.ticker,
            "snapshot": snap,
            "prediction": pred,
            "price_features": price_feats,
            "volume_features": volume_feats,
            "news_features": news_feats,
            "regime_features": regime_feats,
            "explainability": None,
            "explain_helpers": None,
        }
        return ctx

    # ---------------------------
    # snapshot
    # ---------------------------
    def _ctx_snapshot(self) -> Dict[str, Any]:
        """
        ctx.snapshot: asof_date, last_price, currency, window_size, feature_cols_preview
        """
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

    # ---------------------------
    # prediction (MC Dropout 기반)
    # ---------------------------
    def _mc_predict_return(self, X: torch.Tensor, n: int = 30) -> Tuple[float, float]:
        """
        MC Dropout 기반 평균/표준편차 (다음날 '수익률' 스케일)
        - 모델의 출력이 '다음날 수익률'이라고 가정
        """
        assert X.ndim == 3, "X must be (B,T,F)"
        if getattr(self, "model", None) is None:
            self.model = self._build_model()

        # 드롭아웃 활성화를 위해 train()으로 전환
        self.model.train()
        preds: List[float] = []
        with torch.no_grad():
            for _ in range(int(max(1, n))):
                y = self.model(X).squeeze(-1)    # (B,)
                preds.append(float(y.cpu().numpy().reshape(-1)[0]))
        # 다시 평가모드
        self.model.eval()

        mu = float(np.mean(preds)) if preds else 0.0
        std = float(np.std(preds)) if preds else 0.0
        return mu, std

    def _ctx_prediction(self, X_tensor: torch.Tensor, mc_passes: int = 30) -> Dict[str, Any]:
        """
        ctx.prediction: pred_close, pred_return, uncertainty(std, ci95), confidence
        """
        last = getattr(self.stockdata, "last_price", None)
        mu_ret, std_ret = self._mc_predict_return(X_tensor, n=mc_passes)

        pred_close = None
        if last is not None:
            try:
                pred_close = float(last * (1.0 + float(mu_ret)))
            except Exception:
                pred_close = None

        # 단순 신뢰도 휴리스틱 (표준편차가 낮을수록 ↑)
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
            # V3 호환 필드(없으면 null로 두어도 무방)
            "calibrated_prob_up": 0.0,
            "mc_mean_next_close": None,
            "mc_std": None,
        }
        return pred

    # ---------------------------
    # 가격/거래량 데이터 다운로드
    # ---------------------------
    def _download_price_df(self, ticker: str, lookback_days: int = 120) -> pd.DataFrame:
        start = (datetime.now() - timedelta(days=lookback_days*1.4)).strftime("%Y-%m-%d")
        try:
            df = yf.download(ticker, start=start, interval="1d", progress=False)
            df = df.rename(columns=str.title)
            df = df.dropna(subset=["Close"])
            return df
        except Exception as e:
            print(f"yfinance 가격 데이터 오류: {e}")
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    # ---------------------------
    # price_features
    # ---------------------------
    def _ctx_price_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {
                "lag_ret_1": None, "lag_ret_5": None, "lag_ret_20": None,
                "rolling_vol_20": None, "trend_7d": None,
                "atr_14": None, "breakout_20": None,
                "zscore_close_20": None, "drawdown_20": None,
            }

        c = df["Close"].astype(float)
        h = df.get("High", c).astype(float)
        l = df.get("Low", c).astype(float)

        ret = c.pct_change()
        lag_ret_1 = safe_last(ret, 1)
        lag_ret_5 = (c / c.shift(5) - 1.0).iloc[-1] if len(c) >= 6 else None
        lag_ret_20 = (c / c.shift(20) - 1.0).iloc[-1] if len(c) >= 21 else None

        rolling_vol_20 = ret.rolling(20).std().iloc[-1] if len(ret) >= 20 else None

        # 7일 추세(단순 회귀 대신 최근 7일 수익률)
        trend_7d = (c.iloc[-1] / c.shift(7).iloc[-1] - 1.0) if len(c) >= 8 else None

        # ATR(14)
        tr = pd.concat([
            (h - l),
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else None

        # 20일 최고/최저 돌파
        breakout_20 = None
        if len(c) >= 20:
            hh = c.rolling(20).max().iloc[-1]
            ll = c.rolling(20).min().iloc[-1]
            breakout_20 = bool((c.iloc[-1] >= hh) or (c.iloc[-1] <= ll))

        # Z-score(20)
        if len(c) >= 20:
            ma = c.rolling(20).mean().iloc[-1]
            sd = c.rolling(20).std().iloc[-1]
            zscore_close_20 = float((c.iloc[-1] - ma) / sd) if sd and sd > 0 else None
        else:
            zscore_close_20 = None

        # 20일 최대 낙폭
        if len(c) >= 20:
            window = c.iloc[-20:]
            peak = window.cummax()
            dd = (window / peak - 1.0).min()
            drawdown_20 = float(dd)
        else:
            drawdown_20 = None

        return {
            "lag_ret_1": rfloat(lag_ret_1),
            "lag_ret_5": rfloat(lag_ret_5),
            "lag_ret_20": rfloat(lag_ret_20),
            "rolling_vol_20": rfloat(rolling_vol_20),
            "trend_7d": rfloat(trend_7d),
            "atr_14": rfloat(atr_14),
            "breakout_20": bool(breakout_20) if breakout_20 is not None else None,
            "zscore_close_20": rfloat(zscore_close_20),
            "drawdown_20": rfloat(drawdown_20),
        }

    # ---------------------------
    # volume_features
    # ---------------------------
    def _ctx_volume_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty or "Volume" not in df.columns:
            return {"vol_zscore_20": None, "turnover_rate": None, "volume_spike": None}

        v = df["Volume"].astype(float)
        if len(v) >= 20:
            m = v.rolling(20).mean().iloc[-1]
            s = v.rolling(20).std().iloc[-1]
            vol_z = (v.iloc[-1] - m) / s if s and s > 0 else None
            volume_spike = bool(v.iloc[-1] > (m + 2.0 * s)) if (m is not None and s is not None and s > 0) else None
        else:
            vol_z, volume_spike = None, None

        # turnover_rate: 거래대금/시총을 쓰는 게 정석이지만, 시총 없으면 볼륨 정규화 근사
        # 여기서는 간단히 당일 거래량 / 20일 평균 거래량
        turnover_rate = None
        if len(v) >= 20:
            mv = v.rolling(20).mean().iloc[-1]
            turnover_rate = float(v.iloc[-1] / mv) if mv and mv > 0 else None

        return {
            "vol_zscore_20": rfloat(vol_z),
            "turnover_rate": rfloat(turnover_rate),
            "volume_spike": volume_spike,
        }

    # ---------------------------
    # 뉴스 일집계 로드 (경로 탐색) + features
    # ---------------------------
    def _try_load_news_daily(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        가능한 파일 후보:
        - {data_dir}/{TICKER}_news_daily.csv
        - {data_dir}/news/{TICKER}_news_daily.csv
        - {data_dir}/news/{TICKER}_news.csv
        - {data_dir}/processed/{TICKER}_news_daily.csv
        컬럼 기대(있으면 사용): Date, sentiment_mean, sentiment_vol, news_count_1d, news_count_7d
        """
        candidates = [
            os.path.join(self.data_dir, f"{ticker}_news_daily.csv"),
            os.path.join(self.data_dir, "news", f"{ticker}_news_daily.csv"),
            os.path.join(self.data_dir, "news", f"{ticker}_news.csv"),
            os.path.join(self.data_dir, "processed", f"{ticker}_news_daily.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    # 날짜 정규화
                    for col in ["date", "Date"]:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col]).dt.date
                    return df
                except Exception as e:
                    print(f"뉴스 로드 실패({p}): {e}")
        return None

    def _ctx_news_features(self, news_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if news_df is None or news_df.empty:
            return {
                "today": 0.0,
                "trend_7d": 0.0,
                "vol_7d": 0.0,
                "shock_z": 0.0,
                "news_count_1d": 0,
                "news_count_7d": 0,
                "headline_top_keywords": [],
                "keyword_polarity": {},
            }

        df = news_df.copy()
        # 컬럼 표준화
        if "Date" in df.columns:
            dcol = "Date"
        elif "date" in df.columns:
            dcol = "date"
        else:
            return {
                "today": 0.0, "trend_7d": 0.0, "vol_7d": 0.0, "shock_z": 0.0,
                "news_count_1d": 0, "news_count_7d": 0,
                "headline_top_keywords": [], "keyword_polarity": {}
            }
        df[dcol] = pd.to_datetime(df[dcol])

        # 감성 평균/분산 컬럼 후보
        s_mean_col = first_present(df.columns, ["sentiment_mean", "senti_mean", "sentiment"])
        s_vol_col  = first_present(df.columns, ["sentiment_vol", "senti_vol"])
        cnt1_col   = first_present(df.columns, ["news_count_1d", "count_1d", "n_1d"])
        cnt7_col   = first_present(df.columns, ["news_count_7d", "count_7d", "n_7d"])

        df = df.sort_values(dcol)
        today = df.iloc[-1] if len(df) > 0 else None
        today_senti = float_or_none(today[s_mean_col]) if (today is not None and s_mean_col) else 0.0

        # 7일 구간
        if len(df) >= 7:
            last7 = df.iloc[-7:]
            trend_7d = float_or_none(last7[s_mean_col].mean()) if s_mean_col else 0.0
            vol_7d   = float_or_none(last7[s_mean_col].std()) if s_mean_col else 0.0
            # 충격지수: 오늘 감성 - 7일 평균 / 7일 표준편차
            if s_mean_col and vol_7d and vol_7d > 0:
                shock_z = (today_senti - trend_7d) / vol_7d
            else:
                shock_z = 0.0
        else:
            trend_7d, vol_7d, shock_z = 0.0, 0.0, 0.0

        news_count_1d = int_or_none(today[cnt1_col]) if (today is not None and cnt1_col) else 0
        news_count_7d = int_or_none(today[cnt7_col]) if (today is not None and cnt7_col) else 0

        # 키워드/폴라리티 (데이터에 따라 없을 수 있음)
        headline_top_keywords = []
        keyword_polarity = {}

        return {
            "today": rfloat(today_senti),
            "trend_7d": rfloat(trend_7d),
            "vol_7d": rfloat(vol_7d),
            "shock_z": rfloat(shock_z),
            "news_count_1d": int(news_count_1d or 0),
            "news_count_7d": int(news_count_7d or 0),
            "headline_top_keywords": headline_top_keywords,
            "keyword_polarity": keyword_polarity,
        }

    # ---------------------------
    # regime_features (간단 버전: VIX, 섹터모멘텀)
    # ---------------------------
    def _ctx_regime_features(self) -> Dict[str, Any]:
        try:
            vix = yf.download("^VIX", period="3mo", interval="1d", progress=False)["Close"]
        except Exception:
            vix = pd.Series(dtype=float)
        try:
            spy = yf.download("SPY", period="3mo", interval="1d", progress=False)["Close"]
        except Exception:
            spy = pd.Series(dtype=float)
        try:
            xlk = yf.download("XLK", period="3mo", interval="1d", progress=False)["Close"]
        except Exception:
            xlk = pd.Series(dtype=float)

        # VIX 버킷팅
        def vix_bucket(v: Optional[float]) -> Optional[str]:
            if v is None:
                return None
            if v < 15: return "low"
            if v < 25: return "mid"
            return "high"

        vix_last = safe_last(vix, 1)
        vix_bkt = vix_bucket(rfloat(vix_last))

        # 섹터 모멘텀(20일)
        def momentum20(s: pd.Series) -> Optional[float]:
            if s is None or s.empty or len(s) < 21: return None
            return float(s.iloc[-1] / s.iloc[-21] - 1.0)

        spy_m = momentum20(spy)
        xlk_m = momentum20(xlk)

        # 시장 레짐(단순 분류)
        mkt = None
        if spy_m is not None and vix_last is not None:
            if spy_m > 0 and vix_last < 20:
                mkt = "bullish_calm"
            elif spy_m > 0 and vix_last >= 20:
                mkt = "bullish_volatile"
            elif spy_m <= 0 and vix_last >= 20:
                mkt = "bearish_volatile"
            else:
                mkt = "bearish_calm"

        return {
            "market_regime": mkt,
            "sector_momentum": {"XLK_vs_SPY_20d": rfloat((xlk_m or 0) - (spy_m or 0)) if (xlk_m is not None and spy_m is not None) else None},
            "vix_bucket": vix_bkt,
        }

    # ---------------------------
    # LLM 메시지 빌더 3종 (기존)
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
            "문장 개수나 1~5번 형식을 강요하지 않으며, 대신 충분하고 구체적인 증거를 포함한다. "
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


# ---------------------------
# 유틸
# ---------------------------
def safe_last(s: Optional[pd.Series], n: int = 1) -> Optional[float]:
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return None
    try:
        return float(s.iloc[-n])
    except Exception:
        return None

def rfloat(x) -> Optional[float]:
    try:
        return None if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else float(x)
    except Exception:
        return None

def int_or_none(x) -> Optional[int]:
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)
    except Exception:
        return None

def float_or_none(x) -> Optional[float]:
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    except Exception:
        return None

def first_present(cols, candidates) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None
