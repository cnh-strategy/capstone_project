# debate_ver3/agents/sentimental_agent.py
# ===============================================================
# SentimentalAgent: 감성(뉴스/텍스트) + LSTM 기반 예측 에이전트
#  - BaseAgent에 완전 호환 (reviewer_* 로직은 BaseAgent 구현 사용)
#  - 데이터는 core/data_set.py에서 만든 CSV를 BaseAgent.searcher로 로드
#  - 타깃은 "다음날 수익률"이며 BaseAgent.predict가 "가격"으로 변환
#  - LLM 설명력 강화를 위해 ctx를 price/volume/news/regime/uncertainty/explain_helpers로 확장
# ===============================================================

from __future__ import annotations
import json
from typing import Tuple, Optional, Dict, Any, List, Union

import torch
import torch.nn as nn

from debate_ver3.agents.base_agent import BaseAgent, StockData, Target, Opinion
from debate_ver3.config.agents import agents_info
from debate_ver3.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS


# ---------------------------------------------------------------
# 모델 정의: LSTM + Dropout (MC Dropout을 위해 train() 상태에서 사용)
# ---------------------------------------------------------------
class SentimentalNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)  # MC Dropout: 추론 시에도 유지하려면 .train() 유지 필요
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)         # (B, T, H)
        h_last = out[:, -1, :]        # (B, H)
        h_last = self.dropout(h_last) # MC Dropout: 학습/추론 공통 적용
        y = self.fc(h_last)           # (B, 1)  → "다음날 수익률(정규화/원공간 여부는 BaseAgent에 위임)"
        return y


# ---------------------------------------------------------------
# 에이전트 구현
# ---------------------------------------------------------------
class SentimentalAgent(BaseAgent, nn.Module):
    def __init__(self, agent_id: str = "SentimentalAgent", verbose: bool = False, ticker: str = "TSLA"):
        BaseAgent.__init__(self, agent_id=agent_id, verbose=verbose, ticker=ticker)
        nn.Module.__init__(self)

        cfg = agents_info.get("SentimentalAgent", {})
        input_dim  = int(cfg.get("input_dim", 8))
        hidden_dim = int(cfg.get("d_model", 128))
        num_layers = int(cfg.get("num_layers", 2))
        dropout    = float(cfg.get("dropout", 0.2))

        self.net = SentimentalNet(input_dim, hidden_dim, num_layers, dropout)
        self.window_size = int(cfg.get("window_size", 40))

    # -----------------------------------------------------------
    # torch forward
    # -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # -----------------------------------------------------------
    # 내부 유틸: 안전 접근 & 수치 반올림
    # -----------------------------------------------------------
    def _g(self, obj, name, default=None):
        return getattr(obj, name, default)

    def _r(self, x: Optional[Union[int, float]], n: int = 4):
        if x is None:
            return None
        try:
            return round(float(x), n)
        except Exception:
            return None

    # -----------------------------------------------------------
    # 내부 유틸: 예측 블록
    #   - BaseAgent.predict(Target)의 산출물을 우선 사용
    #   - next_close, pred_return, uncertainty(std/ci95/pi80), confidence, calibrated_prob_up
    # -----------------------------------------------------------
    def _compute_prediction_block(self, stock_data: StockData, target: Target) -> Dict[str, Any]:
        last_price = float(self._g(stock_data, "last_price", 0.0) or 0.0)
        pred_close = float(self._g(target, "next_close", 0.0) or 0.0)

        # pred_return 우선순위: target.pred_return -> 유도 계산 -> 0.0
        if hasattr(target, "pred_return") and self._g(target, "pred_return", None) is not None:
            pred_return = float(self._g(target, "pred_return", 0.0) or 0.0)
        elif last_price > 0:
            pred_return = (pred_close - last_price) / last_price
        else:
            pred_return = 0.0

        # uncertainty는 float/dict 모두 허용
        unc = self._g(target, "uncertainty", 0.0) or 0.0
        pi80 = None
        if isinstance(unc, dict):
            std = self._r(unc.get("std", 0.0))
            ci95 = unc.get("ci95") or None
            if "pi80" in unc and isinstance(unc["pi80"], (list, tuple)) and len(unc["pi80"]) == 2:
                pi80 = [self._r(unc["pi80"][0]), self._r(unc["pi80"][1])]
        else:
            std = self._r(unc)
            ci95 = None

        confidence = self._r(self._g(target, "confidence", 0.0), 3)
        prob_up = self._g(target, "calibrated_prob_up", None)
        prob_up = self._r(prob_up, 3) if prob_up is not None else None

        # MC 추정치가 target에 담겨 있다면 노출
        mc_mean = self._g(target, "mc_mean_next_close", None)
        mc_std  = self._g(target, "mc_std", None)

        return {
            "pred_close": self._r(pred_close),
            "pred_return": self._r(pred_return),
            "uncertainty": {
                "std": std,
                "ci95": ci95,
                "pi80": pi80
            },
            "confidence": confidence,
            "calibrated_prob_up": prob_up,
            "mc_mean_next_close": self._r(mc_mean),
            "mc_std": self._r(mc_std),
        }

    # -----------------------------------------------------------
    # 내부 유틸: 뉴스/감성 요약 블록 (유연한 컬럼 매핑)
    #   - today, trend_7d, vol_7d, shock_z, 기사 수(1d/7d)
    # -----------------------------------------------------------
    def _build_news_block(self, stock_data: StockData) -> Dict[str, Any]:
        def _sa() -> Dict[str, Any]:
            return getattr(stock_data, "SentimentalAgent", {}) or {}

        def _pick(*keys, default=None):
            sa = _sa()
            for k in keys:
                if k in sa and sa[k] is not None:
                    return sa[k]
            return default

        today = _pick("news_sentiment", "sentiment_mean", "finbert_mean_1d")
        trend = _pick("sentiment_trend_7d", "sentiment_trend", "finbert_trend_7d")
        vol   = _pick("sentiment_volatility_7d", "sentiment_vol", "finbert_std_7d")
        shock = _pick("sentiment_shock_score", "sentiment_shock", "news_shock")
        n1d   = _pick("news_count_1d", default=0) or 0
        n7d   = _pick("news_count_7d", default=0) or 0

        # 키워드/극성 옵션(있으면 사용)
        keywords: List[str] = _pick("headline_top_keywords", "top_keywords") or []
        polarity_map: Dict[str, Any] = _pick("keyword_polarity") or {}

        return {
            "today":        self._r(today, 3),
            "trend_7d":     self._r(trend, 3),
            "vol_7d":       self._r(vol, 3),
            "shock_z":      self._r(shock, 3),
            "news_count_1d": int(n1d),
            "news_count_7d": int(n7d),
            "headline_top_keywords": keywords[:5],
            "keyword_polarity": polarity_map,
        }

    # -----------------------------------------------------------
    # 내부 유틸: 가격/수익률·변동성 블록 (설명용 피처)
    #   - lag_ret_1/5/20, rolling_vol_20, trend_7d, atr_14, breakout_20
    #   - 입력은 stock_data의 다양한 이름을 허용
    # -----------------------------------------------------------
    def _build_price_block(self, stock_data: StockData) -> Dict[str, Any]:
        def _s() -> Dict[str, Any]:
            # 원천 피처 딕셔너리(파이프라인에 따라 위치 다를 수 있음)
            # 우선순위: stock_data.features -> stock_data.__dict__
            d = {}
            try:
                d.update(getattr(stock_data, "features", {}) or {})
            except Exception:
                pass
            d.update(getattr(stock_data, "__dict__", {}) or {})
            return d

        def _pick(*keys, default=None):
            s = _s()
            for k in keys:
                if k in s and s[k] is not None:
                    return s[k]
            return default

        return {
            "lag_ret_1":        self._r(_pick("lag_ret_1", "ret_lag_1")),
            "lag_ret_5":        self._r(_pick("lag_ret_5", "ret_lag_5")),
            "lag_ret_20":       self._r(_pick("lag_ret_20", "ret_lag_20")),
            "rolling_vol_20":   self._r(_pick("rolling_vol_20", "vol_20")),
            "trend_7d":         self._r(_pick("trend_7d", "price_trend_7d")),
            "atr_14":           self._r(_pick("atr_14", "ATR14", "atr")),
            "breakout_20":      bool(_pick("breakout_20", "is_breakout_20", default=False)),
            "zscore_close_20":  self._r(_pick("zscore_close_20", "z_close_20")),
            "drawdown_20":      self._r(_pick("drawdown_20", "dd_20")),
        }

    # -----------------------------------------------------------
    # 내부 유틸: 거래/유동성 블록
    #   - vol_zscore_20, turnover_rate, volume_spike
    # -----------------------------------------------------------
    def _build_volume_block(self, stock_data: StockData) -> Dict[str, Any]:
        def _s() -> Dict[str, Any]:
            d = {}
            try:
                d.update(getattr(stock_data, "features", {}) or {})
            except Exception:
                pass
            d.update(getattr(stock_data, "__dict__", {}) or {})
            return d

        def _pick(*keys, default=None):
            s = _s()
            for k in keys:
                if k in s and s[k] is not None:
                    return s[k]
            return default

        return {
            "vol_zscore_20": self._r(_pick("vol_zscore_20", "volume_z_20")),
            "turnover_rate": self._r(_pick("turnover_rate", "turnover")),
            "volume_spike":  bool(_pick("volume_spike", "is_volume_spike", default=False)),
        }

    # -----------------------------------------------------------
    # 내부 유틸: 레짐/시장 상태 블록
    #   - market_regime, sector_momentum, vix_bucket
    # -----------------------------------------------------------
    def _build_regime_block(self, stock_data: StockData) -> Dict[str, Any]:
        def _s() -> Dict[str, Any]:
            d = {}
            try:
                d.update(getattr(stock_data, "features", {}) or {})
            except Exception:
                pass
            d.update(getattr(stock_data, "__dict__", {}) or {})
            return d

        def _pick(*keys, default=None):
            s = _s()
            for k in keys:
                if k in s and s[k] is not None:
                    return s[k]
            return default

        regime = _pick("market_regime", "regime", default=None)
        sector_mom = _pick("sector_momentum", "sector_mom", default=None)
        vix_bucket = _pick("vix_bucket", "vix_level_bucket", default=None)

        return {
            "market_regime": regime,
            "sector_momentum": self._r(sector_mom),
            "vix_bucket": vix_bucket,
        }

    # -----------------------------------------------------------
    # 내부 유틸: 설명가능성(요약) 블록
    #   - SHAP/IG/Permutation 결과가 있을 때만 요약형 포함
    #   - raw 배열은 숨기고 Top-K/타임스텝 요약만
    # -----------------------------------------------------------
    def _build_explainability_block(self, stock_data: StockData, target: Target) -> Optional[Dict[str, Any]]:
        top_feats   = self._g(target, "top_features", None) or self._g(stock_data, "top_features", None)
        seq_summary = self._g(target, "sequence_attrib_summary", None) or self._g(stock_data, "sequence_attrib_summary", None)
        global_imp  = self._g(target, "feature_importance_global", None) or self._g(stock_data, "feature_importance_global", None)
        explain_mtd = self._g(target, "explain_method", None) or self._g(stock_data, "explain_method", None)

        if not (top_feats or seq_summary or global_imp):
            return None

        return {
            "top_features": top_feats,                    # 예: [{"name":"news_sentiment","contribution":0.38,"sign":+1}, ...]
            "sequence_attrib_summary": seq_summary,       # 예: {"top_timesteps":[{"t":-3,"weight":0.28},...]}
            "feature_importance_global": global_imp,      # optional
            "method": explain_mtd or {"local": "IG", "global": "PermImp"},
        }

    # -----------------------------------------------------------
    # 내부 유틸: 설명 보조(Explain Helpers)
    #   - feature_delta_top3, perm_importance_top3, constraint_flags
    # -----------------------------------------------------------
    def _build_explain_helpers(self, stock_data: StockData, target: Target) -> Optional[Dict[str, Any]]:
        # 우선 Target, 없으면 StockData
        delta = self._g(target, "feature_delta_top3", None) or self._g(stock_data, "feature_delta_top3", None)
        perm  = self._g(target, "perm_importance_top3", None) or self._g(stock_data, "perm_importance_top3", None)
        flags = self._g(target, "constraint_flags", None) or self._g(stock_data, "constraint_flags", None)

        if not (delta or perm or flags):
            return None

        # 문자열/리스트 모두 허용하되, 길이 제약
        def _norm3(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return list(x)[:3]
            return [str(x)][:3]

        return {
            "feature_delta_top3": _norm3(delta),
            "perm_importance_top3": _norm3(perm),
            "constraint_flags": _norm3(flags),
        }

    # -----------------------------------------------------------
    # LLM: Opinion 메시지 구성 (설명력 강화 ctx)
    # -----------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """
        SentimentalAgent 전용 컨텍스트:
         - snapshot(메타) + price/volume/news/regime
         - prediction(예측/불확실성/확률/MC) + explainability + explain_helpers
        """
        prompt_set = OPINION_PROMPTS.get("SentimentalAgent") or {
            "system": (
                "너는 감성/뉴스 기반 단기 주가 분석 전문가다. "
                "아래 ctx의 지표만으로 의견을 만들고, 외부 추정은 금지한다. "
                "모델의 예측값·불확실성과 데이터를 근거로 간결하게 제시해."
            ),
            "user": "아래 컨텍스트를 바탕으로 한국어 3문장 이내 의견을 작성:\nContext: {context}",
        }

        prediction = self._compute_prediction_block(stock_data, target)
        news       = self._build_news_block(stock_data)
        price      = self._build_price_block(stock_data)
        volume     = self._build_volume_block(stock_data)
        regime     = self._build_regime_block(stock_data)
        explain    = self._build_explainability_block(stock_data, target)
        helpers    = self._build_explain_helpers(stock_data, target)

        last_price = float(self._g(stock_data, "last_price", 0.0) or 0.0)
        ctx = {
            "agent_id": self.agent_id,
            "ticker": self._g(stock_data, "ticker", None),
            "snapshot": {
                "asof_date": self._g(stock_data, "asof_date", None),
                "last_price": self._r(last_price),
                "currency": self._g(stock_data, "currency", None),
                "window_size": getattr(self, "window_size", None),
                "feature_cols_preview": (self._g(stock_data, "feature_cols", None) or [])[:12],
            },
            "prediction": prediction,
            "price_features": price,
            "volume_features": volume,
            "news_features": news,
            "regime_features": regime,
            "explainability": explain,         # 없으면 JSON null
            "explain_helpers": helpers,        # 없으면 JSON null
        }

        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    # -----------------------------------------------------------
    # ctx 확인용: LLM 호출 없이 메시지/컨텍스트 프리뷰
    # -----------------------------------------------------------
    def preview_opinion_ctx(self) -> Dict[str, Any]:
        # 0) 최신 입력/타깃 확보
        if getattr(self, "_last_X", None) is None or self.stockdata is None:
            self.searcher(self.ticker)
        X_last = getattr(self, "_last_X", None)
        tgt = self.predict(X_last)

        # 1) 메시지 생성 (→ 여기서 ctx가 만들어짐)
        sys_txt, usr_txt = self._build_messages_opinion(self.stockdata, tgt)

        # 2) user 메시지에서 ctx만 추출 (문자열에서 첫 '{' ~ 마지막 '}' 구간 파싱)
        ctx_payload: Dict[str, Any] = {"_raw_user": usr_txt}
        try:
            start = usr_txt.find("{"); end = usr_txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                maybe_json = usr_txt[start:end+1]
                ctx_payload = json.loads(maybe_json)
        except Exception:
            pass

        return {
            "system_msg": sys_txt,
            "user_msg": usr_txt,
            "ctx_preview": ctx_payload,
            "target_preview": {
                "next_close": float(self._g(tgt, "next_close", 0.0) or 0.0),
                "uncertainty": getattr(tgt, "uncertainty", None),
                "confidence": getattr(tgt, "confidence", None),
            }
        }

    # -----------------------------------------------------------
    # Rebuttal / Revision 메시지 (기본 템플릿 유지)
    # -----------------------------------------------------------
    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        prompt_set = REBUTTAL_PROMPTS.get("SentimentalAgent") or {
            "system": "너는 토론에서 상대 의견의 강점/약점을 짚는 분석가다.",
            "user": "컨텍스트를 바탕으로 REBUT 또는 SUPPORT와 메시지를 JSON으로 답하라:\n{context}",
        }
        context = kwargs.get("context", "")
        return prompt_set["system"], prompt_set["user"].format(context=context)

    def _build_messages_revision(self, *args, **kwargs) -> Tuple[str, str]:
        prompt_set = REVISION_PROMPTS.get("SentimentalAgent") or REVISION_PROMPTS.get("default") or {
            "system": "수정된 예측값과 반박 내용을 반영해 최종 의견을 간결히 정리하라.",
            "user": "컨텍스트:\n{context}",
        }
        context = kwargs.get("context", "")
        return prompt_set["system"], prompt_set["user"].format(context=context)
