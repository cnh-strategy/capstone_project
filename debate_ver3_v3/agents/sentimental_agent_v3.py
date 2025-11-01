# debate_ver3_v3/agents/sentimental_agent_v3.py
# ===============================================================
# SentimentalAgentV3: 감성(뉴스/텍스트) + LSTM 기반 예측 에이전트 (v3)
#  - BaseAgent에 완전 호환 (reviewer_* 로직은 BaseAgent 구현 사용)
#  - 데이터는 core/data_set.py에서 만든 CSV를 BaseAgent.searcher로 로드
#  - 타깃은 "다음날 수익률"이며 BaseAgent.predict가 "가격"으로 변환
#  - LLM 설명력 강화를 위해 ctx를 price/volume/news/regime/uncertainty/
#    explainability/explain_helpers로 확장
#  - v3: 프롬프트는 debate_ver3_v3.prompts.OPINION_PROMPTS_V3를 사용
#  - PRE-CTX에서도 opinion/reason이 나오도록 minimal_pre_ctx를 제공
# ===============================================================

from __future__ import annotations
import json
from typing import Tuple, Optional, Dict, Any, List, Union

import torch
import torch.nn as nn

# .env 로드 (CAPSTONE_OPENAI_API -> OPENAI_API_KEY 매핑, override=True)
import os
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    if os.getenv("CAPSTONE_OPENAI_API") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("CAPSTONE_OPENAI_API")
except Exception:
    pass

from debate_ver3.agents.base_agent import BaseAgent, StockData, Target, Opinion
from debate_ver3.config.agents import agents_info
from debate_ver3_v3.prompts import OPINION_PROMPTS_V3
from debate_ver3.prompts import REBUTTAL_PROMPTS, REVISION_PROMPTS


# --- ensure agents_info has V3 alias ---
try:
    _ = agents_info["SentimentalAgentV3"]
except KeyError:
    agents_info["SentimentalAgentV3"] = dict(agents_info.get("SentimentalAgent", {}))
# ---------------------------------------


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
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)         # (B, T, H)
        h_last = out[:, -1, :]
        h_last = self.dropout(h_last)
        y = self.fc(h_last)           # (B, 1)  → "다음날 수익률"
        return y


# ---------------------------------------------------------------
# 에이전트 구현 (v3)
# ---------------------------------------------------------------
class SentimentalAgentV3(BaseAgent, nn.Module):
    def __init__(self, agent_id: str = "SentimentalAgentV3", verbose: bool = False, ticker: str = "TSLA"):
        BaseAgent.__init__(self, agent_id=agent_id, verbose=verbose, ticker=ticker)
        nn.Module.__init__(self)

        cfg = agents_info.get("SentimentalAgent", {})
        input_dim  = int(cfg.get("input_dim", 8))
        hidden_dim = int(cfg.get("d_model", 128))
        num_layers = int(cfg.get("num_layers", 2))
        dropout    = float(cfg.get("dropout", 0.2))

        self.net = SentimentalNet(input_dim, hidden_dim, num_layers, dropout)
        self.window_size = int(cfg.get("window_size", 40))

    # torch forward
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
    # -----------------------------------------------------------
    def _compute_prediction_block(self, stock_data: StockData, target: Target) -> Dict[str, Any]:
        last_price = float(self._g(stock_data, "last_price", 0.0) or 0.0)
        pred_close = float(self._g(target, "next_close", 0.0) or 0.0)

        if hasattr(target, "pred_return") and self._g(target, "pred_return", None) is not None:
            pred_return = float(self._g(target, "pred_return", 0.0) or 0.0)
        elif last_price > 0:
            pred_return = (pred_close - last_price) / last_price
        else:
            pred_return = 0.0

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
    # 내부 유틸: 뉴스/감성 블록
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
    # 내부 유틸: 가격/수익률·변동성 블록
    # -----------------------------------------------------------
    def _build_price_block(self, stock_data: StockData) -> Dict[str, Any]:
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
    # -----------------------------------------------------------
    def _build_explainability_block(self, stock_data: StockData, target: Target) -> Optional[Dict[str, Any]]:
        top_feats   = self._g(target, "top_features", None) or self._g(stock_data, "top_features", None)
        seq_summary = self._g(target, "sequence_attrib_summary", None) or self._g(stock_data, "sequence_attrib_summary", None)
        global_imp  = self._g(target, "feature_importance_global", None) or self._g(stock_data, "feature_importance_global", None)
        explain_mtd = self._g(target, "explain_method", None) or self._g(stock_data, "explain_method", None)

        if not (top_feats or seq_summary or global_imp):
            return None

        return {
            "top_features": top_feats,
            "sequence_attrib_summary": seq_summary,
            "feature_importance_global": global_imp,
            "method": explain_mtd or {"local": "IG", "global": "PermImp"},
        }

    # -----------------------------------------------------------
    # 내부 유틸: 설명 보조(Explain Helpers)
    # -----------------------------------------------------------
    def _build_explain_helpers(self, stock_data: StockData, target: Target) -> Optional[Dict[str, Any]]:
        delta = self._g(target, "feature_delta_top3", None) or self._g(stock_data, "feature_delta_top3", None)
        perm  = self._g(target, "perm_importance_top3", None) or self._g(stock_data, "perm_importance_top3", None)
        flags = self._g(target, "constraint_flags", None) or self._g(stock_data, "constraint_flags", None)

        if not (delta or perm or flags):
            return None

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
    # LLM: Opinion 메시지 구성 (v3 프롬프트 사용)
    # -----------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        self._enrich_ctx_before_prompt(stock_data, target)

        prompt_set = OPINION_PROMPTS_V3.get("SentimentalAgentV3") or {
            "system": (
                "너는 감성/뉴스 기반 단기 주가 분석가다. 반드시 아래 원칙을 지켜라.\n"
                "1) 입력은 오직 ctx(JSON)뿐이다. 외부 상식·추정·웹검색을 금지한다.\n"
                "2) 허용된 연산은 'ctx 내 값의 비교·차이·단순 비율/증감' 뿐이다. 새로운 지표를 창조하지 마라.\n"
                "3) '전후값 비교'는 snapshot.last_price(전) ↔ prediction.pred_close(후)를 기준으로 한다.\n"
                "4) 설명력: drivers/counter에는 반드시 ctx의 실제 키 경로를 넣고, 해당 값(value)을 ctx에서 그대로 인용한다.\n"
                "5) 불확실성: prediction.uncertainty.std / .pi80가 있으면 인용하고, 없으면 null을 유지한다.\n"
                "6) surprise_proxy는 다음 중 하나일 때만 true: news_features.shock_z >= 1.5, 또는 volume_features.volume_spike == true.\n"
                "7) 반환은 JSON 한 개만. 주석·설명·마크다운 금지. 키 누락 금지. 값이 없으면 null 또는 빈 배열을 사용한다.\n"
                "8) attempts(초안→검토→최종)에는 각 단계의 수정 이유를 1문장으로 요약하고, 과도한 chain-of-thought 금지.\n"
                "9) evidence에는 drivers, counter, data_notes만 허용한다. 'contradictions' 등 금지 키 생성 금지."
            ),
            "user": (
                "[CONTEXT]\n{context}\n\n"
                "[SCHEMA]\n{schema}\n\n"
                "[VALIDATION]\n"
                "- opinion.label은 'UP' | 'DOWN' | 'FLAT' 중 하나여야 한다.\n"
                "- before_after.direction_delta는 last_price 대비 pred_close의 방향을 'UP'|'DOWN'|'FLAT'으로 판단한다.\n"
                "- evidence.drivers[*].key 및 evidence.counter[*].key는 반드시 ctx의 실제 경로여야 한다.\n"
                "- 모든 수치는 ctx에서 복사한다. ctx 값이 null이면 결과도 null이어야 한다(추정 금지).\n"
                "- 오직 JSON만 출력하고, 추가 텍스트는 절대 포함하지 마라."
            ),
            "schema": json.dumps({
                "opinion": {
                    "label": "UP | DOWN | FLAT",
                    "one_liner": "핵심 한 줄 근거"
                },
                "before_after": {
                    "last_price": "number | null",
                    "pred_close": "number | null",
                    "pred_return": "number | null",
                    "direction_delta": "UP | DOWN | FLAT",
                    "prob_up": "number | null"
                },
                "evidence": {
                    "drivers": [
                        {"key": "ctx 경로", "value": "수치/값", "reason": "이 값이 방향성에 기여한 이유 (1문장)"},
                        {"key": "ctx 경로", "value": "수치/값", "reason": "최대 3개"}
                    ],
                    "counter": [
                        {"key": "ctx 경로", "value": "수치/값", "risk": "상반되는 신호/리스크 (최대 2개)"}
                    ],
                    "data_notes": ["플래그/제약 등(있으면)"]
                },
                "uncertainty": {
                    "level": "low | medium | high",
                    "std": "number | null",
                    "pi80": ["low | null", "high | null"],
                    "notes": ["간단 코멘트(선택)"]
                },
                "attempts": [
                    {"step": "초안", "why": "핵심 지표 근거", "edit": "보완 또는 유지 포인트"},
                    {"step": "검토", "why": "리스크 반영", "edit": "확신도/표현 조정"},
                    {"step": "최종", "why": "균형 판단", "edit": "결론 확정"}
                ],
                "flags": {
                    "surprise_proxy": "boolean",
                    "data_gaps": ["누락/결측 의심 키 경로들"],
                    "constraints": ["helpers.constraint_flags가 있으면 1~3개 인용"]
                }
            }, ensure_ascii=False)
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
            "explainability": explain,
            "explain_helpers": helpers,
        }

        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(
            context=json.dumps(ctx, ensure_ascii=False),
            schema=prompt_set.get("schema", "{}")
        )
        return system_text, user_text

    # -----------------------------------------------------------
    # ctx 확인용: LLM 호출 없이 메시지/컨텍스트 프리뷰
    # -----------------------------------------------------------
    def preview_opinion_ctx(self) -> Dict[str, Any]:
        if getattr(self, "_last_X", None) is None or self.stockdata is None:
            self.searcher(self.ticker)
        X_last = getattr(self, "_last_X", None)
        tgt = self.predict(X_last)

        sys_txt, usr_txt = self._build_messages_opinion(self.stockdata, tgt)

        ctx_payload: Dict[str, Any] = {"_raw_user": usr_txt}

        def _extract_ctx_json(user_msg: str) -> Optional[Dict[str, Any]]:
            tag = "[CONTEXT]"
            pos = user_msg.find(tag)
            if pos == -1:
                return None
            i = user_msg.find("{", pos)
            if i == -1:
                return None
            depth = 0
            end = None
            for j, ch in enumerate(user_msg[i:], start=i):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end is None:
                return None
            try:
                return json.loads(user_msg[i:end])
            except Exception:
                return None

        try:
            maybe = _extract_ctx_json(usr_txt)
            if isinstance(maybe, dict) and "snapshot" in maybe and "prediction" in maybe:
                ctx_payload = maybe
        except Exception:
            pass

        try:
            print("[CTX PREVIEW]", json.dumps(ctx_payload, indent=2, ensure_ascii=False))
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
    # Rebuttal / Revision 메시지
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

    # -----------------------------------------------------------
    # ctx 보강: 윈도우에서 설명용 피처 계산
    # -----------------------------------------------------------
    def _ensure_ctx_features_from_window(self, stock_data: StockData):
        X_last = getattr(self, "_last_X", None)
        fcols  = getattr(stock_data, "feature_cols", None)
        if X_last is None or fcols is None or len(fcols) == 0:
            return

        try:
            import numpy as np
            XF = X_last[0] if X_last.ndim == 3 else X_last

            def _ix(name: str) -> int:
                try:
                    return fcols.index(name)
                except ValueError:
                    return -1

            ix_close  = _ix("Close")
            ix_open   = _ix("Open")
            ix_high   = _ix("High")
            ix_low    = _ix("Low")
            ix_vol    = _ix("Volume")
            ix_ret    = _ix("returns")

            close = XF[:, ix_close] if ix_close >= 0 else None
            high  = XF[:, ix_high]  if ix_high  >= 0 else None
            low   = XF[:, ix_low]   if ix_low   >= 0 else None
            prevc = np.roll(close, 1) if close is not None else None
            vol   = XF[:, ix_vol]   if ix_vol   >= 0 else None
            ret   = XF[:, ix_ret]   if ix_ret   >= 0 else None

            def pct_change(a, lag=1):
                if a is None: return None
                b = np.array(a, dtype=float)
                c = np.full_like(b, np.nan)
                c[lag:] = (b[lag:] - b[:-lag]) / np.where(b[:-lag] == 0, np.nan, b[:-lag])
                return c

            def rolling_mean(a, n):
                if a is None: return None
                import numpy as _np
                b = _np.array(a, dtype=float)
                out = _np.full_like(b, _np.nan)
                for i in range(n-1, len(b)):
                    out[i] = _np.nanmean(b[i-n+1:i+1])
                return out

            def rolling_std(a, n):
                if a is None: return None
                import numpy as _np
                b = _np.array(a, dtype=float)
                out = _np.full_like(b, _np.nan)
                for i in range(n-1, len(b)):
                    out[i] = _np.nanstd(b[i-n+1:i+1], ddof=0)
                return out

            if close is not None:
                if ret is None:
                    ret = pct_change(close, 1)

                lag_ret_1  = np.nan if ret is None else ret[-2]
                lag_ret_5  = np.nan if ret is None else np.nanmean(ret[-6:-1]) if len(ret) >= 6 else np.nan
                lag_ret_20 = np.nan if ret is None else np.nanmean(ret[-21:-1]) if len(ret) >= 21 else np.nan

                trend_7d        = pct_change(close, 7)
                rolling_vol_20  = rolling_std(ret, 20) if ret is not None else None

                if (high is not None) and (low is not None):
                    hl = np.abs(high - low)
                    hc = np.abs(high - prevc) if prevc is not None else None
                    lc = np.abs(low  - prevc) if prevc is not None else None
                    if hc is not None and lc is not None:
                        import numpy as _np
                        tr = _np.nanmax(_np.vstack([hl, hc, lc]), axis=0)
                        atr_14_series = rolling_mean(tr, 14)
                        atr_14_val = atr_14_series[-1] if atr_14_series is not None else np.nan
                    else:
                        atr_14_val = np.nan
                else:
                    atr_14_val = np.nan

                mean20 = rolling_mean(close, 20)
                std20  = rolling_std(close, 20)
                z20 = (close - mean20) / std20 if (mean20 is not None and std20 is not None) else None

                if close is not None:
                    import numpy as _np
                    dd = _np.full_like(close, _np.nan)
                    mx = _np.full_like(close, _np.nan)
                    for i in range(len(close)):
                        start = max(0, i-19)
                        mx[i] = _np.nanmax(close[start:i+1])
                        dd[i] = close[i] / mx[i] - 1.0 if mx[i] not in (0, np.nan) else _np.nan
                    drawdown_20_series = dd
                    breakout_20_flag = bool(close[-1] > mx[-2]) if len(close) >= 21 and not np.isnan(mx[-2]) else False
                else:
                    z20 = None
                    drawdown_20_series = None
                    breakout_20_flag = False

                if vol is not None:
                    vm20 = rolling_mean(vol, 20)
                    vs20 = rolling_std(vol, 20)
                    vol_z20_series = (vol - vm20) / vs20 if (vm20 is not None and vs20 is not None) else None
                    vol_z20_val = vol_z20_series[-1] if vol_z20_series is not None else np.nan
                    denom = vm20[-1] if vm20 is not None else np.nan
                    turnover_rate_val = float(vol[-1] / denom) if denom and not np.isnan(denom) and denom != 0 else 0.0
                    volume_spike_flag = bool(vol_z20_val >= 2.0) if not np.isnan(vol_z20_val) else False
                else:
                    vol_z20_val = np.nan
                    turnover_rate_val = 0.0
                    volume_spike_flag = False

                stock_data.__dict__.update({
                    "lag_ret_1":        None if np.isnan(lag_ret_1) else float(lag_ret_1),
                    "lag_ret_5":        None if np.isnan(lag_ret_5) else float(lag_ret_5),
                    "lag_ret_20":       None if np.isnan(lag_ret_20) else float(lag_ret_20),
                    "rolling_vol_20":   None if (rolling_vol_20 is None or np.isnan(rolling_vol_20[-1])) else float(rolling_vol_20[-1]),
                    "trend_7d":         None if (trend_7d is None or np.isnan(trend_7d[-1])) else float(trend_7d[-1]),
                    "atr_14":           None if np.isnan(atr_14_val) else float(atr_14_val),
                    "breakout_20":      bool(breakout_20_flag),
                    "zscore_close_20":  None if (z20 is None or np.isnan(z20[-1])) else float(z20[-1]),
                    "drawdown_20":      None if (drawdown_20_series is None or np.isnan(drawdown_20_series[-1])) else float(drawdown_20_series[-1]),
                    "vol_zscore_20":    None if np.isnan(vol_z20_val) else float(vol_z20_val),
                    "turnover_rate":    float(turnover_rate_val),
                    "volume_spike":     bool(volume_spike_flag),
                })
        except Exception:
            pass

    def _enrich_ctx_before_prompt(self, stock_data: StockData, target: Target):
        need_price = True
        need_volume = True
        for k in ["lag_ret_1","lag_ret_5","lag_ret_20","rolling_vol_20","trend_7d","atr_14","zscore_close_20","drawdown_20"]:
            if getattr(stock_data, k, None) is not None:
                need_price = False
                break
        for k in ["vol_zscore_20","turnover_rate","volume_spike"]:
            if getattr(stock_data, k, None) is not None:
                need_volume = False
                break
        if need_price or need_volume:
            self._ensure_ctx_features_from_window(stock_data)

    # -----------------------------------------------------------
    # LLM 엔트리포인트 (OpenAI 사용, 실패 시 로컬 Fallback)
    # -----------------------------------------------------------
    def call_llm(self, system: str, user: str):
        try:
            from openai import OpenAI
            client = getattr(self, "_openai_client", None)
            if client is None:
                client = OpenAI()
                self._openai_client = client
            resp = client.chat.completions.create(
                model=getattr(self, "opinion_model", "gpt-4o-mini"),
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception:
            # ===== Local fallback: user에서 Context JSON을 추출 → 간단 규칙으로 요약 =====
            import re, json as _json
            m = re.search(r"Context:\s*(\{.*\})", user, re.S)
            ctx = None
            if m:
                try:
                    ctx = _json.loads(m.group(1))
                except Exception:
                    pass
            if not ctx:
                # 완전 무컨텍스트인 경우: 모델의 전/후 가격만으로 베이스라인 JSON
                if getattr(self, "_last_X", None) is None or getattr(self, "stockdata", None) is None:
                    self.searcher(getattr(self, "ticker", "TSLA"))
                X_last = getattr(self, "_last_X", None)
                tgt    = self.predict(X_last)
                sd     = getattr(self, "stockdata", None)
                last = float(getattr(sd, "last_price", 0.0) or 0.0) if sd else None
                pred = float(getattr(tgt, "next_close", 0.0) or 0.0) if tgt else None
                ret  = None
                if last and pred:
                    ret = (pred - last) / last
                direction = "FLAT"
                if ret is not None:
                    direction = "UP" if ret > 0 else "DOWN" if ret < 0 else "FLAT"
                return _json.dumps({
                    "opinion": {"label": direction, "one_liner": "컨텍스트 없이 모델의 전/후 가격만으로 판단"},
                    "before_after": {
                        "last_price": round(last, 4) if last else None,
                        "pred_close": round(pred, 4) if pred else None,
                        "pred_return": round(ret, 4) if ret is not None else None,
                        "direction_delta": direction,
                        "prob_up": getattr(tgt, "calibrated_prob_up", None)
                    },
                    "evidence": {"drivers": ["prediction.pred_return"], "counter": [], "data_notes": ["no_context"]},
                    "uncertainty": {
                        "level": "medium",
                        "std": (getattr(tgt, "uncertainty", {}) or {}).get("std") if hasattr(tgt, "uncertainty") else None,
                        "pi80": [None, None],
                        "notes": []
                    },
                    "attempts": [{"step": "최종", "why": "정보 없음", "edit": "보수적 판단"}],
                    "flags": {"surprise_proxy": False, "data_gaps": ["context/*"], "constraints": []}
                }, ensure_ascii=False)

            snap = (ctx.get("snapshot") or {})
            pred = (ctx.get("prediction") or {})
            last = snap.get("last_price"); pc = pred.get("pred_close")
            pret = pred.get("pred_return"); std = (pred.get("uncertainty") or {}).get("std")
            conf = pred.get("confidence")

            direction = "FLAT"
            try:
                if pret is not None:
                    direction = "UP" if pret > 0 else "DOWN" if pret < 0 else "FLAT"
                elif last is not None and pc is not None:
                    direction = "UP" if pc > last else "DOWN" if pc < last else "FLAT"
            except Exception:
                pass

            return f"[LOCAL] opinion: {direction}; last={last}, pred={pc}, ret={pret}, std={std}, conf={conf}"


# === BEGIN: compatibility shim for ctx comparison ===
def _compat__opinion__call_llm_fallback(self, system_msg: str, user_msg: str):
    for fn in ["call_llm", "_call_llm", "infer_opinion", "generate_text", "llm_call"]:
        if hasattr(self, fn) and callable(getattr(self, fn)):
            return getattr(self, fn)(system_msg, user_msg)
    if hasattr(self, "llm"):
        llm = getattr(self, "llm")
        for fn in ["generate", "chat", "call", "complete"]:
            if hasattr(llm, fn) and callable(getattr(llm, fn)):
                return getattr(llm, fn)(system_msg, user_msg)
    raise RuntimeError("[compat] LLM 호출 엔트리포인트를 찾지 못했습니다. call_llm(system, user) 형태의 메서드를 하나 노출해 주세요.")


def _compat_build_message_opinion(self, use_ctx: bool = True, ticker: str | None = None):
    # 시스템 프롬프트
    try:
        prompt_set = OPINION_PROMPTS_V3.get("SentimentalAgentV3", {})
        system_msg = (
            prompt_set.get("system")
            or "너는 감성/뉴스 기반 단기 주가 분석 전문가다. 모델의 예측값과 불확실성을 근거로 간결하게 의견을 제시해."
        )
    except Exception:
        system_msg = "너는 감성/뉴스 기반 단기 주가 분석 전문가다. 모델의 예측값과 불확실성을 근거로 간결하게 의견을 제시해."
    base_user = "아래 컨텍스트(있다면)를 바탕으로 한국어 3문장 이내 의견을 작성:"

    # WITH-CTX: 기존 full ctx
    if use_ctx:
        ctx_json = None
        try:
            if hasattr(self, "preview_opinion_ctx") and callable(self.preview_opinion_ctx):
                prev = self.preview_opinion_ctx()
                ctx_json = prev.get("ctx_preview", prev)
            elif hasattr(self, "build_ctx") and callable(self.build_ctx):
                ctx_json = self.build_ctx()
        except Exception:
            ctx_json = None

        if ctx_json is not None:
            try:
                ctx_str = json.dumps(ctx_json, ensure_ascii=False)
            except Exception:
                ctx_str = str(ctx_json)
            user_msg = f"{base_user}\nContext: {ctx_str}"
        else:
            user_msg = base_user
        return system_msg, user_msg

    # PRE-CTX: minimal_pre_ctx (snapshot + prediction만)
    if getattr(self, "_last_X", None) is None or getattr(self, "stockdata", None) is None:
        self.searcher(getattr(self, "ticker", ticker) or "TSLA")
    X_last = getattr(self, "_last_X", None)
    tgt    = self.predict(X_last)
    sd     = getattr(self, "stockdata", None)

    pred_block = self._compute_prediction_block(sd, tgt) if (sd and tgt) else {}
    last_price = float(getattr(sd, "last_price", 0.0) or 0.0) if sd else None

    ctx_min = {
        "_mode": "minimal_pre_ctx",
        "agent_id": getattr(self, "agent_id", "SentimentalAgentV3"),
        "ticker": getattr(sd, "ticker", None) if sd else None,
        "snapshot": {
            "asof_date": getattr(sd, "asof_date", None) if sd else None,
            "last_price": round(last_price, 4) if last_price else None,
            "window_size": getattr(self, "window_size", None),
            "feature_cols_preview": (getattr(sd, "feature_cols", None) or [])[:8] if sd else [],
        },
        "prediction": pred_block,
    }

    user_msg = (
        f"{base_user}\n"
        "[주의] 이 컨텍스트는 최소 정보만 포함합니다. 감성/레짐/거래량 등 부가 지표는 포함하지 마세요.\n"
        f"Context: {json.dumps(ctx_min, ensure_ascii=False)}"
    )
    return system_msg, user_msg


def _compat_opinion(self, use_ctx: bool = True):
    system_msg, user_msg = self.build_message_opinion(use_ctx=use_ctx)
    raw = _compat__opinion__call_llm_fallback(self, system_msg, user_msg)

    # 텍스트 추출
    text = None
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        text = raw.get("content") or raw.get("text") or raw.get("reason") or raw.get("output")
    elif hasattr(raw, "content"):
        text = getattr(raw, "content")
    elif hasattr(raw, "text"):
        text = getattr(raw, "text")
    if text is None:
        text = str(raw)

    # 가능하면 JSON으로 파싱
    parsed = None
    if isinstance(text, str) and text.strip().startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    op = {
        "agent_id": getattr(self, "agent_id", "SentimentalAgentV3"),
        "text": parsed if parsed is not None else text,
        "target": getattr(self, "last_target", None),
        "prediction": getattr(self, "last_prediction", None),
        "confidence": getattr(self, "last_confidence", None),
        "uncertainty": getattr(self, "last_uncertainty", None),
        "pre_post": getattr(self, "pre_post", None),
        "reason": parsed if parsed is not None else text,
    }
    return op


# 클래스에 메서드 주입 (이미 같은 이름이 있으면 덮어쓰지 않음)
if not hasattr(SentimentalAgentV3, "build_message_opinion"):
    SentimentalAgentV3.build_message_opinion = _compat_build_message_opinion
if not hasattr(SentimentalAgentV3, "opinion"):
    SentimentalAgentV3.opinion = _compat_opinion
# === END: compatibility shim for ctx comparison ===
