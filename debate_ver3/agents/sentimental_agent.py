# debate_ver3/agents/sentimental_agent.py
# ===============================================================
# SentimentalAgent: 감성(뉴스/텍스트) + LSTM 기반 예측 에이전트
#  - BaseAgent에 완전 호환 (reviewer_* 로직은 BaseAgent 구현 사용)
#  - 데이터는 core/data_set.py에서 만든 CSV를 BaseAgent.searcher로 로드
#  - 타깃은 "다음날 수익률"이며 BaseAgent.predict가 "가격"으로 변환
# ===============================================================

from __future__ import annotations
import json
from typing import Tuple, Optional, Dict, Any

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
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)         # (B, T, H)
        h_last = out[:, -1, :]        # (B, H)
        h_last = self.dropout(h_last) # MC Dropout: 학습/추론 공통 적용
        y = self.fc(h_last)           # (B, 1)  → "다음날 수익률(정규화 공간)"
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # -----------------------------------------------------------
    # 내부 유틸: 안전 접근
    # -----------------------------------------------------------
    def _g(self, obj, name, default=None):
        return getattr(obj, name, default)

    # -----------------------------------------------------------
    # 내부 유틸: 예측 블록 구성
    #   - pred_close: BaseAgent.predict의 결과를 우선 사용
    #   - pred_return: target.pred_return 있으면 사용, 없으면 last_price로부터 유도
    #   - uncertainty/confidence/calibrated_prob_up: target에서 그대로 전달(없으면 안전값)
    # -----------------------------------------------------------
    def _compute_prediction_block(self, stock_data: StockData, target: Target):
        last_price = float(self._g(stock_data, "last_price", 0.0) or 0.0)
        pred_close = float(self._g(target, "next_close", 0.0) or 0.0)

        # pred_return 우선순위: target.pred_return -> 유도 계산 -> 0.0
        if hasattr(target, "pred_return") and self._g(target, "pred_return", None) is not None:
            pred_return = float(self._g(target, "pred_return", 0.0) or 0.0)
        elif last_price > 0:
            pred_return = (pred_close - last_price) / last_price
        else:
            pred_return = 0.0

        unc = self._g(target, "uncertainty", 0.0) or 0.0
        if isinstance(unc, dict):
            uncertainty = {
                "std": float(unc.get("std", 0.0)),
                "ci95": unc.get("ci95") or None,
            }
        else:
            uncertainty = {"std": float(unc), "ci95": None}

        confidence = float(self._g(target, "confidence", 0.0) or 0.0)
        prob_up = self._g(target, "calibrated_prob_up", None)
        prob_up = float(prob_up) if prob_up is not None else None

        return {
            "pred_close": round(pred_close, 4),
            "pred_return": round(pred_return, 4),
            "uncertainty": {
                "std": round(uncertainty["std"], 4),
                "ci95": uncertainty["ci95"],
            },
            "confidence": round(confidence, 3),
            "calibrated_prob_up": (round(prob_up, 3) if prob_up is not None else None),
        }

    # -----------------------------------------------------------
    # 내부 유틸: 감성(뉴스) 요약 블록
    #  - 다양한 컬럼명을 허용해 매핑 (_pick 사용)
    # -----------------------------------------------------------
    def _build_sentiment_block(self, stock_data: StockData):
        def _sa() -> Dict[str, Any]:
            return getattr(stock_data, "SentimentalAgent", {}) or {}

        def _pick(*keys, default=None):
            sa = _sa()
            for k in keys:
                if k in sa and sa[k] is not None:
                    return sa[k]
            return default

        def _num(x):
            return None if x is None else float(x)

        # 다양한 이름을 허용 (CSV/파이프라인별 컬럼명 차이 흡수)
        today = _num(_pick("news_sentiment", "sentiment_mean"))
        trend = _num(_pick("sentiment_trend_7d", "sentiment_trend"))
        vol   = _num(_pick("sentiment_volatility_7d", "sentiment_vol"))
        shock = _num(_pick("sentiment_shock_score", "sentiment_shock", "news_shock"))
        n1d   = _pick("news_count_1d", default=0) or 0
        n7d   = _pick("news_count_7d", default=0) or 0

        return {
            "today":        (round(today, 3) if today is not None else None),
            "trend_7d":     (round(trend, 3) if trend is not None else None),
            "vol_7d":       (round(vol, 3) if vol is not None else None),
            "shock_z":      (round(shock, 3) if shock is not None else None),
            "news_count_1d": int(n1d),
            "news_count_7d": int(n7d),
        }

    # -----------------------------------------------------------
    # 내부 유틸: 설명가능성(요약) 블록
    #   - SHAP/IG/Permutation 결과가 있을 때만 요약형으로 포함
    #   - raw 배열을 노출하지 않고 Top-K/타임스텝 요약만 포함
    # -----------------------------------------------------------
    def _build_explainability_block(self, stock_data: StockData, target: Target):
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
    # LLM: Opinion 메시지 구성
    # -----------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """
        SentimentalAgent 전용 컨텍스트:
         - 모델 예측/불확실성(상속/기존 값 우선)
         - 뉴스 감성 요약(오늘/추세/이례성/기사수)
         - 설명가능성 요약(SHAP 등 Top-K, 있으면)
         - 스냅샷(가격/화폐/윈도/피처 일부)
        """
        prompt_set = OPINION_PROMPTS.get("SentimentalAgent") or {
            "system": (
                "너는 감성/뉴스 기반 단기 주가 분석 전문가다. "
                "모델의 예측값과 불확실성을 근거로 간결하게 의견을 제시해."
            ),
            "user": "아래 컨텍스트를 바탕으로 한국어 3문장 이내 의견을 작성:\nContext: {context}",
        }

        prediction = self._compute_prediction_block(stock_data, target)
        sentiment  = self._build_sentiment_block(stock_data)
        explain    = self._build_explainability_block(stock_data, target)

        last_price = float(self._g(stock_data, "last_price", 0.0) or 0.0)
        ctx = {
            "agent_id": self.agent_id,
            "ticker": self._g(stock_data, "ticker", None),
            "snapshot": {
                "asof_date": self._g(stock_data, "asof_date", None),
                "last_price": round(last_price, 4),
                "currency": self._g(stock_data, "currency", None),
                "window_size": getattr(self, "window_size", None),
                "feature_cols_preview": (self._g(stock_data, "feature_cols", None) or [])[:12],
            },
            "prediction": prediction,
            "sentiment": sentiment,
            "explainability": explain,  # 없으면 None -> JSON null
        }

        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    # -----------------------------------------------------------
    # ctx 확인용: LLM 호출 없이 메시지/컨텍스트 프리뷰
    # -----------------------------------------------------------
    def preview_opinion_ctx(self):
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
                "next_close": float(tgt.next_close),
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
