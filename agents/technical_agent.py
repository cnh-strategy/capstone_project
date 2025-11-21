# agents/technical_agent.py

import os
import json
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset

from agents.base_agent import (
    BaseAgent, StockData, Target, Opinion, Rebuttal
)

from core.technical_classes.technical_data_set import (
    build_dataset as build_dataset_tech,
    load_dataset as load_dataset_tech,
)

from config.agents import agents_info, dir_info
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS

# ===============================================================
# 유틸리티 함수
# ===============================================================
def r4(x):
    """소수점 4자리 반올림"""
    try:
        return float(f"{float(x):.4f}")
    except:
        return x

class TechnicalAgent(BaseAgent, nn.Module):
    """
    TechnicalAgent: 기술적 분석 기반 주가 예측 에이전트
    
    주가 차트 데이터(가격, 거래량, 기술적 지표)를 분석하여
    주가 예측을 수행하는 에이전트입니다.
    
    주요 기능:
    - RSI, SMA 등 기술적 지표 계산
    - 2층 LSTM + Time-Attention 메커니즘
    - Attention 가중치를 활용한 시간 중요도 분석
    - Grad×Input 및 Occlusion을 통한 피처 중요도 분석
    - Monte Carlo Dropout을 통한 불확실성 추정
    - LLM을 활용한 Opinion, Rebuttal, Revision 생성
    
    Attributes:
        agent_id: 에이전트 식별자 (기본값: "TechnicalAgent")
        window_size: 시계열 윈도우 크기
        hidden_dims: LSTM 레이어별 hidden dimensions
        dropout: Dropout 비율
        input_dim: 입력 feature 차원
    """

    def __init__(self,
        agent_id="TechnicalAgent",
        input_dim=agents_info["TechnicalAgent"]["input_dim"],
        rnn_units1=agents_info["TechnicalAgent"]["rnn_units1"], # 1층 hidden (아연수정)
        rnn_units2=agents_info["TechnicalAgent"]["rnn_units2"], # 2층 hidden (아연수정)
        dropout=agents_info["TechnicalAgent"]["dropout"],
        data_dir=dir_info["data_dir"],
        window_size=agents_info["TechnicalAgent"]["window_size"],
        epochs=agents_info["TechnicalAgent"]["epochs"],
        learning_rate=agents_info["TechnicalAgent"]["learning_rate"],
        batch_size=agents_info["TechnicalAgent"]["batch_size"],
        **kwargs
    ):
        # 1) nn.Module 먼저 초기화
        nn.Module.__init__(self)

        # 2) BaseAgent 초기화
        BaseAgent.__init__(self, agent_id=agent_id, data_dir=data_dir, **kwargs)


        # 모델 하이퍼파라미터 설정 (아연수정)
        self.input_dim = int(input_dim)
        self.u1         = int(rnn_units1)
        self.u2         = int(rnn_units2)
        self.window_size= int(window_size)
        self.epochs     = int(epochs)
        self.lr         = float(learning_rate)
        self.batch_size = int(batch_size)

        # LSTMx2 + time attention (아연수정)
        self.lstm1 = nn.LSTM(self.input_dim, self.u1, batch_first=True)
        self.lstm2 = nn.LSTM(self.u1, self.u2, batch_first=True)
        self.attn_vec = nn.Parameter(torch.randn(self.u2))
        self.fc = nn.Linear(self.u2, 1)
        self.drop = nn.Dropout(float(dropout))


        # Optimizer / Loss 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) # 아연수정 lr
        # 기존: MSE Loss 사용
        # self.loss_fn = nn.MSELoss()
        # 수정: Huber Loss 사용 - 이상치에 덜 민감하고 더 안정적인 학습
        # delta=1.0으로 조정 (타겟 스케일링 후 적절한 값)
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.last_pred = None
        self.last_attn = None  # (아연수정) time-attention 캐시
        self._last_idea = None  # TechnicalAgent 전용 설명 정보 저장용

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM×2 + time-attention이 있으면 사용 (아연수정)
        """
        입력: x (B, T, F)
        출력: (B, 1)  — 다음날 수익률(학습 스케일)
        """
        h1, _ = self.lstm1(x)
        h1 = self.drop(h1)
        h2, _ = self.lstm2(h1)
        h2 = self.drop(h2)

        # tiem-attention: 각 시점 가중치
        w = torch.softmax(torch.matmul(h2, self.attn_vec), dim=1)  # [B,T]
        self._last_attn = w.detach()                               # 아연수정
        ctx = (h2 * w.unsqueeze(-1)).sum(dim=1)                    # [B,u2]
        return self.fc(ctx)                                        # [B,1]

    def _safe_names(self, feature_cols, F):
        """
        기능: 피처 이름 리스트를 모델 입력 차원(F)에 맞춰 보정.
        입력: feature_cols(list|None), F(int)
        출력: 길이 F의 피처명 리스트
        """
        cols = list(feature_cols) if feature_cols else []
        if len(cols) != F:
            cols = cols[:F] + [f"f{i}" for i in range(len(cols), F)]
        return cols

    def _safe_dates(self, dates, T):
        """
        기능: 날짜 리스트를 윈도우 길이(T)에 맞춰 보정.
        입력: dates(list|None), T(int)
        출력: 길이 T의 날짜 문자열 리스트
        """
        if not dates or len(dates) != T:
            return [f"t-{T-1-i}" for i in range(T)]
        return [str(d) for d in dates]

    def _scale_like_train(self, X_np):
        """
        기능: 학습 시 사용한 스케일러(self.scaler)로 입력 스케일 정합.
        입력: X_np(np.ndarray) (1,T,F)
        출력: 스케일된 np.ndarray (실패 시 원본 반환)
        """
        try:
            out = self.scaler.transform(X_np)
            if isinstance(out, tuple) and len(out) >= 1:
                return out[0]
            return out
        except Exception:
            return X_np

    @torch.no_grad()
    def time_importance_from_attention(self, X_last: torch.Tensor) -> np.ndarray:
        """
        기능: 모델의 time-attention 가중치로 시간 중요도 계산.
        입력: X_last(torch.Tensor) (1,T,F)
        출력: 정규화된 시간 중요도 np.ndarray (T,)
        주의: forward를 1회 호출해 _last_attn 생성.
        """
        self.eval()
        _ = self(X_last)
        attn = getattr(self, "_last_attn", None)  # [B,T]

        if attn is None:
            T = X_last.shape[1]
            return np.ones(T, dtype=float) / T
        w = attn[0].abs().cpu().numpy()
        # 1차원으로 변환 (T, 1) -> (T,)
        w = w.flatten() if w.ndim > 1 else w
        s = w.sum()
        result = w / s if s > 0 else np.ones_like(w) / len(w)
        # 반환값이 1차원인지 확인
        return result.flatten() if result.ndim > 1 else result

    def gradxinput_attrib(self, X_last: torch.Tensor, eps: float = 0.0):
        """
        기능: Grad×Input으로 (시간, 피처) 기여도 산출.
        입력: X_last(torch.Tensor):(1,T,F), eps(float): 입력 노이즈 안정화
        출력: (per_time(T,), per_feat(F,), gi(T,F)) 각 np.ndarray
        주의: eval 모드, y.sum()에 대해 역전파.
        """
        self.eval()
        x = X_last.clone().detach().to(next(self.parameters()).device)

        if eps > 0:
            x = x + eps * torch.randn_like(x)
        x.requires_grad_(True)
        y = self(x).sum()
        self.zero_grad(set_to_none=True)
        y.backward()
        gi = (x.grad * x).abs()[0].detach().cpu().numpy()  # (T,F)
        per_time = gi.sum(axis=1)
        per_feat = gi.mean(axis=0)
        return per_time, per_feat, gi

    @torch.no_grad()
    def occlusion_time(self, X_last: torch.Tensor, fill: str = "zero", batch: int = 32):
        """
        기능: 한 시점씩 가려 Δ예측으로 시간 중요도 계산.
        입력: X_last(1,T,F), fill: 'zero' 또는 평균치 대체, batch: 배치 크기
        출력: 정규화된 시간 중요도 np.ndarray (T,)
        복잡도: O(T) 전향 패스(배치 처리)
        """
        self.eval()
        base = float(self(X_last).item())
        _, T, F = X_last.shape
        Xs = []
        for t in range(T):
            x = X_last.clone()
            if fill == "zero":
                x[:, t, :] = 0
            else:
                x[:, t, :] = X_last.mean(dim=1, keepdim=True)[:, 0, :]
            Xs.append(x)
        deltas = []
        for i in range(0, T, batch):
            xb = torch.cat(Xs[i:i+batch], dim=0)
            yb = self(xb).flatten().cpu().numpy()
            deltas.extend(np.abs(yb - base).tolist())
        s = sum(deltas)
        return np.array([v/s if s > 0 else 1.0/T for v in deltas], dtype=float)

    @torch.no_grad()
    def occlusion_feature(self, X_last: torch.Tensor, fill: str = "zero", batch: int = 32):
        """
        기능: 한 피처씩 가려 Δ예측으로 피처 중요도 계산.
        입력: X_last(1,T,F), fill: 'zero' 또는 평균치 대체, batch: 배치 크기
        출력: 정규화된 피처 중요도 np.ndarray (F,)
        복잡도: O(F) 전향 패스(배치 처리)
        """
        self.eval()
        base = float(self(X_last).item())
        _, T, F = X_last.shape
        Xs = []
        for f in range(F):
            x = X_last.clone()
            if fill == "zero":
                x[:, :, f] = 0
            else:
                x[:, :, f] = X_last.mean(dim=(1, 2), keepdim=True)[:, 0, 0]
            Xs.append(x)
        deltas = []
        for i in range(0, F, batch):
            xb = torch.cat(Xs[i:i+batch], dim=0)
            yb = self(xb).flatten().cpu().numpy()
            deltas.extend(np.abs(yb - base).tolist())
        s = sum(deltas)
        return np.array([v/s if s > 0 else 1.0/F for v in deltas], dtype=float)

    def explain_last(
        self,
        X_last: torch.Tensor,
        dates: list | None = None,
        top_k: int = 5,
        use_shap: bool = True, # 기본은 빠르게 off, 필요시 true
        shap_weight_time: float = 0.20,      # 시간 중요도에서 SHAP 가중치(임의설정)
        shap_weight_feat: float = 0.30       # 피처 중요도에서 SHAP 가중치(임의설정)
        ):
        """
        기능: Attention + Grad×Input + Occlusion 융합으로 최신 윈도우 설명 패킷 생성.
        입력: X_last(1,T,F), dates(list|None), top_k(int)
        처리:
        1) 학습 스케일 정합 → 텐서화
        2) time-attention, Grad×Input, Occlusion 계산
        3) 정규화·가중 평균으로 per_time/per_feature 산출
        4) 날짜별 상위 피처(top_k)와 증거 벡터(evidence) 포함
        출력: dict
        - per_time: [{date,sum_abs}]
        - per_feature: [{feature,sum_abs}]
        - time_attention: {date:weight}
        - time_feature: {date:{feat:score}}
        - evidence: 원천 지표들(attention, gradxinput, occlusion)
        - raw: Grad×Input 원시(T×F)
        """
        # 스케일 정합
        device = next(self.parameters()).device
        X_np = X_last.detach().cpu().numpy()
        X_scaled = self._scale_like_train(X_np)
        Xs = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        T, F = Xs.shape[1], Xs.shape[2]
        feat_cols_src = getattr(self.stockdata, "feature_cols", [])
        feat_names = self._safe_names(feat_cols_src, F)
        if dates is None:
            dates = getattr(self.stockdata, f"{self.agent_id}_dates", [])
        dates = self._safe_dates(dates, T)

        # 시간 중요도
        time_attn = self.time_importance_from_attention(Xs)  # (T,)

        # GradxInput
        g_time, g_feat, gi_raw = self.gradxinput_attrib(Xs, eps=0.0) # (T,), (F,)

        # Occlusion
        occ_time = self.occlusion_time(Xs, fill="zero", batch=32) # (T,)
        occ_feat = self.occlusion_feature(Xs, fill="zero", batch=32) # (F,)

        # 정규화 및 융합
        g_time_n = g_time / (g_time.sum() + 1e-12)
        g_feat_n = g_feat / (g_feat.sum() + 1e-12)
        occ_feat_n = occ_feat / (occ_feat.sum() + 1e-12)

        # (옵션) SHAP 추가
        shap_time = None
        shap_feat = None
        shap_used = False
        if use_shap:
            try:
                shap_res = self.shap_last(Xs, background_k=64)  # 아래에 정의
                shap_time = shap_res["per_time"]    # (T,) 합=1
                shap_feat = shap_res["per_feature"] # (F,) 합=1
                shap_used = True
            except Exception as _:
                shap_time = None
                shap_feat = None
                shap_used = False  # 실패 시 무시하고 기본 3요소로 진행

        # 융합 가중치(간단 휴리스틱, 필요시 검증셋에서 학습 가능)
        if shap_time is not None and shap_feat is not None:
            # 시간 중요도: attn 0.4, GI 0.25, occ 0.15, shap (인자) → 합 1로 재정규화
            w_time = np.array([0.4, 0.25, 0.15, float(shap_weight_time)], dtype=float)
            w_time = w_time / w_time.sum()
            per_time = (
                w_time[0]*time_attn +
                w_time[1]*g_time_n +
                w_time[2]*occ_time +
                w_time[3]*shap_time
            )
            # 피처 중요도: GI 0.5, occ 0.2, shap (인자) → 합 1로 재정규화
            w_feat = np.array([0.5, 0.2, float(shap_weight_feat)], dtype=float)
            w_feat = w_feat / w_feat.sum()
            per_feat = (
                w_feat[0]*g_feat_n +
                w_feat[1]*occ_feat_n +
                w_feat[2]*shap_feat
            )
        else:
            # SHAP 미사용/실패 시 기존 고정 비율
            per_time = 0.5 * time_attn + 0.3 * g_time_n + 0.2 * occ_time
            per_feat = 0.7 * g_feat_n   + 0.3 * occ_feat_n

        # per_time과 per_feat이 1차원인지 확인하고 변환
        per_time = per_time.flatten() if per_time.ndim > 1 else per_time
        per_feat = per_feat.flatten() if per_feat.ndim > 1 else per_feat

        # 날짜별 상위 피처(Grad×Input 기준으로 간단)
        gi_abs = np.abs(gi_raw)
        time_feature = {}
        for t_idx, d in enumerate(dates):
            pairs = sorted(
                zip(feat_names, gi_abs[t_idx].tolist()),
                key=lambda z: z[1], reverse=True
            )[:top_k]
            time_feature[str(d)] = {k: float(v) for k, v in pairs}

        # 결과 패킷
        time_attention = {str(d): r4(w) for d, w in zip(dates, time_attn.tolist())}
        per_time_list = [{"date": str(d), "sum_abs": r4(v)} for d, v in zip(dates, per_time.tolist())]
        per_feat_list = [{"feature": k, "sum_abs": r4(v)} for k, v in sorted(zip(feat_names, per_feat.tolist()),
                                                                          key=lambda z: z[1], reverse=True)]

        evidence = {
            "attention": [r4(x) for x in time_attn.tolist()],
            "gradxinput_feat": [r4(x) for x in g_feat.tolist()],
            "occlusion_time": [r4(x) for x in occ_time.tolist()],
            "window_size": int(T),
            "shap_used": bool(shap_used)
            }

        return {
            "per_time": per_time_list,
            "per_feature": per_feat_list,
            "time_attention": time_attention,
            "time_feature": time_feature,
            "evidence": evidence,
            "raw": {"gradxinput": gi_abs.tolist()}  # 원시값은 비라운딩 유지 가능
          }

    # ---------------- SHAP 보조: 배경 샘플 추출 ----------------
    def _background_windows(self, k: int = 64):
        """
        학습/검증 구간에서 윈도우 k개를 균등 간격으로 뽑아 배경으로 사용.
        파일 상단 수정 없이 내부에서 lazy import.
        """
        try:
            from core.technical_classes.technical_data_set import load_dataset  
            X, _, _, _ = load_dataset(self.ticker, agent_id=self.agent_id, save_dir=self.data_dir)
            if len(X) <= 1:
                return None
            k = min(int(k), len(X) - 1)
            idx = np.linspace(0, len(X) - 2, num=k, dtype=int)
            X_bg = X[idx]
            X_bg_scaled, _ = self.scaler.transform(X_bg)
            dev = next(self.parameters()).device
            return torch.tensor(X_bg_scaled, dtype=torch.float32, device=dev)
        except Exception:
            return None

    # ---------------- SHAP 계산(GradientExplainer) ----------------
    #@torch.no_grad()
    def shap_last(self, X_last: torch.Tensor, background_k: int = 64):
        """
        GradientExplainer로 SHAP 값을 1개 윈도우(1,T,F)에 대해 계산.
        반환: {"per_time": (T,), "per_feature": (F,)}
        """
        try:
            import shap  # lazy import (상단 수정 불필요)
        except Exception as e:
            raise RuntimeError("shap 미설치 또는 로드 실패: pip install shap==0.45.0") from e

        self.eval()

        # 배경 구성(없으면 현재 입력 복제)
        X_bg = self._background_windows(k=background_k)
        if X_bg is None:
            X_bg = X_last.repeat(32, 1, 1)

        # 입력 스케일 맞추기
        X_np = X_last.detach().cpu().numpy()
        X_scaled = self._scale_like_train(X_np)
        X_in = torch.tensor(X_scaled, dtype=torch.float32, device=next(self.parameters()).device)
        X_in.requires_grad_(True)  # 추가(shap 용)

        # PyTorch 모델 직접 전달
        explainer = shap.GradientExplainer(self, X_bg)
        sv = explainer.shap_values(X_in)  # np.ndarray 또는 list

        if isinstance(sv, list):
            sv = sv[0]
        if sv.ndim == 2:  # (T,F) → (1,T,F) 호환
            sv = sv[None, ...]

        sv_abs = np.abs(sv)          # (1,T,F)
        per_time = sv_abs.sum(axis=2)[0]      # (T,)
        per_feat = sv_abs.mean(axis=1)[0]     # (F,)

        # 정규화(합=1)
        per_time = per_time / (per_time.sum() + 1e-12)
        per_feat = per_feat / (per_feat.sum() + 1e-12)
        return {"per_time": per_time, "per_feature": per_feat}




    # -----------------------------------------------------------
    # 아이디어 압축(LLM 토큰 절약용)
    # -----------------------------------------------------------
    @staticmethod
    def _pack_idea(exp: dict, top_time=8, top_feat=6, coverage=0.8):
        """상위 시간, 피처만 압축, 커버리지 80%까지 누적"""
        per_time = sorted(exp["per_time"], key=lambda z: z["sum_abs"], reverse=True)
        total = sum(z["sum_abs"] for z in per_time) or 1.0
        acc, picked = 0.0, []
        for z in per_time:
            acc += z["sum_abs"]
            picked.append({"date": z["date"], "weight": r4(z["sum_abs"]/total)})
            if acc/total >= coverage or len(picked) >= top_time:
                break

        per_feat = sorted(exp["per_feature"], key=lambda z: z["sum_abs"], reverse=True)[:top_feat]
        top_features = [{"feature": f["feature"], "weight": r4(f["sum_abs"])} for f in per_feat]
        peak = picked[0]["date"] if picked else None
        return {
            "top_time": picked,
            "top_features": top_features,
            "peak_date": peak,
            "window_size": exp.get("evidence",{}).get("window_size")}


    # -----------------------------------------------------------
    # ctx 생성용 헬퍼 블록들
    # -----------------------------------------------------------

    # LLM Reasoning 메시지 (아연수정)

    def _build_messages_opinion(self, stock_data, target):
        """TechnicalAgent용 LLM 프롬프트 메시지 구성 + 설명값 포함"""
        last = float(getattr(stock_data, "last_price", target.next_close))

        # 최신 윈도우 설명 산출
        X_last = self.searcher(self.ticker)
        if not isinstance(X_last, torch.Tensor):
          X_last = torch.tensor(X_last, dtype=torch.float32)
        T = X_last.shape[1]
        # dates 수정
        dates = getattr(self.stockdata, f"{self.agent_id}_dates", [])

        exp = self.explain_last(X_last, dates, top_k=5, use_shap=True)
        idea = self._pack_idea(exp)  # 항상 새로 계산
        self._last_idea = idea  # 인스턴스 변수로 저장 (TechnicalAgent 전용)

        # 기본 컨텍스트
        ctx = {
            "ticker": getattr(stock_data, "ticker", "Unknown"),
            "last_price": r4(last),
            "next_close": r4(target.next_close),
            "uncertainty": r4(target.uncertainty),
            "confidence": r4(target.confidence),
            "sigma": r4(target.uncertainty or 0.0),
            "beta": r4(target.confidence or 0.0),
            "window_size": int(self.window_size),
            "idea": idea,  # 핵심만
            # "evidence": exp.get("evidence", {})
        }

        system_text = OPINION_PROMPTS[self.agent_id]["system"]
        tmpl = OPINION_PROMPTS[self.agent_id]["user"]
        user_text = tmpl.replace("{context}", json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text


    def _build_messages_rebuttal(self,
                                my_opinion: Opinion,
                                target_opinion: Opinion,
                                stock_data: StockData) -> tuple[str, str]:

        t = stock_data.ticker or "UNKNOWN"
        ccy = (stock_data.currency or "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        ctx = {
            "ticker": t,
            "currency": ccy,
            "data_summary": getattr(stock_data, "feature_cols", []), # 수정
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:2000],
                "uncertainty": float(my_opinion.target.uncertainty),
                "confidence": float(my_opinion.target.confidence),
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
                "uncertainty": float(target_opinion.target.uncertainty),
                "confidence": float(target_opinion.target.confidence),
            }
        }
        # 각 컬럼별 최근 시계열 그대로 포함
    
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-self.window_size:] # 수정
            else:
                ctx[col] = [values]

        # 아연 수정
        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        tmpl = REBUTTAL_PROMPTS[self.agent_id]["user"]
        user_text = tmpl.replace("{context}", json.dumps(ctx, ensure_ascii=False))
    
        return system_text, user_text


    def _build_messages_revision(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: Optional[List[Rebuttal]] = None,
        stock_data: StockData = None,
    ) -> tuple[str, str]:
        """
        Revision용 LLM 메시지 생성기
        - 내 의견(my_opinion), 타 에이전트 의견(others), 주가데이터(stock_data) 기반
        - rebuttals 중 나(self.agent_id)를 대상으로 한 내용만 포함
        """
        # 기본 메타데이터
        t = getattr(stock_data, "ticker", "UNKNOWN")
        ccy = getattr(stock_data, "currency", "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        # 타 에이전트 의견 및 rebuttal 통합 요약
        others_summary = []
        for o in others:
            entry = {
                "agent_id": o.agent_id,
                "predicted_price": float(o.target.next_close),
                "confidence": float(o.target.confidence),
                "uncertainty": float(o.target.uncertainty),
                "reason": str(o.reason)[:500],
            }

            # 나에게 온 rebuttal만 stance/message 추출
            if rebuttals:
                related_rebuts = [
                    {"stance": r.stance, "message": r.message}
                    for r in rebuttals
                    if r.from_agent_id == o.agent_id and r.to_agent_id == self.agent_id
                ]
                if related_rebuts:
                    entry["rebuttals_to_me"] = related_rebuts

            others_summary.append(entry)
            

        # Context 구성
        ctx = {
            "ticker": t,
            "currency": ccy,
            "agent_type": self.agent_id,
            "my_opinion": {
                "predicted_price": float(my_opinion.target.next_close),
                "confidence": float(my_opinion.target.confidence),
                "uncertainty": float(my_opinion.target.uncertainty),
                "reason": str(my_opinion.reason)[:1000],
            },
            "others_summary": others_summary,
            "data_summary": getattr(stock_data, "feature_cols", []), # 수정
        }

        # 최근 시계열 데이터 포함 (기술/심리적 패턴)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-14:]  # 최근 14일치
            else:
                ctx[col] = [values]

        # Prompt 구성
        prompt_set = REVISION_PROMPTS.get(self.agent_id)
        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

        return system_text, user_text

    # ===============================================================
    # TechnicalAgent 전용 메서드들 (TechnicalBaseAgent에서 이동)
    # ===============================================================

    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """TechnicalAgent 전용 searcher - technical_data_set 사용"""
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        self.ticker = ticker
        
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")
        cfg = agents_info.get(self.agent_id, {}) 

        need_build = rebuild or (not os.path.exists(dataset_path))
        if need_build:
            print(f"⚙️ {ticker} {agent_id} dataset not found. Building new dataset..." if not os.path.exists(dataset_path) else f"⚙️ {ticker} {agent_id} rebuild requested. Building dataset...")
            build_dataset_tech(
                ticker=ticker,
                save_dir=self.data_dir,
                period=cfg.get("period", "5y"),
                interval=cfg.get("interval", "1d"),
            )
    
        # CSV 로드
        X, y, feature_cols, dates_all = load_dataset_tech(
            ticker, agent_id=agent_id, save_dir=self.data_dir
            )

        # 최근 window
        X_latest = X[-1:]

        # StockData 구성
        self.stockdata = StockData(ticker=ticker)
        self.stockdata.feature_cols = feature_cols
        
        # dates_all 구조에 따라 "마지막 윈도우" 날짜만 추출
        if dates_all:
            if isinstance(dates_all[0], (list, tuple)):
                # dates_all: [ [윈도우1의 T개 날짜], [윈도우2의 T개 날짜], ... ]
                last_dates = dates_all[-1]
            else:
                # 만약 1차원 리스트라면, 뒤에서 window_size만큼 사용
                win = int(self.window_size)
                last_dates = dates_all[-win:]
        else:
            last_dates = []
        
        
        # 전체는 *_dates_all로, 마지막 윈도우는 *_dates로 저장
        setattr(self.stockdata, f"{agent_id}_dates_all", dates_all or [])
        setattr(self.stockdata, f"{agent_id}_dates",     last_dates or [])
        
        # last_price 안전 변환
        try:
            data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
            if data is not None and not data.empty:
                last_val = data["Close"].iloc[-1]
                self.stockdata.last_price = float(last_val.item() if hasattr(last_val, "item") else last_val)
            else:
                self.stockdata.last_price = None
        except Exception:
            self.stockdata.last_price = None

        # 통화코드
        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception:
            self.stockdata.currency = "USD"

        df_latest = pd.DataFrame(X_latest[0], columns=feature_cols)  # (T, F)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)

        # StockData 생성 완료 (로그는 DebateAgent에서 처리)

        return torch.tensor(X_latest, dtype=torch.float32)

    def pretrain(self):
        """Agent별 사전학습 루틴 (모델 생성, 학습, 저장, self.model 연결까지 포함)"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]

        # 데이터 로드
        X, y, cols, _ = load_dataset_tech(self.ticker, self.agent_id, save_dir=self.data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 타깃 스케일 조정 - 상승/하락율을 100배로 스케일링
        y_train *= 100.0
        y_val   *= 100.0

        self.scaler.fit_scalers(X_train, y_train)
        self.scaler.save(self.ticker)

        X_train, y_train = map(torch.tensor, self.scaler.transform(X_train, y_train))
        X_train, y_train = X_train.float(), y_train.float()

        # 모델 = self (nn.Module)
        model = self
        # 혹시 예전에 잘못 등록된 submodule "model"이 있으면 제거
        self._modules.pop("model", None)

        # 학습
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.HuberLoss(delta=1.0)

        train_loader = DataLoader(TensorDataset(X_train, y_train.view(-1, 1)),
                                  batch_size=batch_size, shuffle=True)

        # 학습 루프
        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")

        # 모델 저장 및 연결
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        print(f" {self.agent_id} 모델 학습 및 저장 완료: {model_path}")

    def predict(self, X, n_samples: int = 30, current_price: float = None, X_last: np.ndarray = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)
        """
        # 1) 모델 및 스케일러 준비
        model = self  # TechnicalAgent 자체가 nn.Module
        self.scaler.load(self.ticker)

        # 2) 입력 변환 + 학습과 동일 스케일로 변환
        if isinstance(X, np.ndarray):
            X_raw_np = X.copy()
        elif isinstance(X, torch.Tensor):
            X_raw_np = X.detach().cpu().numpy().copy()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        X_scaled, _ = self.scaler.transform(X_raw_np)
        device = next(model.parameters()).device
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        # 3) Monte Carlo Dropout 추론
        model.train()  # dropout 활성화
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor).cpu().numpy().flatten()
                preds.append(y_pred)

        preds = np.stack(preds)              # (n_samples, seq_len or 1)
        mean_pred = preds.mean(axis=0)       # (seq_len,)
        std_pred = np.abs(preds.std(axis=0)) # 항상 양수

        # 4) σ 기반 confidence 계산
        sigma = float(std_pred[-1])
        sigma = max(sigma, 1e-6)
        confidence = 1 / (1 + np.log1p(sigma))

        # 5) 타깃 역스케일링 및 가격 변환
        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        # current_price 결정
        if current_price is None:
            last_price = getattr(getattr(self, "stockdata", None), "last_price", None)
            current_price = 100.0 if last_price is None else last_price

        # 학습 타깃은 "다음날 수익률(%)"이므로 100으로 나눠서 사용
        predicted_return = float(mean_pred[-1]) / 100.0
        predicted_price = current_price * (1 + predicted_return)

        # 6) Target 생성
        target = Target(
            next_close=float(predicted_price),
            uncertainty=sigma,
            confidence=float(confidence),
        )
        return target

    def reviewer_draft(self, stock_data: StockData = None, target: Target = None) -> Opinion:
        """(1) searcher → (2) predicter → (3) LLM(JSON Schema)로 reason 생성 → Opinion 반환"""

        # 1) 데이터 수집
        if stock_data is not None:
            self.stockdata = stock_data
        else:
        # 내부에 없으면 searcher 한 번 돌려서 만든다
            if getattr(self, "stockdata", None) is None:
                if not self.ticker:
                    raise RuntimeError(
                        f"[{self.agent_id}] ticker가 설정되지 않았습니다. "
                        "reviewer_draft 호출 전에 ticker를 지정하거나 searcher(ticker)를 먼저 호출하세요."
                    )
                _ = self.searcher(self.ticker)  # self.stockdata 세팅
            stock_data = self.stockdata

        # 2) 예측값 생성
        if target is None:
            # searcher는 위에서 한 번 돌았으므로, 여기서는 최신 윈도우로 predict만 수행
            X_input = self.searcher(self.ticker)              # (1,T,F)
            target = self.predict(X_input)

        # 3) LLM 호출(reason 생성) - 전달받은 stock_data 사용
        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object", 
                "properties": {"reason": {"type": "string"}}, 
                "required": ["reason"], 
                "additionalProperties": False}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")

        # 4) Opinion 기록/반환 (항상 최신 값 append)
        self.opinions.append(Opinion(
                    agent_id=self.agent_id, 
                    target=target, 
                    reason=reason))

        # 최신 오피니언 반환
        return self.opinions[-1]

    def reviewer_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """LLM을 통해 상대 의견에 대한 반박/지지 생성"""

        # 메시지 생성 (context 구성은 별도 헬퍼에서)
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )

        # LLM 호출
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"}
                },
                "required": ["stance", "message"],
                "additionalProperties": False
            }
        )

        # 결과 정리 및 기록
        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=parsed.get("stance", "REBUT"),
            message=parsed.get("message", "(반박/지지 사유 생성 실패)")
        )

        # 저장
        self.rebuttals[round].append(result)

        # 디버깅 로그
        if self.verbose:
            print(
                f"[{self.agent_id}] rebuttal 생성 → {result.stance} "
                f"({my_opinion.agent_id} → {other_opinion.agent_id})"
            )

        return result
    
    # DebateAgent.get_rebuttal() 호환용 래퍼
    def reviewer_rebuttal(
        self,
        my_opinion: Opinion,
        other_opinion: Opinion,
        round_index: int,
    ) -> Rebuttal:
        return self.reviewer_rebut(
            my_opinion=my_opinion,
            other_opinion=other_opinion,
            round=round_index,
        )

    def reviewer_revise(
        self,
        my_opinion: Opinion,
        others: List[Opinion],
        rebuttals: List[Rebuttal],
        stock_data: StockData,
        fine_tune: bool = True,
        lr: float = 1e-4,
        epochs: int = 20,
    ):
        """
        Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - fine-tuning (수익률 단위)
        - reasoning 생성
        """
        gamma = getattr(self, "gamma", 0.3)               # 수렴율 (0~1)
        delta_limit = getattr(self, "delta_limit", 0.05)  # fine-tuning 보정 한계
        current_price = getattr(self.stockdata, "last_price", 100.0)  # 수정: 항상 초반에 현재가 확보

        try:
            # β 계산 (불확실성 작을수록 신뢰 높음)
            my_price = float(my_opinion.target.next_close)           # 수정: float 캐스팅
            my_sigma = abs(my_opinion.target.uncertainty or 1e-6)

            # 수정: others가 없을 때 방어
            if len(others) == 0:
                revised_price = my_price
            else:
                other_prices = np.array([o.target.next_close for o in others], dtype=float)
                other_sigmas = np.array([abs(o.target.uncertainty or 1e-6) for o in others], dtype=float)

                all_sigmas = np.concatenate([[my_sigma], other_sigmas])

                inv_sigmas = 1 / (all_sigmas + 1e-6)
                betas = inv_sigmas / inv_sigmas.sum()

                # 논문식 수렴 업데이트
                # y_i_rev = y_i + γ Σ β_j (y_j - y_i)
                delta = np.sum(betas[1:] * (other_prices - my_price))
                revised_price = my_price + gamma * delta

        except Exception as e:
            print(f"[{self.agent_id}] revised_target 계산 실패: {e}")
            revised_price = my_opinion.target.next_close  # 수정: 여기서는 가격만 되돌림

        # 수정: 항상 delta_limit로 클램프 (try/except 밖에서 공통 적용)
        price_uplimit = current_price * (1 + delta_limit)
        price_downlimit = current_price * (1 - delta_limit)
        revised_price = float(min(max(revised_price, price_downlimit), price_uplimit))

        # Fine-tuning (return 단위)
        loss_value = None
        if fine_tune:
            try:
                revised_return = (revised_price / current_price) - 1.0   # 예: 0.012
                revised_return_scaled = revised_return * 100.0           # 예: 1.2

                # 스케일러 기준에 맞추어 타깃 변환
                if getattr(self.scaler, "y_scaler", None) is not None:
                    y_target_scaled = self.scaler.y_scaler.transform(
                        np.array([[revised_return_scaled]], dtype=float)
                    )[0, 0]
                else:
                    y_target_scaled = revised_return_scaled

                # 최신 입력
                X_input = self.searcher(self.ticker)  # (1, T, F)

                # TechnicalAgent(nn.Module) → self 자체 사용
                model = self
                device = next(model.parameters()).device

                if isinstance(X_input, torch.Tensor):
                    X_tensor = X_input.to(device).float()
                else:
                    X_tensor = torch.tensor(X_input, dtype=torch.float32, device=device)

                y_tensor = torch.tensor([[y_target_scaled]], dtype=torch.float32, device=device)

                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # pretrain과 통일: HuberLoss
                criterion = torch.nn.HuberLoss(delta=1.0)

                for _ in range(epochs):
                    optimizer.zero_grad()
                    pred = model(X_tensor)
                    loss = criterion(pred, y_tensor)
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
                print(f"[{self.agent_id}] fine-tuning 완료: loss={loss_value:.6f}")

            except Exception as e:
                print(f"[{self.agent_id}] fine-tuning 실패: {e}")

        # fine-tuning 이후 새 예측 생성
        try:
            X_latest = self.searcher(self.ticker)
            new_target = self.predict(X_latest)
        except Exception as e:
            print(f"[{self.agent_id}] predict 실패: {e}")
            new_target = my_opinion.target

        # reasoning 생성
        try:
            sys_text, user_text = self._build_messages_revision(
                my_opinion=my_opinion,
                others=others,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )
        except Exception as e:
            print(f"[{self.agent_id}] _build_messages_revision 실패: {e}")
            sys_text, user_text = (
                "너는 금융 분석가다. 간단히 reason만 생성하라.",
                json.dumps({"reason": "기본 메시지 생성 실패"}),
            )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        )

        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=new_target,
            reason=revised_reason,
        )

        self.opinions.append(revised_opinion)
        print(f"[{self.agent_id}] revise 완료 → new_close={new_target.next_close:.2f}, loss={loss_value}")
        return self.opinions[-1]

    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치 로드 (객체/딕셔너리/state_dict 자동 인식 + model 자동 생성)"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            return False

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

            # 혹시 예전에 잘못 등록된 서브모듈 "model"이 있으면 제거
            # (self를 서브모듈로 넣어버린 과거 코드 대비용)
            self._modules.pop("model", None)

            # 다양한 저장 포맷 처리
            if isinstance(checkpoint, torch.nn.Module):
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
            else:
                print(f"[{self.agent_id}] 알 수 없는 체크포맷: {type(checkpoint)}")
                return False

            # 바로 self에 로드
            self.load_state_dict(state_dict)
            self.eval()

            # self.model 에 self를 넣으면 순환 참조(submodule 등록)라서 넣지 않는 게 안전
            # (TechnicalAgent.predict / pretrain 은 model = self 로 동작하므로 별도 self.model 필요 없음)

            return True

        except Exception as e:
            print(f"[{self.agent_id}] load_model 실패: {e}")
            return False

    def evaluate(self, ticker: str = None):
        """검증 데이터로 성능 평가"""
        if ticker is None:
            ticker = self.ticker

        # 1) 데이터 로드
        X, y, feature_cols, _ = load_dataset_tech(
            ticker,
            agent_id=self.agent_id,
            save_dir=self.data_dir
        )

        # 2) 시계열 분할 (80% 훈련, 20% 검증)
        split_idx = int(len(X) * 0.8)
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        # 3) 스케일러 로드 + y 스케일(학습과 동일하게 ×100)  # 수정
        self.scaler.load(ticker)

        # 🔧 수정: y를 1D로 맞춰줍니다.
        y_val_scaled = (y_val * 100.0).reshape(-1)

        X_val_scaled, y_val_scaled = self.scaler.transform(
            X_val,
            y_val_scaled
        )

        # 🔧 수정: transform 결과도 확실히 1D로 정리
        y_val_scaled = np.asarray(y_val_scaled).reshape(-1)

        # 4) 모델 가중치 로드 (없으면 pretrain)                # 수정
        model_path = os.path.join(self.model_dir, f"{ticker}_{self.agent_id}.pt")
        if not self.load_model(model_path):                      # 수정
            self.pretrain()
            self.load_model(model_path)

        model = self
        model.eval()                                             # 수정

        # 5) 검증 데이터 예측
        predictions = []
        actual_returns = []

        with torch.no_grad():                                    # 수정
            for i in range(len(X_val_scaled)):
                X_input = X_val_scaled[i:i+1]   # (1, T, F)
                X_tensor = torch.tensor(X_input, dtype=torch.float32)

                pred_scaled = model(X_tensor).item()             # 예측값 (스케일된 y)
                predictions.append(pred_scaled)
                actual_returns.append(float(y_val_scaled[i]))       # 스케일된 타깃

        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)

        # 6) 성능 지표 계산 (스케일된 수익률 기준)              # 수정
        mae = np.mean(np.abs(predictions - actual_returns))
        rmse = np.sqrt(np.mean((predictions - actual_returns) ** 2))

        # 상관계수 (분산 0 방지)                                 # 수정
        if np.std(predictions) == 0 or np.std(actual_returns) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(predictions, actual_returns)[0, 1])

        # 7) 방향 정확도 (부호 기준 → 상승/하락 일치율)
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual_returns)
        direction_accuracy = float(np.mean(pred_direction == actual_direction) * 100.0)

        return {
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "direction_accuracy": direction_accuracy,
            "n_samples": len(predictions),
        }
