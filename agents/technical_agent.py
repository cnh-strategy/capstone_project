# agents/technical_agent.py

import os
import json
from typing import List, Optional, Union
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
    load_dataset as load_dataset_tech,
)
import importlib
import warnings
warnings.filterwarnings('ignore')

from config.agents import agents_info, dir_info, common_params
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
        # Huber Loss 사용 - 이상치에 덜 민감하고 더 안정적인 학습
        # config에서 delta 값 가져오기
        huber_delta = common_params.get("huber_loss_delta", 1.0)
        self.loss_fn = nn.HuberLoss(delta=huber_delta)
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
    def occlusion_time(self, X_last: torch.Tensor, fill: str = "zero", batch: Optional[int] = None):
        """
        기능: 한 시점씩 가려 Δ예측으로 시간 중요도 계산.
        입력: X_last(1,T,F), fill: 'zero' 또는 평균치 대체, batch: 배치 크기
        출력: 정규화된 시간 중요도 np.ndarray (T,)
        복잡도: O(T) 전향 패스(배치 처리)
        """
        if batch is None:
            batch = agents_info.get(self.agent_id, {}).get("occlusion_batch_size", 32)
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
    def occlusion_feature(self, X_last: torch.Tensor, fill: str = "zero", batch: Optional[int] = None):
        """
        기능: 한 피처씩 가려 Δ예측으로 피처 중요도 계산.
        입력: X_last(1,T,F), fill: 'zero' 또는 평균치 대체, batch: 배치 크기
        출력: 정규화된 피처 중요도 np.ndarray (F,)
        복잡도: O(F) 전향 패스(배치 처리)
        """
        if batch is None:
            batch = agents_info.get(self.agent_id, {}).get("occlusion_batch_size", 32)
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
        top_k: Optional[int] = None,
        use_shap: bool = True, # 기본은 빠르게 off, 필요시 true
        shap_weight_time: Optional[float] = None,      # 시간 중요도에서 SHAP 가중치
        shap_weight_feat: Optional[float] = None       # 피처 중요도에서 SHAP 가중치
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
        # config에서 파라미터 가져오기
        cfg = agents_info.get(self.agent_id, {})
        if top_k is None:
            top_k = cfg.get("top_k_features", 5)
        if shap_weight_time is None:
            shap_weight_time = cfg.get("shap_weight_time", 0.20)
        if shap_weight_feat is None:
            shap_weight_feat = cfg.get("shap_weight_feat", 0.30)
        
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

        # Occlusion (config에서 batch_size 가져오기)
        occlusion_batch = cfg.get("occlusion_batch_size", 32)
        occ_time = self.occlusion_time(Xs, fill="zero", batch=occlusion_batch) # (T,)
        occ_feat = self.occlusion_feature(Xs, fill="zero", batch=occlusion_batch) # (F,)

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

        # 융합 가중치 (config에서 가져오기)
        attention_weights = cfg.get("attention_weights", [0.4, 0.25, 0.15])
        feature_weights = cfg.get("feature_weights", [0.5, 0.2])
        
        if shap_time is not None and shap_feat is not None:
            # 시간 중요도: attn, GI, occ, shap → 합 1로 재정규화
            w_time = np.array([attention_weights[0], attention_weights[1], attention_weights[2], float(shap_weight_time)], dtype=float)
            w_time = w_time / w_time.sum()
            per_time = (
                w_time[0]*time_attn +
                w_time[1]*g_time_n +
                w_time[2]*occ_time +
                w_time[3]*shap_time
            )
            # 피처 중요도: GI, occ, shap → 합 1로 재정규화
            w_feat = np.array([feature_weights[0], feature_weights[1], float(shap_weight_feat)], dtype=float)
            w_feat = w_feat / w_feat.sum()
            per_feat = (
                w_feat[0]*g_feat_n +
                w_feat[1]*occ_feat_n +
                w_feat[2]*shap_feat
            )
        else:
            # SHAP 미사용/실패 시 config에서 가져온 가중치 사용
            per_time = attention_weights[0] * time_attn + attention_weights[1] * g_time_n + attention_weights[2] * occ_time
            per_feat = feature_weights[0] * g_feat_n + feature_weights[1] * occ_feat_n

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
    def _pack_idea(exp: dict, top_time: Optional[int] = None, top_feat: Optional[int] = None, coverage: Optional[float] = None):
        """상위 시간, 피처만 압축, 커버리지까지 누적"""
        # config에서 파라미터 가져오기 (agent_id는 클래스 메서드이므로 exp에서 추론)
        agent_id = exp.get("evidence", {}).get("agent_id", "TechnicalAgent")
        cfg = agents_info.get(agent_id, {})
        if top_time is None:
            top_time = cfg.get("pack_idea_top_time", 8)
        if top_feat is None:
            top_feat = cfg.get("pack_idea_top_feat", 6)
        if coverage is None:
            coverage = cfg.get("pack_idea_coverage", 0.8)
        
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

        # stockdata에서 이미 저장된 데이터 재사용 (중복 searcher 방지)
        agent_data = getattr(stock_data, self.agent_id, {})
        
        if isinstance(agent_data, dict) and agent_data:
            # DataFrame으로 복원
            df = pd.DataFrame(agent_data)
            X_last = torch.tensor(
                df.tail(self.window_size).values, 
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, F)
        else:
            # 만약 stockdata가 비어있으면 searcher() 재호출
            print(f"[WARN] {self.agent_id} stockdata가 비어있음, searcher 재호출")
            X_last = self.searcher(self.ticker)
            if not isinstance(X_last, torch.Tensor):
                X_last = torch.tensor(X_last, dtype=torch.float32)
        
        T = X_last.shape[1]
        # dates 수정
        dates = getattr(self.stockdata, f"{self.agent_id}_dates", [])

        # config에서 top_k 가져오기
        cfg = agents_info.get(self.agent_id, {})
        top_k = cfg.get("top_k_features", 5)
        exp = self.explain_last(X_last, dates, top_k=top_k, use_shap=True)
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

    def _fetch_ticker_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """yfinance로 데이터 다운로드 및 기본 지표 계산"""
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        df.dropna(inplace=True)

        # 컬럼명 정리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # 기본 기술적 지표
        df["returns"] = df["Close"].pct_change().fillna(0)
        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["rsi"] = self._compute_rsi(df["Close"])
        df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)

        df.dropna(inplace=True)
        return df

    def _compute_rsi(self, series, window=14):
        """RSI 계산"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def _create_sequences(self, features, target, window_size):
        """시퀀스 생성"""
        X, y = [], []
        for i in range(len(features) - window_size):
            X.append(features[i:i + window_size])
            y.append(target[i + window_size])
        return np.array(X), np.array(y)

    def _build_features_technical(self, df_price: pd.DataFrame) -> pd.DataFrame:
        """
        테크니컬 피처 생성 (build_features_technical 로직 인라인화)
        입력: OHLCV DataFrame[Open,High,Low,Close,Volume]
        출력: TECH 13개 DataFrame(float32), 인덱스 동일
        """
        o, h, l, c, v = df_price["Open"], df_price["High"], df_price["Low"], df_price["Close"], df_price["Volume"]
        out = pd.DataFrame(index=df_price.index)
        
        # 헬퍼 함수들
        def _ema(s, span):
            return s.ewm(span=span, adjust=False).mean()
        
        def _bollinger_pb(s, p=20, k=2.0, eps=1e-12):
            m = s.rolling(p).mean()
            sd = s.rolling(p).std()
            up = m + k*sd
            lo = m - k*sd
            denom = (up - lo).replace(0.0, np.nan)
            return ((s - lo) / (denom + eps))
        
        def _momentum(s, n):
            return s - s.shift(n)
        
        def _true_range(h, l, c):
            tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
            return tr
        
        def _adx(h, l, c, n=14):
            up = h.diff().to_numpy().reshape(-1)
            dn = (-l.diff()).to_numpy().reshape(-1)
            plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
            minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
            tr = _true_range(h, l, c)
            atr_n = tr.ewm(alpha=1/n, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
            minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
            denom = (plus_di + minus_di).replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (denom + 1e-12)
            return dx.ewm(alpha=1/n, adjust=False).mean()
        
        def _obv(c, v):
            direction = np.sign(c.diff().fillna(0.0))
            return (direction * v.fillna(0.0)).cumsum()
        
        # 주차 순환 인코딩
        week = df_price.index.isocalendar().week.astype(float)
        out["weekofyear_sin"] = np.sin(2 * np.pi * week / 52.0)
        out["weekofyear_cos"] = np.cos(2 * np.pi * week / 52.0)
        
        # 기술적 지표
        out["log_ret_lag1"] = np.log(c.shift(1) / c.shift(2))
        out["ret_3d"] = c.pct_change(3)
        out["mom_10"] = _momentum(c, 10)
        out["ma_200"] = c.rolling(200).mean()
        
        ema12, ema26 = _ema(c, 12), _ema(c, 26)
        out["macd"] = ema12 - ema26
        out["bbp"] = _bollinger_pb(c, 20, 2.0)
        out["adx_14"] = _adx(h, l, c, 14)
        out["obv"] = _obv(c, v)
        out["vol_ma_20"] = v.rolling(20).mean()
        out["vol_chg"] = v.pct_change(1)
        
        ret_1d = c.pct_change(1)
        out["vol_20d"] = ret_1d.rolling(20).std()
        
        # 수치화→이상치 처리→ffill→최종 dropna→float32
        out = (out.apply(pd.to_numeric, errors="coerce")
                  .replace([np.inf, -np.inf], np.nan)
                  .ffill()
                  .dropna()
                  .astype(np.float32))
        
        # TECH_COLS 순서 보장
        tech_cols = [
            "weekofyear_sin", "weekofyear_cos", "log_ret_lag1",
            "ret_3d", "mom_10", "ma_200",
            "macd", "bbp", "adx_14",
            "obv", "vol_ma_20", "vol_chg", "vol_20d"
        ]
        out = out.reindex(columns=tech_cols).astype(np.float32)
        return out

    def searcher(self, ticker: Optional[str] = None, rebuild: bool = False):
        """TechnicalAgent 전용 searcher - 내부 구현 (외부 모듈 의존성 제거)"""
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        self.ticker = ticker
        
        raw_csv_path = os.path.join(os.path.dirname(self.data_dir), "raw", f"{ticker}_{agent_id}_raw.csv")
        cfg = agents_info.get(agent_id, {})
        
        # common_params에서 period 가져오기
        period_to_use = common_params.get("period", "2y")
        interval_to_use = cfg.get("interval", "1d")

        need_build = rebuild or (not os.path.exists(raw_csv_path))
        if need_build:
            print(f"⚙️ {ticker} {agent_id} raw CSV not found. Building..." if not os.path.exists(raw_csv_path) else f"⚙️ {ticker} {agent_id} rebuild requested. Building raw CSV...")
            
            # 1) 데이터 다운로드
            df = self._fetch_ticker_data(ticker, period_to_use, interval_to_use)
            
            # 2) 테크니컬 피처 생성 (내부 메서드 사용)
            feat = self._build_features_technical(df[["Open", "High", "Low", "Close", "Volume"]])
            if not isinstance(feat, pd.DataFrame):
                raise TypeError("_build_features_technical는 DataFrame을 반환해야 합니다.")

            # 3) Raw CSV 저장 (Date 첫 컬럼, Close 마지막 컬럼)
            try:
                os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
                raw_tech = feat.copy()
                raw_tech.index.name = "Date"
                raw_tech.reset_index(inplace=True)
                
                # ticker 컬럼 추가
                if "ticker" not in raw_tech.columns:
                    raw_tech.insert(1, "ticker", ticker)
                
                # Close 컬럼 추가 (마지막으로)
                close_df = df[["Close"]].copy()
                close_df.index.name = "Date"
                close_df.reset_index(inplace=True)
                raw_tech = raw_tech.merge(close_df, on="Date", how="left")
                
                # Close를 마지막 컬럼으로 이동
                if "Close" in raw_tech.columns:
                    cols = [c for c in raw_tech.columns if c != "Close"] + ["Close"]
                    raw_tech = raw_tech[cols]
                
                raw_tech.to_csv(raw_csv_path, index=False)
                print(f"✅ {ticker} TechnicalAgent raw features saved to {raw_csv_path} ({len(raw_tech)} rows)")
            except Exception as e:
                print(f"⚠️ Failed to save TechnicalAgent raw features: {e}")
        
        # 4) Raw CSV에서 최신 window_size만큼 직접 추출
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        feature_cols = cfg["data_cols"]
        window_size = cfg["window_size"]
        
        # 피처 추출 (Date, ticker, Close 제외)
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        # 최신 window_size만큼 추출
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        X_latest = X_all[-window_size:].reshape(1, window_size, -1)  # (1, T, F)
        
        # 날짜 추출
        dates_all = df_raw["Date"].values[-window_size:].tolist()
        dates_all = [[str(d) for d in dates_all]]

        # StockData 구성
        self.stockdata = StockData(ticker=ticker)
        self.stockdata.feature_cols = feature_cols
        
        # 날짜 정보 저장
        last_dates = dates_all[0] if dates_all else []
        setattr(self.stockdata, f"{agent_id}_dates_all", dates_all or [])
        setattr(self.stockdata, f"{agent_id}_dates", last_dates or [])
        
        # last_price 안전 변환
        try:
            data = yf.download(ticker, period=period_to_use, interval=interval_to_use, auto_adjust=True, progress=False)
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

        return torch.tensor(X_latest, dtype=torch.float32)

    def pretrain(self):
        """Agent별 사전학습 루틴 - data/raw CSV에서 직접 로드하여 scaling/window 처리"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]
        
        if not self.ticker:
            raise ValueError("TechnicalAgent.pretrain: ticker가 설정되지 않았습니다.")
        
        ticker = self.ticker
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")
        
        # 1) data/raw CSV 로드
        raw_csv_path = os.path.join(os.path.dirname(self.data_dir), "raw", f"{ticker}_{self.agent_id}_raw.csv")
        if not os.path.exists(raw_csv_path):
            print(f"⚙️ {ticker} {self.agent_id} raw CSV not found. Running searcher() to generate it...")
            _ = self.searcher(ticker, rebuild=True)
            if not os.path.exists(raw_csv_path):
                raise FileNotFoundError(f"Raw CSV not found after searcher: {raw_csv_path}")
        
        # raw CSV 읽기 (Date 첫 컬럼, Close 마지막 컬럼)
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        # 피처 컬럼 추출 (Date, ticker, Close 제외)
        cfg = agents_info.get(self.agent_id, {})
        feature_cols = cfg["data_cols"]
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        # 타겟 생성 (다음날 수익률)
        close_prices = df_raw["Close"].values
        y_all = (close_prices[1:] / close_prices[:-1] - 1.0).reshape(-1, 1).astype(np.float32)
        X_all = X_all[:-1]  # 마지막 행 제외 (타겟이 없음)
        
        # 백테스팅 모드: simulation_date 이전 데이터만 필터링
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            sim_date = datetime.strptime(self.simulation_date, "%Y-%m-%d")
            print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터만 사용")
            dates = df_raw["Date"].values[:-1]  # 마지막 제외
            valid_mask = pd.to_datetime(dates) <= sim_date
            X_all = X_all[valid_mask]
            y_all = y_all[valid_mask]
            print(f"[INFO] 필터링 후 데이터: {len(X_all)}개 샘플")
        
        # 2) Window 처리 (시퀀스 생성)
        window_size = self.window_size
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        X_seq, y_seq = self._create_sequences(X_all, y_all, window_size)
        print(f"[INFO] 시퀀스 생성 완료: {X_seq.shape}, {y_seq.shape}")
        
        # 3) 타깃 스케일 조정
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_seq = y_seq * y_scale_factor
        
        # 4) Scaling
        self.scaler.fit_scalers(X_seq, y_seq)
        self.scaler.save(ticker)
        
        X_train, y_train = map(torch.tensor, self.scaler.transform(X_seq, y_seq))
        X_train, y_train = X_train.float(), y_train.float()
        
        # 5) 모델 학습
        model = self
        self._modules.pop("model", None)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        huber_delta = common_params.get("huber_loss_delta", 1.0)
        loss_fn = torch.nn.HuberLoss(delta=huber_delta)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train.view(-1, 1)),
                                  batch_size=batch_size, shuffle=True)
        
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
        
        # 6) 모델 저장
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        
        # model_loaded 플래그 설정
        self.model_loaded = True
        
        # 7) 전처리된 데이터 저장 (선택적)
        dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_id}_dataset.csv")
        flattened_data = []
        dates_list = df_raw["Date"].values[:-1]  # 마지막 제외
        
        for sample_idx in range(len(X_seq)):
            for time_idx in range(window_size):
                date_idx = sample_idx + time_idx
                row = {
                    'sample_id': sample_idx,
                    'time_step': time_idx,
                    'date': str(dates_list[date_idx]) if date_idx < len(dates_list) else None,
                    'target': y_seq[sample_idx, 0] if time_idx == window_size - 1 else np.nan,
                }
                for feat_idx, feat_name in enumerate(feature_cols):
                    row[feat_name] = X_seq[sample_idx, time_idx, feat_idx]
                flattened_data.append(row)
        
        dataset_df = pd.DataFrame(flattened_data)
        os.makedirs(self.data_dir, exist_ok=True)
        dataset_df.to_csv(dataset_path, index=False)
        print(f"✅ {self.agent_id} 모델 학습 및 저장 완료: {model_path}")
        print(f"✅ 전처리된 데이터 저장 완료: {dataset_path}")

    def predict(self, X, n_samples: Optional[int] = None, current_price: Optional[float] = None, X_last: Optional[np.ndarray] = None):
        """
        Monte Carlo Dropout 기반 예측 + 불확실성(σ) 및 confidence 계산 (안정형)
        """
        # n_samples 설정 (config에서 가져오기)
        if n_samples is None:
            n_samples = common_params.get("n_samples", 30)
        
        # 1) 모델 및 스케일러 준비
        if not self.ticker:
            raise ValueError("ticker가 설정되지 않았습니다. 먼저 searcher(ticker)를 호출하세요.")
        
        # 모델 파일 확인
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        if not os.path.exists(model_path):
            print(f"[{self.agent_id}] 모델이 없어 pretrain()을 실행합니다...")
            self.pretrain()
        else:
            # 모델이 있으면 로드 (이미 로드되었는지 확인)
            if not hasattr(self, "model_loaded") or not self.model_loaded:
                self.load_model(model_path)
        
        # 스케일러 파일 확인
        scaler_x_path = os.path.join(self.scaler.save_dir, f"{self.ticker}_{self.agent_id}_xscaler.pkl")
        scaler_y_path = os.path.join(self.scaler.save_dir, f"{self.ticker}_{self.agent_id}_yscaler.pkl")
        if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
            print(f"[{self.agent_id}] 스케일러가 없어 pretrain()을 실행합니다...")
            self.pretrain()
        
        model = self  # TechnicalAgent 자체가 nn.Module
        self.scaler.load(self.ticker)

        # 2) 입력 변환 + 학습과 동일 스케일로 변환 (StockData 지원)
        if isinstance(X, StockData):
            sd = X
            X_in = getattr(sd, "X_seq", None)
            if X_in is None:
                # StockData에 X_seq가 없으면 agent_id로 찾기
                X_in = getattr(sd, self.agent_id, None)
                if isinstance(X_in, dict):
                    # dict 형태면 DataFrame으로 변환
                    df = pd.DataFrame(X_in)
                    X_in = df.values
            if X_in is None:
                raise ValueError(f"StockData에 {self.agent_id} 데이터가 없습니다. searcher()를 먼저 호출하세요.")
            if current_price is None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
            X = X_in
        
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
        sigma_min = common_params.get("sigma_min", 1e-6)
        sigma = max(sigma, sigma_min)
        confidence = 1 / (1 + np.log1p(sigma))

        # 5) 타깃 역스케일링 및 가격 변환
        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        # current_price 결정
        if current_price is None:
            last_price = getattr(getattr(self, "stockdata", None), "last_price", None)
            default_price = common_params.get("default_current_price", 100.0)
            current_price = default_price if last_price is None else last_price

        # 학습 타깃은 "다음날 수익률(%)"이므로 스케일 팩터로 나눠서 사용
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        predicted_return = float(mean_pred[-1]) / y_scale_factor
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
            # stockdata에서 X 재구성 (중복 searcher 방지)
            agent_data = getattr(stock_data, self.agent_id, {})
            if isinstance(agent_data, dict) and agent_data:
                df = pd.DataFrame(agent_data)
                X_input = torch.tensor(
                    df.tail(self.window_size).values, 
                    dtype=torch.float32
                ).unsqueeze(0)  # (1,T,F)
            else:
                # 만약 비어있으면 searcher 재호출
                X_input = self.searcher(self.ticker)
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
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
    ):
        """
        Revision 단계
        - σ 기반 β-weighted 신뢰도 계산
        - γ 수렴율로 예측값 보정
        - fine-tuning (수익률 단위)
        - reasoning 생성
        """
        # Fine-tuning 파라미터 설정 (config에서 가져오기)
        if lr is None:
            lr = common_params.get("fine_tune_lr", 1e-4)
        if epochs is None:
            epochs = agents_info.get(self.agent_id, {}).get("fine_tune_epochs", 20)
        
        gamma = getattr(self, "gamma", 0.3)               # 수렴율 (0~1)
        delta_limit = getattr(self, "delta_limit", 0.05)  # fine-tuning 보정 한계
        default_price = common_params.get("default_current_price", 100.0)
        current_price = getattr(self.stockdata, "last_price", default_price)  # 수정: 항상 초반에 현재가 확보

        try:
            # β 계산 (불확실성 작을수록 신뢰 높음)
            my_price = float(my_opinion.target.next_close)           # 수정: float 캐스팅
            sigma_min = common_params.get("sigma_min", 1e-6)
            my_sigma = abs(my_opinion.target.uncertainty or sigma_min)

            # 수정: others가 없을 때 방어
            if len(others) == 0:
                revised_price = my_price
            else:
                other_prices = np.array([o.target.next_close for o in others], dtype=float)
                other_sigmas = np.array([abs(o.target.uncertainty or sigma_min) for o in others], dtype=float)

                all_sigmas = np.concatenate([[my_sigma], other_sigmas])

                inv_sigmas = 1 / (all_sigmas + sigma_min)
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
                y_scale_factor = common_params.get("y_scale_factor", 100.0)
                revised_return_scaled = revised_return * y_scale_factor           # 예: 1.2

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
                # pretrain과 통일: HuberLoss (config에서 가져오기)
                huber_delta = common_params.get("huber_loss_delta", 1.0)
                criterion = torch.nn.HuberLoss(delta=huber_delta)

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
            
            # model_loaded 플래그 설정
            self.model_loaded = True

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

        # 2) 시계열 분할 (config에서 분할 비율 가져오기)
        split_ratio = common_params.get("eval_split_ratio", 0.8)
        split_idx = int(len(X) * split_ratio)
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        # 3) 스케일러 로드 + y 스케일(학습과 동일하게 ×100)  # 수정
        self.scaler.load(ticker)

        # 🔧 수정: y를 1D로 맞춰줍니다. (config에서 스케일 팩터 가져오기)
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_val_scaled = (y_val * y_scale_factor).reshape(-1)

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
