import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import os
from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal
from config.agents import agents_info, dir_info
from agents.base_agent import r4 # 아연수정
import json
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
import torch.nn as nn
from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal
from config.agents import agents_info, dir_info
import json
from prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from typing import List, Optional
import numpy as np
from agents.base_agent import r4, pct4  # 아연수정



class TechnicalAgent(BaseAgent, nn.Module):
    """Technical Agent: BaseAgent + LSTM×2 + time-attention"""
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
        BaseAgent.__init__(self, agent_id, **kwargs)
        nn.Module.__init__(self)


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



    '''
    생략(아연수정)
    def _build_model(self):
        """TechnicalAgent 기본 GRU 모델 자동 생성"""
        import torch.nn as nn
        import torch

        input_dim = getattr(self, "input_dim", 10)
        hidden_dim = getattr(self, "hidden_dim", 64)
        dropout_rate = getattr(self, "dropout_rate", 0.2)

        class GRUNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, dropout_rate):
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout_rate)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    # nn.Tanh()  # 기존: 출력을 -1~1로 제한 (문제 원인)
                    # 수정: Tanh 제거하여 선형 출력으로 변경 - 상승/하락율 예측에 적합
                )

            def forward(self, x):
                out, _ = self.gru(x)          # out: (batch, seq, hidden)
                out = out[:, -1, :]           # 마지막 시점(hidden state)
                out = self.dropout(out)
                return self.fc(out)           # (batch, 1)

        model = GRUNet(input_dim, hidden_dim, dropout_rate)
        print(f"🧠 GRU 모델 생성됨 (input={input_dim}, hidden={hidden_dim}, dropout={dropout_rate})")
        return model
    '''

    # (아연수정) 기존 GRU 팩토리 우회 용도
    def _build_model(self):
        """TechnicalAgent용 LSTM×2 + time-attention 모델 생성기"""
        return self  # 이미 __init__에서 모델 구성 완료


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM×2 + time-attention이 있으면 사용 (아연수정)
        h1, _ = self.lstm1(x)
        h1 = self.drop(h1)
        h2, _ = self.lstm2(h1)
        h2 = self.drop(h2)

        w = torch.softmax(torch.matmul(h2, self.attn_vec), dim=1)  # [B,T]
        self._last_attn = w.detach()                               # 아연수정
        ctx = (h2 * w.unsqueeze(-1)).sum(dim=1)                    # [B,u2]
        return self.fc(ctx)                                        # [B,1]


    # 아연수정
    # ------------ 설명 유틸 ------------
    @torch.no_grad()
    def time_attention_dict(self, dates: list) -> dict:
        """직전 forward의 softmax 가중치(w)를 날짜와 매핑."""
        attn = getattr(self, "_last_attn", None)          # [B,T]
        if attn is None:
            return {}
        a = attn[0].detach().cpu().tolist()

        # 날짜 길이와 맞지 않으면 기본 인덱스 사용
        if not dates or len(dates) != len(a):
            dates = [f"t-{len(a)-1-i}" for i in range(len(a))]
        return {str(d): float(w) for d, w in zip(dates, a)}

    def _time_feature_attrib_gradxinput(self, x: torch.Tensor, dates: list, top_k: int = 5) -> dict:
        """
        x: [1,T,F] 단일 배치 입력. 날짜 길이=T.
        반환: {date: {feature: score,...}}  (상위 top_k)
        """
        self.eval()
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            y = self(x).sum()
            self.zero_grad(set_to_none=True)
            y.backward()
            grads = x.grad.abs()    # [1, T, F]
            attn = getattr(self, "_last_attn", None)  # [1,T] or None
            contrib = grads * x.abs()    # [1,T,F]

        # attention 가중치가 있으면 곱하기
        if attn is not None:
            contrib = contrib * attn.detach().unsqueeze(-1)

        contrib = contrib[0].detach().cpu().numpy()        # [T,F]
        cols = list(getattr(self.stockdata, "feature_cols", []))[: self.input_dim] # 아연수정
        if not dates or len(dates) != contrib.shape[0]: # 아연수정
            dates = [f"t-{contrib.shape[0] - 1 - i}" for i in range(contrib.shape[0])]

        out = {}
        for t, d in enumerate(dates):
            pairs = sorted(
                zip(cols, contrib[t].tolist()), key=lambda z: z[1], reverse=True)[:top_k]
            out[str(d)] = {k: float(v) for k, v in pairs}
        return out

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
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / len(w)

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

    def explain_last(self, X_last: torch.Tensor, dates: list | None = None, top_k: int = 3):
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
        # 1) 스케일 정합
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

        # 2) Attention
        time_attn = self.time_importance_from_attention(Xs)  # (T,)

        # 3) Grad×Input
        g_time, g_feat, gi_raw = self.gradxinput_attrib(Xs, eps=0.0)

        # 4) Occlusion
        occ_time = self.occlusion_time(Xs, fill="zero", batch=32)
        occ_feat = self.occlusion_feature(Xs, fill="zero", batch=32)

        # 5) 융합 요약(가중 평균)
        g_time_n = g_time / (g_time.sum() + 1e-12)
        g_feat_n = g_feat / (g_feat.sum() + 1e-12)
        per_time = 0.5 * time_attn + 0.3 * g_time_n + 0.2 * occ_time
        occ_feat_n = occ_feat / (occ_feat.sum() + 1e-12)
        per_feat = 0.7 * g_feat_n + 0.3 * occ_feat_n

        # 6) 날짜별 상위 피처(Grad×Input 기준)
        time_feature = {}
        gi_abs = np.abs(gi_raw)
        for t_idx, d in enumerate(dates):
            pairs = sorted(zip(feat_names, gi_abs[t_idx].tolist()), key=lambda z: z[1], reverse=True)[:top_k]
            time_feature[str(d)] = {k: float(v) for k, v in pairs}

        # 7) time-attention 맵
        time_attention = {str(d): r4(w) for d, w in zip(dates, time_attn.tolist())}

        # 8) 패킷
        per_time_list = [{"date": str(d), "sum_abs": r4(v)} for d, v in zip(dates, per_time.tolist())]
        per_feat_list = [{"feature": k, "sum_abs": r4(v)} for k, v in sorted(zip(feat_names, per_feat.tolist()),
                                                                          key=lambda z: z[1], reverse=True)]
        # 날짜별 상위 피처(Grad×Input 기준, 라운딩)
        time_feature = {}
        gi_abs = np.abs(gi_raw)
        for t_idx, d in enumerate(dates):
            pairs = sorted(zip(feat_names, gi_abs[t_idx].tolist()),
                       key=lambda z: z[1], reverse=True)[:top_k]
            time_feature[str(d)] = {k: r4(v) for k, v in pairs}

        evidence = {
            "attention": [r4(x) for x in time_attn.tolist()],
            "gradxinput_feat": [r4(x) for x in g_feat.tolist()],
            "occlusion_time": [r4(x) for x in occ_time.tolist()],
            "window_size": int(T),
            }

        return {
            "per_time": per_time_list,
            "per_feature": per_feat_list,
            "time_attention": time_attention,
            "time_feature": time_feature,
            "evidence": evidence,
            "raw": {"gradxinput": gi_abs.tolist()}  # 원시값은 비라운딩 유지 가능
          }

    # idea 핵심만
    def _pack_idea(exp: dict, top_time=8, top_feat=6, coverage=0.8):
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

        attn = exp.get("evidence", {}).get("attention", [])
        peak = picked[0]["date"] if picked else None
        return {"top_time": picked, "top_features": top_features,
            "peak_date": peak, "window_size": exp.get("evidence",{}).get("window_size")}


   # LLM Reasoning 메시지 (아연수정)

    def _build_messages_opinion(self, stock_data, target):
        """TechnicalAgent용 LLM 프롬프트 메시지 구성 + 설명값 포함"""
        last = float(getattr(stock_data, "last_price", target.next_close))
        delta = (target.next_close / last) - 1.0

        # 최신 윈도우 설명 산출
        X_last = self.searcher(self.ticker)
        if not isinstance(X_last, torch.Tensor):
          X_last = torch.tensor(X_last, dtype=torch.float32)
        T = X_last.shape[1]
        dates = getattr(self.stockdata, f"{self.agent_id}_dates", [])[-T:] or [f"t-{T-1-i}" for i in range(T)]
        exp = self.explain_last(X_last, dates, top_k=5) if not getattr(target, "idea", None) else None
        idea = target.idea if target.idea else _pack_idea(exp)
        target.idea = idea  # Target에 저장


        # 기본 컨텍스트
        ctx = {
            "ticker": getattr(stock_data, "ticker", "Unknown"),
            "last_price": r4(last),
            "next_close": r4(target.next_close),
            "uncertainty": r4(target.uncertainty),
            "confidence": r4(target.confidence),
            "delta_pct": pct4(delta),
            "sigma": r4(target.uncertainty or 0.0),
            "beta": r4(target.confidence or 0.0),
            "window_size": int(self.window_size),
            "idea": idea,  # 핵심만
        }



        from prompts import OPINION_PROMPTS

        system_text = OPINION_PROMPTS[self.agent_id]["system"]
        user_text = OPINION_PROMPTS[self.agent_id]["user"].format(
        context=json.dumps(ctx, ensure_ascii=False)
        )
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
            "data_summary": getattr(stock_data, self.agent_id, {}).get("feature_cols", []),
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
        # (최근 7~14일 정도면 LLM이 이해 가능한 범위)
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[self.window_size:]  # 최근 14일치 전체 시계열
            else:
                ctx[col] = [values]

        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        user_text   = REBUTTAL_PROMPTS[self.agent_id]["user"].format(
            context=json.dumps(ctx, ensure_ascii=False)
        )
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
            "data_summary": getattr(stock_data, self.agent_id, {}).get("feature_cols", []),
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