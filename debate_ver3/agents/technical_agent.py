# technical_agent.py
# ============================================================
# Stage 1 — TECH(12) + FUND(7) → [GRU ⊕ MLP] with Attention + 2-layer Gating
# Inference-only: BaseAgent 상속 + MC Dropout 예측(Target.idea에 explain + LLM opinion 포함)
# 학습·특징생성·해석 유틸은 technical_core.py에서 임포트
# ============================================================

from __future__ import annotations
import os, json, csv, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.preprocessing import MinMaxScaler

# 프로젝트 BaseAgent
from debate_ver3.agents.base_agent import BaseAgent

# ---------- core import ----------
from .technical_core import (
    # 데이터/설정
    START_DATE, END_DATE_INCLUSIVE, JAN_START, JAN_END, OUTDIR,
    nasdaq100_kor, FIXED,
    FEATURES_TECH, FEATURES_FUND, FEATURES,
    # 모델과 빌더
    DualBranchGRUGatedRegressor,
    fetch_ohlcv_fixed_window, build_features,
    # 해석/저장
    explain_basis_json, update_json,
)

# =========================
# 재현성 및 디바이스
# =========================
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LLM 환경
# =========================
_OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_KEY = os.getenv("CAPSTONE_OPENAI_API")  # ← 여기에 키 설정
_OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "120"))

# =========================
# Target 컨테이너
# =========================
@dataclass
class Target:
    next_close: float
    uncertainty: float
    confidence: float
    idea: Dict

# =========================
# TechnicalAgent
# =========================
class TechnicalAgent(BaseAgent):
    """
    - searcher(): 입력 패키징
    - predict(): MC Dropout 예측 + explain + LLM opinion
    - predict_month_with_daily_assimilation(): explain.json 누적
    """
    def __init__(self, agent_id: str = "TechnicalAgent", outdir: str = OUTDIR):
        super().__init__(agent_id=agent_id)
        self.outdir = outdir
        self.fixed = FIXED.copy()
        self.device = DEVICE

        self.model: Optional[DualBranchGRUGatedRegressor] = None
        self.scaler_tech: Optional[MinMaxScaler] = None
        self.scaler_fund: Optional[MinMaxScaler] = None
        self.id_map: Dict[str, int] = {}
        self.lookback: int = int(self.fixed["lookback"])
        self.feat_cols: List[str] = FEATURES
        self.tickers: List[str] = sorted(set(nasdaq100_kor.values()))
        self._load_inference_bundle_if_any()

    # ---------------- Bundle 로드 ----------------
    def _load_inference_bundle_if_any(self):
        path = os.path.join(self.outdir, "inference_bundle.pt")
        if not os.path.exists(path):
            self.id_map = {t: i for i, t in enumerate(self.tickers)}
            return
        bundle = torch.load(path, map_location=self.device, weights_only=False)
        self.lookback    = int(bundle.get("lookback", self.lookback))
        self.feat_cols   = list(bundle.get("feat_cols", FEATURES))
        self.tickers     = list(bundle.get("tickers", self.tickers))
        self.id_map      = dict(bundle.get("id_map", {t: i for i, t in enumerate(self.tickers)}))
        self.scaler_tech = bundle.get("scaler_tech", None)
        self.scaler_fund = bundle.get("scaler_fund", None)

        tech_in = len(FEATURES_TECH) + len(self.id_map)
        fund_in = len(FEATURES_FUND) + len(self.id_map)
        self.model = DualBranchGRUGatedRegressor(
            tech_input_dim=tech_in, fund_input_dim=fund_in,
            u1=self.fixed["units1"], u2=self.fixed["units2"],
            mlp_hidden=self.fixed["mlp_hidden"], dropout=self.fixed["dropout"],
            gate_hidden=self.fixed["gate_hidden"]
        ).to(self.device)
        self.model.load_state_dict(bundle["state_dict"])

    # ---------------- 내부 유틸 ----------------
    @staticmethod
    def _resolve_ticker(user_input: str) -> str:
        u = user_input.strip().upper()
        for k, v in nasdaq100_kor.items():
            if u in [k.upper(), v.upper()]:
                return v.upper()
        return u

    def _build_single_inputs(self, ticker: str, start_date: str, end_inclusive: str
                             ) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        raw = fetch_ohlcv_fixed_window(ticker, start_date, end_inclusive)
        feat = build_features(raw, ticker).replace([np.inf, -np.inf], np.nan).ffill()
        feat = feat.reindex(columns=self.feat_cols).ffill().fillna(0.0)
        if len(feat) < self.lookback + 1:
            raise ValueError(f"Not enough rows for {ticker} to build a window of {self.lookback}")
        close = raw["Close"]; last_close = float(close.iloc[-1])

        Xwin_tech = feat[FEATURES_TECH].iloc[-self.lookback:].values.astype(np.float32)  # (W,Ft)
        xvec_fund = feat[FEATURES_FUND].iloc[-1:].values.astype(np.float32)              # (1,Ff)

        if self.scaler_tech is None:
            sc = MinMaxScaler(); sc.fit(Xwin_tech.reshape(-1, Xwin_tech.shape[1])); self.scaler_tech = sc
        if self.scaler_fund is None:
            scf = MinMaxScaler(); scf.fit(xvec_fund); self.scaler_fund = scf

        W, Ft = Xwin_tech.shape
        Xwin_tech_s = self.scaler_tech.transform(Xwin_tech.reshape(W, Ft)).reshape(1, W, Ft).astype(np.float32)
        xvec_fund_s = self.scaler_fund.transform(xvec_fund).astype(np.float32)

        K = len(self.id_map)
        if ticker in self.id_map:
            idx = self.id_map[ticker]
            oh_seq = np.repeat(np.eye(K, dtype=np.float32)[idx][None, None, :], repeats=W, axis=1)
            oh_vec = np.eye(K, dtype=np.float32)[idx][None, :]
        else:
            oh_seq = np.zeros((1, W, K), dtype=np.float32)
            oh_vec = np.zeros((1, K), dtype=np.float32)

        Xseq_in  = np.concatenate([Xwin_tech_s, oh_seq], axis=2).astype(np.float32)  # (1,W,Ft+K)
        Xfund_in = np.concatenate([xvec_fund_s, oh_vec], axis=1).astype(np.float32)  # (1,Ff+K)

        if self.model is None:
            self.model = DualBranchGRUGatedRegressor(
                tech_input_dim=Xseq_in.shape[2],
                fund_input_dim=Xfund_in.shape[1],
                u1=self.fixed["units1"], u2=self.fixed["units2"],
                mlp_hidden=self.fixed["mlp_hidden"], dropout=self.fixed["dropout"],
                gate_hidden=self.fixed["gate_hidden"]
            ).to(self.device)

        win_index = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in feat.index[-self.lookback:]]
        return Xseq_in, Xfund_in, last_close, win_index

    # ---------------- 공개: searcher ----------------
    def searcher(self, ticker: str,
                 start_date: str = START_DATE,
                 basis_date: str = END_DATE_INCLUSIVE) -> Dict:
        tkr = self._resolve_ticker(ticker)
        Xseq, Xfund, last_close, win_index = self._build_single_inputs(tkr, start_date, basis_date)
        return {"ticker": tkr, "basis_date": basis_date, "Xseq": Xseq, "Xfund": Xfund,
                "last_close": last_close, "win_index": win_index}

    # ---------------- 내부: 1-step 전방 ----------------
    def _forward_once(self, Xseq_in: np.ndarray, Xfund_in: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            xs = torch.from_numpy(Xseq_in).to(self.device)
            xf = torch.from_numpy(Xfund_in).to(self.device)
            y, g, y_gru, y_mlp, attn = self.model(xs, xf)
            return (float(y.cpu().numpy().ravel()[0]),
                    float(g.cpu().numpy().ravel()[0]),
                    float(y_gru.cpu().numpy().ravel()[0]),
                    float(y_mlp.cpu().numpy().ravel()[0]),
                    attn.cpu().numpy().ravel())

    # ---------------- 프롬프트 빌더 ----------------
    def _build_messages_opinion(self, ticker: str, last_price: float,
                                mc_stats: Dict, exp: Dict) -> tuple[str, str]:
        # 요약 축소본
        attn = exp.get("time_attention", {}) or {}
        tech_agg = exp.get("time_feature", {}).get("avg_gxi") or {}
        fund_imp = exp.get("fund_importance", {}) or {}
        gate = exp.get("gate", {}) or {}
        pred = exp.get("pred", {}) or {}

        # attention 농도 지표
        attn_vals = np.array(list(attn.values()), dtype=float)
        attn_sum = float(attn_vals.sum()) if attn_vals.size else 0.0
        topk = 3
        top3_sum = float(np.sort(attn_vals)[-topk:].sum()) if attn_vals.size else 0.0
        top3_ratio = float(top3_sum / attn_sum) if attn_sum > 0 else 0.0
        if attn_sum > 0:
            p = attn_vals / attn_sum
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            entropy = 0.0

        system_text = (
            "너는 금융 시계열 예측 모델의 결과를 설명하는 분석가다. 한국어로만 답한다. "
            "오직 JSON 한 개 오브젝트만 반환하라. 코드펜스/여분 텍스트 금지. "
            "미래 수치 예측 금지. 입력 블록 내 데이터만 재인용. "
            "반드시 지켜라: 1) 'tech_top_features' 최소 2개, 2) 'fund_top_features' 최소 2개, "
            "3) 'risks' 최소 2개, 4) 'caveats' 최소 2개, 5) 각 항목 evidence 포함."
        )

        schema = {
            "narrative": "string",
            "attention_summary": {"top_days": [{"date":"YYYY-MM-DD","attention":0.0}]},
            "tech_top_features": [{"feature":"...","importance":0.0,"sign":"pos|neg"}],
            "fund_top_features": [{"feature":"...","importance":0.0,"sign":"pos|neg"}],
            "risks": [{"text":"...","evidence":"tech:..., fund:..., attn:..., gate:..., mc:..."}],
            "caveats": [{"text":"...","evidence":"tech:..., fund:..., attn:..., gate:..., mc:..."}],
            "key": "string"
        }

        rules = (
            "[RISKS 지시]\n"
            "- mc.std_log가 크거나 mc.cv_price가 높으면 '불확실성 확대'.\n"
            "- mc.gate_mean≈0.5±0.1이면 'tech vs fund 충돌'.\n"
            "- attention.top3_ratio<0.5이면 '집중도 낮음'.\n"
            "- tech_top vs fund_top 방향 상충 시 '신호 불일치'.\n"
            "[CAVEATS 지시]\n"
            "- fund_importance 전반 약하면 '펀더멘털 비중 낮음'.\n"
            "- 소수 피처에 과집중이면 '특정 신호 의존'.\n"
            "- 외부 거시/이벤트 미포함 등 구조적 한계 명시.\n"
            "각 항목 evidence에 출처를 요약하라."
        )

        ctx = {
            "ticker": ticker,
            "last_price": float(last_price),
            "mc": mc_stats,  # samples, mean_log, std_log, cv_price, gate_mean
            "attention": attn,
            "attention_metrics": {"top3_ratio": top3_ratio, "entropy": entropy},
            "tech_feature_agg": tech_agg,
            "fund_importance": fund_imp,
            "gate": gate,
            "pred": pred
        }

        user_text = (
            "[TASK] 다음 입력을 근거 중심으로 요약하라. 숫자 예측 생성 금지.\n"
            f"[CONTEXT]\n{json.dumps(ctx, ensure_ascii=False)}\n"
            f"{rules}\n"
            "[OUTPUT_JSON_SCHEMA]\n" + json.dumps(schema, ensure_ascii=False)
        )
        return system_text, user_text

    # ---------------- LLM 호출 ----------------
    def _call_llm_opinion(self, system_text: str, user_text: str) -> Dict:
        if not _OPENAI_KEY:
            return {"error": "OPENAI_API_KEY not set"}
        url = f"{_OPENAI_BASE.rstrip('/')}/chat/completions"
        payload = {
            "model": _OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
        }
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=_OPENAI_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"].get("content") or "{}"
            try:
                return json.loads(content)
            except Exception:
                i, j = content.find("{"), content.rfind("}")
                if i != -1 and j != -1:
                    try:
                        return json.loads(content[i:j+1])
                    except Exception:
                        pass
                return {"error": "invalid_json", "raw": content[:1000]}
        except requests.RequestException as e:
            return {"error": "request_failed", "detail": str(e)[:500]}
        except Exception as e:
            return {"error": "unknown", "detail": str(e)[:500]}

    # ---------------- 공개: MC Dropout 예측 ----------------
    def predict(self, X_data: Dict, n_samples: int = 50) -> Target:
        ticker = X_data["ticker"]; basis = X_data["basis_date"]
        Xseq = X_data["Xseq"]; Xfund = X_data["Xfund"]; last_close = float(X_data["last_close"])

        preds = []; gates = []; comps = []
        self.model.train()
        with torch.no_grad():
            xs = torch.from_numpy(Xseq).to(self.device)
            xf = torch.from_numpy(Xfund).to(self.device)
            for _ in range(n_samples):
                y, g, y_gru, y_mlp, _ = self.model(xs, xf)
                preds.append(float(y.cpu().numpy().ravel()[0]))
                gates.append(float(g.cpu().numpy().ravel()[0]))
                comps.append((float(y_gru.cpu().numpy().ravel()[0]), float(y_mlp.cpu().numpy().ravel()[0])))
        preds = np.asarray(preds)
        mean_log = float(preds.mean()); std_log = float(preds.std())
        gate_mean = float(np.mean(gates)) if gates else float("nan")

        next_close_mean = float(last_close * np.exp(mean_log))
        next_close_std  = float(last_close * np.exp(mean_log) * max(std_log, 1e-12))
        cv_price = float(next_close_std / max(next_close_mean, 1e-8))

        artifact = {
            "model": self.model,
            "scaler_tech": self.scaler_tech,
            "scaler_fund": self.scaler_fund,
            "id_map": self.id_map,
            "lookback": self.lookback,
            "feat_cols": self.feat_cols
        }
        exp = explain_basis_json(ticker, basis, artifact, start_date=START_DATE, topk_per_time=3)

        # LLM opinion
        mc_stats = {
            "samples": int(n_samples),
            "mean_log": mean_log,
            "std_log": std_log,
            "gate_mean": gate_mean,
            "cv_price": cv_price
        }
        sys_text, usr_text = self._build_messages_opinion(ticker, last_close, mc_stats, exp)
        llm_op = self._call_llm_opinion(sys_text, usr_text)

        idea = {
            "basis": basis,
            "ticker": ticker,
            "mc_stats": mc_stats,
            "explain": exp,
            "llm_opinion": llm_op
        }

        uncertainty = float(next_close_std)
        confidence  = float(1.0 / (uncertainty + 1e-8))
        return Target(next_close=next_close_mean, uncertainty=uncertainty, confidence=confidence, idea=idea)

    # ---------------- 공개: 월간 동화 + explain.json ----------------
    def predict_month_with_daily_assimilation(self, user_input: str,
                                              month_start: str = JAN_START, month_end: str = JAN_END,
                                              start_date: str = START_DATE, last_train_day: str = END_DATE_INCLUSIVE):
        ticker = self._resolve_ticker(user_input)
        basis = last_train_day; rows = []

        pred_csv_path  = os.path.join(self.outdir, "pred.csv")
        explain_json   = os.path.join(self.outdir, "explain.json")
        evid_fund_json = os.path.join(self.outdir, "feature_evidence_fund.json")

        def _fetch_next_close(basis_date_inclusive: str) -> Tuple[str, float]:
            start = (pd.to_datetime(basis_date_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            end_exclusive = (pd.to_datetime(basis_date_inclusive) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            from .technical_core import robust_yf_history, _coerce_tz_naive
            df = robust_yf_history(ticker, start, end_exclusive)
            df = _coerce_tz_naive(df)[["Close"]].dropna()
            actual_dt = pd.to_datetime(df.index[0]).strftime("%Y-%m-%d")
            actual_close = float(df["Close"].iloc[0])
            return actual_dt, actual_close

        while True:
            X = self.searcher(ticker, start_date=start_date, basis_date=basis)
            yp_log, g, y_gru, y_mlp, attn = self._forward_once(X["Xseq"], X["Xfund"])

            next_dt, actual_close = _fetch_next_close(basis)
            next_dt_ts = pd.to_datetime(next_dt)
            if next_dt_ts < pd.to_datetime(month_start):
                basis = next_dt; continue
            if next_dt_ts > pd.to_datetime(month_end):
                break

            today_close = X["last_close"]
            pred_next = float(today_close * np.exp(yp_log))
            rows.append({
                "기준일": basis, "예측대상일": next_dt, "ticker": ticker,
                "기준일종가": today_close, "예측종가": pred_next, "실제종가": actual_close,
                "오차": pred_next - actual_close,
                "오차율(%)": ((pred_next - actual_close) * 100.0 / actual_close)
            })

            new = not os.path.exists(pred_csv_path)
            with open(pred_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new: w.writerow(["date", "ticker", "today_close", "pred_next_close"])
                w.writerow([basis, ticker, today_close, pred_next])

            artifact = {
                "model": self.model,
                "scaler_tech": self.scaler_tech,
                "scaler_fund": self.scaler_fund,
                "id_map": self.id_map,
                "lookback": self.lookback,
                "feat_cols": self.feat_cols
            }
            exp = explain_basis_json(ticker, basis, artifact, start_date=START_DATE, topk_per_time=3)
            update_json(explain_json, f"{basis}|{ticker}", exp)

            fund_map = exp.get("fund_importance", {})
            fund_list = [{"name": k, "score": v} for k, v in fund_map.items()]
            update_json(evid_fund_json, f"{basis}|{ticker}", {"top_features": fund_list})

            basis = next_dt
            if pd.to_datetime(basis) >= pd.to_datetime(month_end):
                break

        df = pd.DataFrame(rows)
        if not df.empty:
            prev = df["기준일종가"].values
            direction_pred = np.sign(df["예측종가"].values - prev)
            direction_true = np.sign(df["실제종가"].values - prev)
            df["방향일치"] = (direction_pred == direction_true).astype(int)
            mape = float(np.mean(np.abs((df["예측종가"] - df["실제종가"]) / df["실제종가"])) * 100.0)
            dir_acc = float(df["방향일치"].mean() * 100.0)
        else:
            mape, dir_acc = np.nan, np.nan

        out_csv = os.path.join(self.outdir, f"jan2025_{ticker}.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        return df, mape, dir_acc, out_csv


# ============================================================
# 단독 실행 예시
# ============================================================
if __name__ == "__main__":
    agent = TechnicalAgent(agent_id="TechnicalAgent")
    Xdata = agent.searcher(ticker="MSFT", start_date=START_DATE, basis_date=END_DATE_INCLUSIVE)
    tgt = agent.predict(Xdata, n_samples=50)

    print("next_close:", round(tgt.next_close, 4),
          "uncertainty:", round(tgt.uncertainty, 4),
          "confidence:", round(tgt.confidence, 6))

    # LLM opinion 출력
    op = tgt.idea.get("llm_opinion", {})
    print("\nllm_opinion:\n", json.dumps(op, ensure_ascii=False, indent=2))
