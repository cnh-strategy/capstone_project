# scripts/sentimental_demo/demo_ctx_prompt.py
# ===============================================================
# SentimentalAgent CTX 데모 스크립트
#  - 데이터셋 로드 및 모델 예측
#  - 예측 결과를 ctx(dict) 형태로 구성
#  - opinion/reason/prompt 출력
# ===============================================================

from __future__ import annotations
import argparse
import json
from datetime import datetime
import pytz

# ===== 에이전트 로드 =====
from agents.sentimental_agent import SentimentalAgent


# ===============================================================
# OPINION / REASON / PROMPT 출력 유틸
# ===============================================================
def format_numbers(pred_close, pred_return, last_price, uncertainty_std, confidence):
    pc = None if pred_close is None else round(float(pred_close), 2)
    pr = None if pred_return is None else round(float(pred_return), 4)
    pr_pct = None if pred_return is None else round(float(pred_return) * 100, 2)
    lp = None if last_price is None else round(float(last_price), 2)
    std = None if uncertainty_std is None else round(float(uncertainty_std), 4)
    conf = None if confidence is None else round(float(confidence), 3)
    return pc, pr, pr_pct, lp, std, conf


def build_opinion_reason_prompt(ctx: dict) -> tuple[str, str, str]:
    snap = ctx.get("snapshot", {})
    pred = ctx.get("prediction", {})

    last_price = snap.get("last_price")
    features = snap.get("feature_cols_preview") or []
    pred_close = pred.get("pred_close")
    pred_return = pred.get("pred_return")
    uncertainty_std = (pred.get("uncertainty") or {}).get("std")
    confidence = pred.get("confidence")

    pc, pr, pr_pct, lp, std, conf = format_numbers(
        pred_close, pred_return, last_price, uncertainty_std, confidence
    )

    # --- OPINION ---
    direction = None
    if pr is not None:
        direction = "상승" if pr > 0 else "하락" if pr < 0 else "보합"

    opinion_lines = []
    if direction and pr_pct is not None and pc is not None:
        opinion_lines.append(f"TSLA는 단기적으로 **{direction}(≈ {pr_pct}%)**을 예상합니다.")
        opinion_lines.append(f"모델이 추정한 다음날 종가는 **≈ ${pc}**입니다.")
    elif pc is not None:
        opinion_lines.append(f"TSLA의 다음날 예상 종가는 **≈ ${pc}**입니다.")
    if conf is not None:
        opinion_lines.append(f"신뢰도는 **약 {conf}** 수준입니다.")
    opinion = " ".join(opinion_lines) if opinion_lines else "단기 의견을 생성했습니다."

    # --- REASON ---
    reason_parts = []
    if pr is not None and pc is not None and lp is not None:
        reason_parts.append(f"1) 예측 수익률 {pr}({pr_pct}%), 예측 종가 ${pc} → 현재가 ${lp} 대비 방향성 도출.")
    if std is not None and conf is not None:
        reason_parts.append(f"2) 불확실성 표준편차 ≈ {std}, confidence ≈ {conf}.")
    if features:
        reason_parts.append(f"3) 핵심 피처: {', '.join(features[:8])}.")
    if not reason_parts:
        reason_parts.append("모델 출력과 핵심 지표를 종합해 의견을 도출했습니다.")
    reason = "\n".join(reason_parts)

    # --- PROMPT (LLM용) ---
    ctx_compact = {
        "agent_id": ctx.get("agent_id", "SentimentalAgent"),
        "ticker": ctx.get("ticker", "TSLA"),
        "last_price": lp,
        "pred_return": pr,
        "pred_close": pc,
        "uncertainty_std": std,
        "confidence": conf,
        "features_preview": features[:8],
    }
    prompt = (
        "아래 컨텍스트를 바탕으로 한국어로 3문장 이내의 단기 의견을 작성하시오.\n"
        "- 수치는 반올림하여 자연스럽게 표현하고, 과도한 단정 대신 근거(예측 수익률, 불확실성/신뢰도, 핵심 지표)를 간결히 언급한다.\n"
        "- 최종 문장에는 리스크 코멘트를 한 줄로 덧붙인다.\n"
        f"Context: {ctx_compact}"
    )

    return opinion, reason, prompt


def print_opinion_reason_prompt(ctx: dict) -> None:
    opinion, reason, prompt = build_opinion_reason_prompt(ctx)
    print("\n# OPINION")
    print(opinion)
    print("\n# REASON")
    print(reason)
    print("\n# PROMPT")
    print(prompt)


# ===============================================================
# MAIN 실행
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="TSLA")
    args = parser.parse_args()

    # ----- 에이전트 초기화 -----
    ag = SentimentalAgent(verbose=True)

    # ----- 데이터/예측 -----
    X = ag.searcher(args.ticker)
    current_price = ag.stockdata.last_price
    target = ag.predict(X, current_price=current_price)

    # ----- CTX 구성 -----
    asof_date = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    ctx = {
        "agent_id": ag.agent_id,
        "ticker": args.ticker,
        "snapshot": {
            "asof_date": asof_date,
            "last_price": current_price,
            "currency": "USD",
            "window_size": ag.window_size,
            "feature_cols_preview": (ag.feature_cols or [])[:8],
        },
        "prediction": {
            "pred_close": target.next_close,
            "pred_return": (
                None
                if current_price is None or target.next_close is None
                else (target.next_close / current_price - 1.0)
            ),
            "uncertainty": {
                "std": target.uncertainty,
                "ci95": None,
                "pi80": None,
            },
            "confidence": target.confidence,
        },
    }

    print("=== CTX PREVIEW ===")
    print(json.dumps(ctx, ensure_ascii=False, indent=2))

    # ----- OPINION / REASON / PROMPT 출력 -----
    print_opinion_reason_prompt(ctx)


if __name__ == "__main__":
    main()
