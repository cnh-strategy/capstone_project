# scripts/v3_demo/demo_ctx_prompt_v3.py

import json
import argparse
from pathlib import Path
import sys, os

# 프로젝트 루트 import 경로
sys.path.append(os.path.abspath("."))

# v3 에이전트 임포트
from debate_ver3_v3.agents.sentimental_agent_v3 import SentimentalAgentV3

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def build_agent(agent_name: str, ticker: str):
    """
    단일 파일 내에서는 SentimentalAgentV3만 사용하지만,
    --agent 인자를 받아 확장 가능하게 구성.
    """
    name = (agent_name or "SentimentalAgentV3").strip()
    if name == "SentimentalAgentV3":
        return SentimentalAgentV3(ticker=ticker)
    # 향후 추가 에이전트 분기 지점
    raise ValueError(f"Unknown agent: {name}")


def load_ctx(ticker: str, agent_name: str):
    ag = build_agent(agent_name, ticker=ticker)
    prev = ag.preview_opinion_ctx()
    ctx  = prev.get("ctx_preview") or {}
    sysm = prev.get("system_msg", "")
    usrm = prev.get("user_msg", "")
    return ctx, sysm, usrm


def mock_llm_reply(ctx: dict) -> dict:
    # ctx에 들어있는 실제 키를 근거로 간단 목업 응답 생성
    pf = ctx.get("price_features") or {}
    vf = ctx.get("volume_features") or {}
    nf = ctx.get("news_features") or {}

    drivers = []
    if pf.get("trend_7d"): drivers.append("price_features.trend_7d↑")
    if nf.get("trend_7d"): drivers.append("news_features.trend_7d↑")
    if vf.get("volume_spike"): drivers.append("volume_features.volume_spike")

    contradictions = []
    if pf.get("rolling_vol_20"):
        contradictions.append("price_features.rolling_vol_20↑(변동성 리스크)")

    pi80 = (ctx.get("prediction") or {}).get("uncertainty", {}).get("pi80")

    return {
        "opinion": "보합~소폭 상승. 추세/감성 개선이나 변동성 유의.",
        "evidence": {
            "drivers": drivers[:3] or ["(no-strong-driver)"],
            "contradictions": contradictions[:2],
            "data_notes": (ctx.get("explain_helpers") or {}).get("constraint_flags", [])
        },
        "uncertainty": {"level": "medium", "pi80": pi80},
        "attempts": [
            {"step":"초안","why":"trend/sentiment↑","edit":"volume_spike 반영"},
            {"step":"검토","why":"volatility↑","edit":"확신도↓"},
            {"step":"최종","why":"지표 균형","edit":"보합~상승 범위"}
        ]
    }


def evidence_coverage(reply: dict, ctx: dict):
    # drivers/contradictions 문구에 실제 ctx 키명이 들어갔는지 간단 점수
    keys = set()
    for sec in ("price_features", "volume_features", "news_features", "regime_features"):
        keys |= set((ctx.get(sec) or {}).keys())
    # prediction 블록의 대표 키 몇 개 추가
    keys |= {"pred_return", "calibrated_prob_up", "mc_std", "pred_close", "confidence"}

    def score(items):
        if not items: 
            return 1.0
        hit = 0
        for s in items:
            txt = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
            for k in keys:
                if k in txt:
                    hit += 1
                    break
        return round(hit / max(1, len(items)), 3)

    ev = reply.get("evidence", {})
    return score(ev.get("drivers", [])), score(ev.get("contradictions", []))


def parse_args():
    parser = argparse.ArgumentParser()
    # 단수/복수 모두 지원
    parser.add_argument("--ticker", type=str, help="단일 티커")
    parser.add_argument("--tickers", nargs="+", help="티커 리스트")
    parser.add_argument("--agent", type=str, default="SentimentalAgentV3", help="Agent class name")
    parser.add_argument("--mock", action="store_true", help="실제 LLM 대신 목업 응답 사용")
    parser.add_argument("--save-ab", action="store_true", help="간단 비교 CSV 저장")
    return parser.parse_args()


def normalize_tickers(args) -> list:
    if args.tickers and args.ticker:
        # 둘 다 있으면 병합
        return list(dict.fromkeys([args.ticker] + args.tickers))
    if args.tickers:
        return args.tickers
    if args.ticker:
        return [args.ticker]
    # 기본값
    return ["TSLA", "AAPL"]


def main():
    args = parse_args()
    tickers = normalize_tickers(args)

    rows = []
    for tk in tickers:
        ctx, sysm, usrm = load_ctx(tk, agent_name=args.agent)
        
    # === VALIDATION & LOGGING (ctx 전후비교/서프라이즈/근거 힌트) ===
    from pathlib import Path
    ART = Path("artifacts"); ART.mkdir(exist_ok=True)

    # 1) surprise_proxy 규칙 점검 (shock_z>=1.5 or volume_spike)
    nf = ctx.get("news_features", {}) or {}
    vf = ctx.get("volume_features", {}) or {}
    surprise_proxy = (nf.get("shock_z") is not None and nf.get("shock_z") >= 1.5) or bool(vf.get("volume_spike"))
    print(f"[VALID] surprise_proxy? {surprise_proxy}  (shock_z={nf.get('shock_z')}, volume_spike={vf.get('volume_spike')})")

    # 2) before/after 전후값 스냅샷 저장
    pred = ctx.get("prediction", {}) or {}
    ba = {
        "last_price": (ctx.get("snapshot", {}) or {}).get("last_price"),
        "pred_close": pred.get("pred_close"),
        "pred_return": pred.get("pred_return"),
        "direction_delta": ("UP" if (pred.get("pred_return") or 0) > 0
                            else "DOWN" if (pred.get("pred_return") or 0) < 0 else "FLAT"),
        "prob_up": pred.get("calibrated_prob_up"),
    }
    (ART / f"before_after_{ctx.get('ticker')}_v3.json").write_text(
        json.dumps(ba, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[VALID] before_after:", ba)

    # 3) drivers/contradictions 자동 힌트 생성 & 저장
    pf = ctx.get("price_features", {}) or {}
    vf = ctx.get("volume_features", {}) or {}
    nf = ctx.get("news_features", {}) or {}

    suggested_drivers = []
    # 가격 지표
    if pf.get("zscore_close_20") is not None and abs(pf["zscore_close_20"]) >= 1.5:
        suggested_drivers.append({"key":"price_features.zscore_close_20","value":pf["zscore_close_20"],"reason":"평균대비 이탈 확대"})
    if pf.get("trend_7d") is not None and abs(pf["trend_7d"]) >= 0.02:
        suggested_drivers.append({"key":"price_features.trend_7d","value":pf["trend_7d"],"reason":"최근 1주 추세 유효"})
    # 뉴스/감성 지표
    if nf.get("trend_7d") is not None and abs(nf["trend_7d"]) >= 0.1:
        suggested_drivers.append({"key":"news_features.trend_7d","value":nf["trend_7d"],"reason":"뉴스 감성의 뚜렷한 방향성"})
    if nf.get("shock_z") is not None and nf["shock_z"] >= 1.5:
        suggested_drivers.append({"key":"news_features.shock_z","value":nf["shock_z"],"reason":"뉴스 쇼크로 단기 과잉반응 가능"})
    # 거래량 지표
    if vf.get("volume_spike"):
        suggested_drivers.append({"key":"volume_features.volume_spike","value":True,"reason":"수급 과열 신호"})

    suggested_counter = []
    if pf.get("rolling_vol_20") is not None and pf["rolling_vol_20"] >= 0.03:
        suggested_counter.append({"key":"price_features.rolling_vol_20","value":pf["rolling_vol_20"],"risk":"변동성 확대 리스크"})
    # 필요시 추가: drawdown, zscore 음의 과매도 등…

    hint = {"drivers": suggested_drivers[:3], "counter": suggested_counter[:2]}
    (ART / f"suggested_evidence_{ctx.get('ticker')}_v3.json").write_text(
        json.dumps(hint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[HINT] suggested drivers/counter saved →", f"artifacts/suggested_evidence_{ctx.get('ticker')}_v3.json")

    # 4) (선택) mock 응답 커버리지 찍기 — 스크립트에 mock_llm_reply, evidence_coverage가 이미 있다면:
    try:
        reply = mock_llm_reply(ctx)
        d_cov, c_cov = evidence_coverage(reply, ctx)
        print(f"[COVERAGE] drivers={d_cov:.3f}, counter={c_cov:.3f}, tokens={len(json.dumps(reply, ensure_ascii=False))}")
    except NameError:
        pass





        # ctx / 프롬프트 원본 저장
        (ART/f"ctx_preview_{tk}_v3.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
        (ART/f"prompt_{tk}_system_v3.txt").write_text(sysm, encoding="utf-8")
        (ART/f"prompt_{tk}_user_v3.txt").write_text(usrm, encoding="utf-8")

        # LLM 결과 (목업은 --mock이 있을 때만)
        reply = mock_llm_reply(ctx) if args.mock else {}
        (ART/f"opinion_{tk}_v3.json").write_text(json.dumps(reply, ensure_ascii=False, indent=2), encoding="utf-8")

        d_cov, c_cov = evidence_coverage(reply, ctx) if reply else (0.0, 0.0)
        rows.append({
            "ticker": tk,
            "drivers_cov": d_cov,
            "contrad_cov": c_cov,
            "tokens": len(json.dumps(reply, ensure_ascii=False)) if reply else 0
        })

    if args.save_ab:
        lines = ["ticker,drivers_cov,contrad_cov,tokens\n"] + [
            f"{r['ticker']},{r['drivers_cov']},{r['contrad_cov']},{r['tokens']}\n" for r in rows
        ]
        (ART/"ab_compare_v3.csv").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
