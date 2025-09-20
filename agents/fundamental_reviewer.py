import json
from base_agent import BaseAgent

class FunReviewer(BaseAgent):
    """
    Reviewer:
    - Round1: FundamentalAgent(FMP+Yahoo) 결과 검증 + CER 설명
    - Round2: Sentiment, Event, Valuation과 비교 → 최종 합의
    """

    def __init__(self, use_llm=False, **kwargs):
        super().__init__(**kwargs)
        self.use_llm = use_llm

    # -----------------------------
    # Round1
    # -----------------------------
    def review_round1(self, fmp_result: dict, yahoo_result: dict):
        claim = f"FMP 예측={fmp_result.get('opinion')}, Yahoo 예측={yahoo_result.get('opinion')}"
        evidence = f"FMP 종가={fmp_result.get('predicted_price'):.2f}, Yahoo 종가={yahoo_result.get('predicted_price'):.2f}"
        rebuttal = "양 소스 간 차이가 크면 신뢰도 낮음으로 표시"
        confidence = (fmp_result.get("confidence", 0) + yahoo_result.get("confidence", 0)) / 2

        result = {
            "claim": claim,
            "evidence": evidence,
            "rebuttal": rebuttal,
            "final_opinion": fmp_result.get("opinion") if fmp_result.get("opinion") == yahoo_result.get("opinion") else "Hold",
            "confidence": confidence
        }
        return result

    # -----------------------------
    # Round2
    # -----------------------------
    def review_round2(self, agents_result: dict, prev_reviewer: dict):
        opinions = {k: v.get("opinion") if isinstance(v, dict) else v for k, v in agents_result.items()}
        agreements = {k: 1 if v == prev_reviewer["final_opinion"] else 0 for k, v in opinions.items()}

        claim = f"다른 에이전트 의견={opinions}"
        evidence = f"동의율={agreements}"
        rebuttal = "불일치하는 근거는 요약 필요"
        confidence = (sum(agreements.values()) + prev_reviewer["confidence"]) / (len(agreements) + 1)

        return {
            "claim": claim,
            "evidence": evidence,
            "rebuttal": rebuttal,
            "final_opinion": prev_reviewer["final_opinion"],
            "confidence": confidence
        }
