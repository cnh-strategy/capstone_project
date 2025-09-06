import os
import json
import requests
import statistics
from typing import List, Tuple, Union, Optional
from .base_agent import BaseAgent   # BaseAgent 상속

class StrategyAgent(BaseAgent):
    """
    세 개의 에이전트(Valuation / Sentiment / Event)의 결과를 종합하여
    최종 매수/매도가와 이유를 제시하는 에이전트.
    """

    Triple = Union[List[Union[int, float, str]], Tuple[Union[int, float, str], ...]]
    Parsed = Tuple[float, float, str]

    def __init__(
        self,
        use_llm_reason: bool = True,          # True면 LLM으로 reason 요약
        model: Optional[str] = None,
        preferred_models: Optional[List[str]] = None,
        temperature: float = 0.1,
        round_decimals: Optional[int] = None  # None이면 원래 스케일 유지
    ):
        super().__init__(model=model, preferred_models=preferred_models, temperature=temperature)

        self.use_llm_reason = use_llm_reason and bool(os.getenv("CAPSTONE_OPENAI_API"))
        self.round_decimals = round_decimals

    # ---------- Public ----------
    def run(self, valuation_out: Triple, sentimental_out: Triple, event_out: Triple) -> List[Union[float, str]]:
        v = self._parse(valuation_out)
        s = self._parse(sentimental_out)
        e = self._parse(event_out)

        # 1) 숫자 집계 (중앙값)
        buys  = [v[0], s[0], e[0]]
        sells = [v[1], s[1], e[1]]

        final_buy  = self._median_clean(buys)
        final_sell = self._median_clean(sells)

        # 2) 정합성 보장 (sell ≥ buy)
        if final_sell < final_buy:
            final_sell = final_buy

        # 3) 옵션: 반올림
        if self.round_decimals is not None:
            final_buy  = round(final_buy,  self.round_decimals)
            final_sell = round(final_sell, self.round_decimals)

        # 4) reason 생성
        final_reason = self._compose_reason(
            final_buy, final_sell,
            reasons={"valuation": v[2], "sentimental": s[2], "event": e[2]},
            proposals={"valuation": (v[0], v[1]), "sentimental": (s[0], s[1]), "event": (e[0], e[1])}
        )

        return [final_buy, final_sell, final_reason]

    # ---------- Parsing / Utils ----------
    def _parse(self, x: Triple) -> Parsed:
        if not isinstance(x, (list, tuple)) or len(x) < 3:
            raise ValueError("각 에이전트 출력은 [buy, sell, reason] 형식이어야 합니다.")
        buy, sell, reason = x[0], x[1], x[2]
        buy_f  = float(buy)
        sell_f = float(sell)
        reason_s = str(reason) if reason is not None else ""
        return (buy_f, sell_f, reason_s)

    def _median_clean(self, arr: List[float]) -> float:
        nums = [float(x) for x in arr if self._is_num(x)]
        if not nums:
            raise ValueError("집계할 숫자 데이터가 없습니다.")
        try:
            return float(statistics.median(nums))
        except Exception:
            return float(sum(nums) / len(nums))

    @staticmethod
    def _is_num(x) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    # ---------- Reason Composition ----------
    def _compose_reason(self, buy: float, sell: float, reasons: dict, proposals: dict) -> str:
        base_reason = (
            f"세 에이전트의 제안을 종합해 매수 {buy}, 매도 {sell}으로 결정했습니다.\n"
            f"- 평가(Valuation): {proposals['valuation'][0]}~{proposals['valuation'][1]} — {self._shorten(reasons['valuation'])}\n"
            f"- 심리(Sentiment): {proposals['sentimental'][0]}~{proposals['sentimental'][1]} — {self._shorten(reasons['sentimental'])}\n"
            f"- 이벤트(Event): {proposals['event'][0]}~{proposals['event'][1]} — {self._shorten(reasons['event'])}"
        )

        if not self.use_llm_reason:
            return base_reason

        try:
            return self._llm_reason(buy, sell, reasons, proposals)
        except Exception:
            return base_reason

    @staticmethod
    def _shorten(text: str, maxlen: int = 500) -> str:
        t = (text or "").strip().replace("\n", " ")
        return t if len(t) <= maxlen else (t[:maxlen].rstrip() + "…")

    # ---------- LLM: reason 요약 ----------
    def _llm_reason(self, buy: float, sell: float, reasons: dict, proposals: dict) -> str:
        schema_obj = {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "최종 매수/매도 수치(buy/sell)에 부합하는 간결한 한국어 근거 4~5문장"}
            },
            "required": ["reason"],
            "additionalProperties": False
        }

        sys = (
            "너는 세 개의 에이전트(Valuation/Sentiment/Event)의 제안을 종합한 전략가다. "
            "아래 제공된 최종 수치(buy/sell)는 그대로 유지하며, 세 에이전트의 핵심 논거를 합쳐 "
            "숫자와 논리가 일치하는 4~5문장의 간결한 한국어 요약만 생성한다. "
            "출력은 JSON 객체로만 반환하고 키는 reason 하나만 포함한다."
        )
        user = {
            "final_numbers": {"buy": buy, "sell": sell},
            "valuation": {"proposal": {"buy": proposals["valuation"][0], "sell": proposals["valuation"][1]},
                          "reason": reasons["valuation"]},
            "sentimental": {"proposal": {"buy": proposals["sentimental"][0], "sell": proposals["sentimental"][1]},
                            "reason": reasons["sentimental"]},
            "event": {"proposal": {"buy": proposals["event"][0], "sell": proposals["event"][1]},
                      "reason": reasons["event"]},
        }

        body_base = {
            "input": [
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "StrategyReason",
                    "strict": True,
                    "schema": schema_obj
                }
            },
            "temperature": self.temperature
        }

        last_err = None
        for m in self.preferred_models:
            body = dict(body_base, model=m)
            r = requests.post(self.OPENAI_URL, json=body, headers=self.headers, timeout=60)
            if r.ok:
                data = r.json()
                txt = data.get("output_text") or data["output"][0]["content"][0]["text"]
                obj = json.loads(txt)
                return obj["reason"]
            if r.status_code in (400, 404):
                last_err = (r.status_code, r.text)
                continue
            r.raise_for_status()

        raise RuntimeError(f"All models failed. Last: {last_err}")