# agent 로드
from agents.base_agent import BaseAgent, Target, Opinion, Rebuttal, RoundLog, StockData
from agents.fundamental_agent import FundamentalAgent
from agents.technical_agent import TechnicalAgent
from agents.sentimental_agent import SentimentalAgent
# from agents.strategy_agent import StrategyAgent

from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
import statistics

class Debate:
    """
    Multi-Agent Debate Orchestrator
    - 세 개 에이전트(Sentimental / Technical / Fundamental)를 기반으로
      초안 → 반박/지지 → 수정 라운드를 진행
    - 각 라운드의 스냅샷을 RoundLog로 보관하고, 최종 집계를 반환
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],  # {"SentimentalAgent": obj, "TechnicalAgent": obj, "FundamentalAgent": obj}
        verbose: bool = False,
    ):
        if not agents or len(agents) < 2:
            raise ValueError("두 개 이상의 에이전트가 필요합니다.")
        self.agents: Dict[str, BaseAgent] = agents
        self.verbose = verbose
        self.logs: List[RoundLog] = []

    # -------------------------
    # 내부 유틸
    # -------------------------
    def _p(self, msg: str):
        if self.verbose:
            print(f"[Debate] {msg}\n")

    def _latest_opinions(self) -> Dict[str, Opinion]:
        """각 에이전트의 최신 Opinion만 뽑아 Dict로."""
        out = {}
        for aid, ag in self.agents.items():
            if ag.opinions:
                out[aid] = ag.opinions[-1]
        return out

    def _choose_common_snapshot(self) -> Optional[StockData]:
        """
        공용 StockData 스냅샷 선택:
        - last_price가 존재하는 에이전트를 우선
        - 없으면 첫 번째 에이전트의 스냅샷
        """
        cand = None
        for ag in self.agents.values():
            if getattr(ag, "stockdata", None) is not None:
                if getattr(ag.stockdata, "last_price", None) is not None:
                    return ag.stockdata
                cand = cand or ag.stockdata
        return cand

    def _summarize(self) -> Dict[str, Target]:
        """현재 최신 Opinion을 {agent_id: Target} 딕셔너리로 요약."""
        latest = self._latest_opinions()
        return {aid: op.target for aid, op in latest.items()}

    # -------------------------
    # 실행 플로우
    # -------------------------
    def run(self, ticker: str, rounds: int = 1) -> Tuple[List[RoundLog], Dict]:
        """
        Debate 실행
        - 0단계: 각 에이전트 초안 생성
        - 1~R단계: 라운드 반복(반박/지지 → 수정)
        - 결과: (라운드 로그 목록, 최종 집계 dict)
        """
        if rounds < 1:
            raise ValueError("rounds는 1 이상이어야 합니다.")

        self._p(f"Start debate on ticker={ticker}, rounds={rounds}")

        # 0) 각 에이전트 초안(Opinion) 생성
        for aid, agent in self.agents.items():
            op = agent.reviewer_draft(ticker)
            self._p(f"draft[{aid}]: next_close={op.target.next_close}")

        # 공용 스냅샷 선택
        common_sd = self._choose_common_snapshot()

        # 라운드 수행
        for r in range(1, rounds + 1):
            self._p(f"=== Round {r} ===")

            # 최신 의견 스냅샷
            latest = self._latest_opinions()

            # 1) 반박/지지 생성
            all_rebuttals: List[Rebuttal] = []
            for my_id, agent in self.agents.items():
                my_latest = latest.get(my_id)
                if not my_latest:
                    continue
                others_latest = {oid: op for oid, op in latest.items() if oid != my_id}

                # 각 에이전트가 타인에 대한 Rebuttal을 생성
                rbts = agent.reviewer_rebut(
                    round_num=r,
                    my_lastest=my_latest,
                    others_latest=others_latest,
                    stock_data=common_sd,
                )
                # BaseAgent.reviewer_rebut는 해당 라운드의 리스트를 반환
                all_rebuttals.extend(rbts or [])

            # 2) 반박/지지 반영하여 수정
            #    타겟 에이전트별로 받은 rebuttal 묶어주기
            received_by_agent: Dict[str, List[Rebuttal]] = {aid: [] for aid in self.agents.keys()}
            for rb in all_rebuttals:
                if rb.to_agent_id in received_by_agent:
                    received_by_agent[rb.to_agent_id].append(rb)

            # 수정 실행
            for my_id, agent in self.agents.items():
                my_latest = latest.get(my_id)
                if not my_latest:
                    continue
                others_latest = {oid: op for oid, op in latest.items() if oid != my_id}
                recv = received_by_agent.get(my_id, [])

                revised = agent.reviewer_revise(
                    my_lastest=my_latest,
                    others_latest=others_latest,
                    received_rebuttals=recv,
                    stock_data=common_sd,
                )
                self._p(f"revise[{my_id}]: next_close={revised.target.next_close}")

            # 라운드 로그 적재
            round_log = RoundLog(
                round_no=r,
                opinions=[self.agents[aid].opinions[-1] for aid in self.agents.keys() if self.agents[aid].opinions],
                rebuttals=all_rebuttals,
                summary=self._summarize(),
            )
            self.logs.append(round_log)

        # 최종 집계
        latest = self._latest_opinions()
        final_points = [float(op.target.next_close) for op in latest.values() if op and op.target]
        ensemble = {
            "ticker": ticker,
            "agents": {aid: float(op.target.next_close) for aid, op in latest.items()},
            "mean_next_close": (statistics.fmean(final_points) if final_points else None),
            "median_next_close": (statistics.median(final_points) if final_points else None),
            "currency": (common_sd.currency if common_sd else None),
            "last_price": (float(common_sd.last_price) if (common_sd and common_sd.last_price is not None) else None),
        }

        self._p(f"Ensemble mean={ensemble['mean_next_close']} median={ensemble['median_next_close']}")
        return self.logs, ensemble
    
def format_debate_summary(
    logs: list[RoundLog],
    final: dict,
    order_hint: list[str] | None = None,
) -> str:
    """
    라운드별로
      □ round N :
      - sentimental : 목표가 / 이유
      - technical   : 목표가 / 이유
      - fundamental : 목표가 / 이유

    마지막에는
      □ 결론
      - 목표가
      - 이유

    형식으로 문자열을 만들어 반환.
    """
    # 에이전트 표시명 정렬 힌트 (원하는 순서)
    order_hint = order_hint or ["SentimentalAgent", "TechnicalAgent", "FundamentalAgent"]

    # 에이전트 → 표시명 매핑 (소문자 키)
    def display_name(agent_id: str) -> str:
        lid = agent_id.lower()
        if "sentiment" in lid:
            return "sentimental"
        if "technical" in lid:
            return "technical"
        if "fundamental" in lid:
            return "fundamental"
        return agent_id  # fallback

    # 정렬 키: order_hint 기준, 없으면 알파벳
    def sort_key(aid: str) -> tuple[int, str]:
        try:
            return (order_hint.index(aid), aid)
        except ValueError:
            return (len(order_hint), aid)

    lines: list[str] = []

    # --- 라운드별 출력 ---
    for rl in logs:
        lines.append(f"□ round {rl.round_no} :")
        # 최신 의견 리스트를 에이전트 ID로 정렬
        ops = {op.agent_id: op for op in rl.opinions}
        for aid in sorted(ops.keys(), key=sort_key):
            op = ops[aid]
            name = display_name(aid)
            target = op.target.next_close
            reason = op.reason.strip().replace("\n", " ")
            lines.append(f"- {name} : {target} / {reason}")
        lines.append("")  # 라운드 간 빈 줄
        lines.append("----")
        lines.append("")

    # --- 결론(앙상블) ---
    # 최종 라운드의 의견들 중 'median'에 가장 가까운 에이전트의 reason 사용
    last_ops = {op.agent_id: op for op in (logs[-1].opinions if logs else [])}
    median_val = final.get("median_next_close")
    pick_reason = ""
    if last_ops and median_val is not None:
        closest_aid = min(
            last_ops.keys(),
            key=lambda aid: abs(float(last_ops[aid].target.next_close) - float(median_val))
        )
        pick_reason = (last_ops[closest_aid].reason or "").strip().replace("\n", " ")

    lines.append("□ 결론 ")
    # 목표가는 median 기준(없으면 mean)
    target_final = median_val if median_val is not None else final.get("mean_next_close")
    if target_final is None:
        target_final = "N/A"
    lines.append(f" - 목표가 : {target_final}")
    lines.append(f" - 이유 : {pick_reason if pick_reason else '에이전트 의견을 종합한 결과(중앙값 기준)'}")

    return "\n".join(lines)

if __name__ == "__main__":
    # 필요한 에이전트 import
    from agents.sentimental_agent import SentimentalAgent
    from agents.technical_agent import TechnicalAgent
    from agents.fundamental_agent import FundamentalAgent

    # 사용자 입력 받기
    ticker = input("분석할 티커를 입력하세요 (예: PLTR, AAPL, TSLA): ").strip().upper()
    try:
        rounds = int(input("라운드 수를 입력하세요 (기본=1): ").strip() or "1")
    except ValueError:
        rounds = 1

    # 에이전트 생성
    agents = {
        "SentimentalAgent": SentimentalAgent(agent_id="SentimentalAgent", verbose=True),
        "TechnicalAgent":   TechnicalAgent(agent_id="TechnicalAgent", verbose=True),
        "FundamentalAgent": FundamentalAgent(agent_id="FundamentalAgent", verbose=True),
    }

    # Debate 실행
    d = Debate(agents, verbose=True)
    logs, final = d.run(ticker=ticker, rounds=rounds)

    # 결과 요약 출력
    print("\n=== Debate 결과 요약 ===")
    print(format_debate_summary(logs, final))
    
    # 시각화 옵션
    try:
        from visualization import DebateVisualizer
        
        visualize = input("\n시각화를 생성하시겠습니까? (y/n): ").strip().lower()
        if visualize in ['y', 'yes', '예', 'ㅇ']:
            visualizer = DebateVisualizer()
            
            print("\n=== 시각화 옵션 ===")
            print("1. 라운드별 의견 변화")
            print("2. 의견 일치도 분석") 
            print("3. 반박/지지 네트워크")
            print("4. 투자의견 표")
            print("5. 주식 컨텍스트")
            print("6. 인터랙티브 대시보드")
            print("7. 전체 리포트 생성")
            
            choice = input("선택하세요 (1-7, all=전체): ").strip().lower()
            
            if choice == '1':
                visualizer.plot_round_progression(logs, final)
            elif choice == '2':
                visualizer.plot_consensus_analysis(logs, final)
            elif choice == '3':
                visualizer.plot_rebuttal_network(logs)
            elif choice == '4':
                visualizer.plot_opinion_table(logs, final)
            elif choice == '5':
                visualizer.plot_stock_context(ticker)
            elif choice == '6':
                visualizer.create_interactive_dashboard(logs, final, ticker)
            elif choice == '7' or choice == 'all':
                visualizer.generate_report(logs, final, ticker)
            else:
                print("잘못된 선택입니다.")
                
    except ImportError:
        print("\n시각화 모듈을 찾을 수 없습니다. visualization.py 파일을 확인하세요.")
    except Exception as e:
        print(f"\n시각화 생성 중 오류 발생: {e}")