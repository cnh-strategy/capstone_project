from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent
from typing import Dict, List
from collections import defaultdict
from core.data_set import build_dataset, load_dataset

class DebateAgent(BaseAgent):
    def __init__(self, rounds: int = 3, ticker: str = None):
        self.agents = {
            "TechnicalAgent": TechnicalAgent("TechnicalAgent", ticker=ticker),
            "FundamentalAgent": FundamentalAgent("FundamentalAgent", ticker=ticker),
            "SentimentalAgent": SentimentalAgent("SentimentalAgent", ticker=ticker),
        }
        self.rounds = rounds
        self.opinions = {}
        self.rebuttals = {}
        self.ticker = ticker

    def get_opinion(self, round: int, ticker: str = None):
        """각 agent의 Opinion(주장) 생성"""
        if not hasattr(self, "opinions"):
            self.opinions = {}

        opinions = {}
        for agent_id, agent in self.agents.items():
            # 1️⃣ 데이터 로드
            X = agent.searcher(ticker)

            # 2️⃣ 예측 수행
            target = agent.predict(X)

            # 3️⃣ Opinion 생성 (LLM Reason 포함)
            opinion = agent.reviewer_draft(agent.stockdata, target)
            opinions[agent_id] = opinion

        self.opinions[round] = opinions
        print(f"✅ Round {round} 의견 수집 완료 ({len(opinions)} agents)")
        return opinions


    def get_rebuttal(self, round: int) -> Dict[str, List[Rebuttal]]:
        """모든 agent 간 상호 rebuttal 수행"""
        round_rebuttals = list()

        opinions = self.opinions[round-1]

        for agent_id, agent in self.agents.items():
            my_opinion = opinions[agent_id]

            # ✅ 각 agent가 자신 외 모든 agent에게 rebuttal 생성
            for other_agent_id, other_opinion in opinions.items():
                if other_agent_id == agent_id:
                    continue

                rebuttal = agent.reviewer_rebut(my_opinion, other_opinion, round)
                round_rebuttals.append(rebuttal)

        self.rebuttals[round] = round_rebuttals
        print(f"✅ Round {round} rebuttals 생성 완료 ({len(round_rebuttals)} agents)")
        return round_rebuttals

    def get_revise(self, round: int):
        """모든 agent 간 상호 revise 수행 및 opinions 갱신"""
        round_revises = {}

        for agent_id, agent in self.agents.items():
            my_opinion = self.opinions[round - 1][agent_id]
            other_opinions = [
                self.opinions[round - 1][other_id]
                for other_id in self.agents.keys()
                if other_id != agent_id
            ]
            rebuttals = [
                r for r in self.rebuttals.get(round, [])
                if getattr(r, "to_agent_id", None) == agent_id
            ]
            stock_data = getattr(agent, "stockdata", None)

            revise = agent.reviewer_revise(
                my_opinion,
                other_opinions,
                rebuttals,
                stock_data
            )

            # revise 결과 opinion 갱신
            round_revises[agent_id] = revise

        # ✅ opinions에 다음 라운드 의견으로 등록
        self.opinions[round] = round_revises
        print(f"✅ Round {round} revise 완료 및 opinions 갱신 ({len(round_revises)} agents)")

        return round_revises


    def run(self):
        build_dataset(self.ticker)
        self.get_opinion(0) 

        for round in range(1, self.rounds + 1):
            self.get_rebuttal(round)
            self.get_revise(round)

    def get_ensemble(self):
        """토론 결과를 바탕으로 ensemble 정보 생성"""
        import statistics
        import yfinance as yf
        
        # 최종 라운드의 의견 가져오기
        final_round = max(self.opinions.keys()) if self.opinions else 0
        final_opinions = self.opinions.get(final_round, {})
        final_points = [float(op.target.next_close) for op in final_opinions.values() if op and op.target]
        
        # 에이전트별 최종 예측가
        agents_data = {}
        for agent_id, opinion in final_opinions.items():
            if opinion and opinion.target:
                agents_data[f"{agent_id}_next_close"] = float(opinion.target.next_close)
        
        # Yahoo Finance에서 현재가 정보 가져오기
        current_price = None
        currency = "USD"
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
            currency = info.get('currency', 'USD')
        except Exception as e:
            print(f"현재가 정보를 가져올 수 없습니다: {e}")
        
        return {
            "ticker": self.ticker,
            "agents": agents_data,
            "mean_next_close": (statistics.fmean(final_points) if final_points else None),
            "median_next_close": (statistics.median(final_points) if final_points else None),
            "currency": currency,
            "last_price": current_price,
        }

    def debate(self, stock_data: StockData, target: Target) -> Opinion:
        return super().reviewer_draft(stock_data, target)

    def rebut(self, my_opinion: Opinion, other_opinion: Opinion) -> Rebuttal:
        return super().reviewer_rebut(my_opinion, other_opinion)