# agents/debate_agent.py
import os
from config.agents import dir_info

from datetime import datetime
from typing import Dict, List
from collections import defaultdict

from agents.base_agent import BaseAgent
from agents.macro_agent import MacroPredictor
from agents.technical_agent import TechnicalAgent
from agents.sentimental_agent import SentimentalAgent

import yfinance as yf
import statistics

# 테크니컬 별칭으로 따로 구분
from core.technical_classes.technical_data_set import (
    build_dataset as build_dataset_tech, 
    load_dataset as load_dataset_tech,
)

from core.data_set import build_dataset, load_dataset
from core.macro_classes.macro_funcs import macro_sercher


class DebateAgent(BaseAgent):
    def __init__(self, rounds: int = 3, ticker: str = None):
        self.agents = {
            "TechnicalAgent": TechnicalAgent(agent_id = "TechnicalAgent", ticker=ticker),
            "MacroSentiAgent": MacroPredictor(agent_id= "MacroSentiAgent", ticker=ticker,
                                              base_date=datetime.today(),
                                              window=40),
            "SentimentalAgent": SentimentalAgent(ticker=ticker),
        }
        self.rounds = rounds
        self.opinions = {}
        self.rebuttals = {}
        self.ticker = ticker


    def get_opinion(self, round: int, ticker: str = None, rebuild: bool = True, force_pretrain: bool = True):
        """각 agent의 Opinion(주장) 생성"""
        if not hasattr(self, "opinions"):
            self.opinions = {}

        ticker = ticker or self.ticker
        opinions = {}
        X_scaled = None
        pred_prices = None

        for agent_id, agent in self.agents.items():
            # === Macro: macro_sercher → m_predictor → macro_reviewer_draft ===
            if agent_id == 'MacroSentiAgent':
                print(f"{agent_id}의 데이터 로드.. macro_sercher")
                X, X_scaled = macro_sercher(agent, ticker)

                print(f"{agent_id}의 예측")
                pred_prices, target = agent.m_predictor(X)   #macro_4_predictor(self, macro_sub, X_seq) 로 묶어둠

                print("[MacroSentiAgent] LLM (macro_reviewer_draft)")
                _, opinion = agent.macro_reviewer_draft(X_scaled, pred_prices, target)  #llm_starter(X_scaled, pred_prices, target)


            elif agent_id == 'TechnicalAgent':
                # === Technical: searcher → (조건부) pretrain → predict → reviewer_draft ===
                print("[TechnicalAgent] searcher 실행")
                X = agent.searcher(ticker, rebuild=True)
                model_path = agent.model_path()
                # 모델 가중치 확인 후 필요시 학습
                model_path = os.path.join(dir_info["model_dir"], f"{ticker}_{agent_id}.pt")
                if force_pretrain or (not os.path.exists(model_path)):
                    print("[TechnicalAgent] pretrain 실행")
                    agent.pretrain()
                else:
                    print(f"[TechnicalAgent] 기존 모델 사용: {model_path}")

                print("[TechnicalAgent] predict 실행")
                target = agent.predict(X)

                print("[TechnicalAgent] reviewer_draft 실행")
                opinion = agent.reviewer_draft(agent.stockdata, target)

            
            else: 
                # === Sentimental: searcher → predict → reviewer_draft ===
                print("[SentimentalAgent] searcher 실행")
                X = agent.searcher(ticker)      # base_agent에 존재 - 리턴: X_tensor

                print("[SentimentalAgent] predict 실행")
                target = agent.predict(X)

                print("[SentimentalAgent] reviewer_draft 실행")
                opinion = agent.reviewer_draft(agent.stockdata, target)

            
            opinions[agent_id] = opinion
            try:
                print(f"  - {agent_id}: next_close={opinion.target.next_close:.4f}")
            except Exception:
                pass    
        
        self.opinions[round] = opinions
        print(f" Round {round} 의견 수집 완료 ({len(opinions)} agents)")
        return opinions


    def get_rebuttal(self, round: int):
        """모든 agent 간 상호 rebuttal 수행"""
        round_rebuttals = list()

        opinions = self.opinions[round-1]

        for agent_id, agent in self.agents.items():
            my_opinion = opinions[agent_id]

            # 각 agent가 자신 외 모든 agent에게 rebuttal 생성
            for other_agent_id, other_opinion in opinions.items():
                if other_agent_id == agent_id:
                    continue

                rebuttal = agent.reviewer_rebut(my_opinion, other_opinion, round)
                round_rebuttals.append(rebuttal)

        self.rebuttals[round] = round_rebuttals
        print(f" Round {round} rebuttals 생성 완료 ({len(round_rebuttals)} agents)")
        return round_rebuttals


    # 각 에이전트가 토론 이후 자신의 예측을 수정하는 단계
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

            # BaseAgent의 reviewer_revise() 시그니처에 맞게 호출
            revised_opinion = agent.reviewer_revise(
                my_opinion=my_opinion,
                others=other_opinions,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )

            # revise 결과 opinion 갱신
            round_revises[agent_id] = revised_opinion

        # opinions에 다음 라운드 의견으로 등록
        self.opinions[round] = round_revises
        print(f" Round {round} revise 완료 및 opinions 갱신 ({len(round_revises)} agents)")

        return round_revises

    def run_dataset(self):      #[메크로 테스트용]테스트 후 삭제필요
        build_dataset(self.ticker)

    def run(self):
        build_dataset(self.ticker)      #매크로는 MacroSentimentAgentDataset 활용 (함수:macro_dataset)
        self.get_opinion(0, self.ticker)

        for round in range(1, self.rounds + 1):
            self.get_rebuttal(round)
            self.get_revise(round)
            print(f" Round {round} 토론 완료")

        print(self.get_ensemble())  # 최종 결과 출력


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

