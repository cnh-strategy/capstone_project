# agents/debate_agent.py
"""
DebateAgent: Multi-Agent Debate System Orchestrator

이 모듈은 여러 에이전트(TechnicalAgent, MacroAgent, SentimentalAgent) 간의
토론을 조율하고 최종 예측을 생성합니다.

주요 기능:
- Opinion 수집: 각 에이전트의 초기 예측 수집
- Rebuttal 생성: 에이전트 간 상호 반박/지지 메시지 생성
- Revision: 토론 후 예측 수정
- Ensemble: 최종 통합 예측 생성
"""
import os
from config.agents import dir_info, agents_info

from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from agents.base_agent import BaseAgent
from agents.macro_agent import MacroAgent
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
# macro_sercher는 더 이상 사용하지 않음 (MacroAgent.searcher()로 대체)
from core.macro_classes.macro_llm import (
    LLMExplainer, Opinion, Rebuttal,
    GradientAnalyzer,
)

class DebateAgent(BaseAgent):
    """
    Multi-Agent Debate System Orchestrator

    여러 에이전트 간의 토론을 조율하여 최종 예측을 생성합니다.

    Attributes:
        agents: 에이전트 딕셔너리 (TechnicalAgent, MacroAgent, SentimentalAgent)
        rounds: 토론 라운드 수
        opinions: 라운드별 의견 딕셔너리 {round: {agent_id: Opinion}}
        rebuttals: 라운드별 반박 딕셔너리 {round: [Rebuttal]}
        ticker: 분석 대상 종목 코드
        _data_built: 데이터셋 생성 여부 플래그
    """

    def __init__(self, rounds: int = 3, ticker: str | None = None):
        """
        DebateAgent 초기화

        Args:
            rounds: 토론 라운드 수 (기본값: 3)
            ticker: 분석 대상 종목 코드
        """
        # Config에서 window_size 가져오기
        macro_window = agents_info.get("MacroAgent", {}).get("window_size", 40)

        self.agents = {
            "TechnicalAgent": TechnicalAgent(agent_id="TechnicalAgent", ticker=ticker),
            "MacroAgent": MacroAgent(
                agent_id="MacroAgent",
                ticker=ticker,
                base_date=datetime.today(),
                window=macro_window,  # Config에서 가져옴
            ),
            "SentimentalAgent": SentimentalAgent(ticker=ticker),
        }
        self.rounds = rounds
        self.opinions: Dict[int, Dict[str, Opinion]] = {}
        self.rebuttals: Dict[int, List[Rebuttal]] = {}
        self.ticker = ticker
        self._data_built = False  # 데이터셋 생성 여부 플래그

        # 각 에이전트별로 "있으면" 모델을 사전 로드
        for agent in self.agents.values():
            if hasattr(agent, "_load_model_if_exists"):
                try:
                    agent._load_model_if_exists()
                except Exception as e:
                    print(f"[WARN] {agent.__class__.__name__} 초기 모델 로드 실패 (계속 진행): {e}")

    def _check_agent_ready(self, agent_id: str, ticker: str) -> bool:
        """
        에이전트가 준비되었는지 확인 (모델 및 스케일러 파일 존재 여부)
        
        Args:
            agent_id: 에이전트 ID
            ticker: 종목 코드
            
        Returns:
            bool: 에이전트가 준비되었으면 True, 아니면 False
        """
        model_path = os.path.join(dir_info["model_dir"], f"{ticker}_{agent_id}.pt")

        # 모델 파일 확인
        if not os.path.exists(model_path):
            return False

        # MacroAgent는 별도 스케일러 파일 확인
        if agent_id == "MacroAgent":
            scaler_X_path = os.path.join(dir_info["model_dir"], "scalers", f"{ticker}_{agent_id}_xscaler.pkl")
            scaler_y_path = os.path.join(dir_info["model_dir"], "scalers", f"{ticker}_{agent_id}_yscaler.pkl")
            if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
                return False

        # 다른 Agent들도 스케일러 확인 (필요시)
        # TechnicalAgent와 SentimentalAgent는 BaseAgent의 scaler를 사용하므로
        # 별도 파일 체크는 선택적

        return True

    def get_opinion(self, round: int, ticker: str = None, rebuild: bool = False, force_pretrain: bool = False):
        """
        각 agent의 Opinion(주장) 생성
        
        Args:
            round: 라운드 번호
            ticker: 종목 코드 (None이면 self.ticker 사용)
            rebuild: 데이터셋 재생성 여부 (기본값: False)
            force_pretrain: 강제 pretrain 실행 여부 (기본값: False)
            
        Returns:
            Dict[str, Opinion]: 에이전트별 Opinion 딕셔너리
        """
        if not hasattr(self, "opinions"):
            self.opinions = {}

        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError("ticker가 지정되지 않았습니다.")

        opinions = {}

        for agent_id, agent in self.agents.items():
            # === 모든 에이전트: (조건부) pretrain → searcher → predict → reviewer_draft ===
            # 통일된 체크 메서드 사용
            is_ready = self._check_agent_ready(agent_id, ticker)
            needs_pretrain = force_pretrain or (not is_ready)

            if needs_pretrain:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] pretrain 실행 (모델/스케일러 없음)")
                agent.pretrain()
            else:
                model_path = os.path.join(dir_info["model_dir"], f"{ticker}_{agent_id}.pt")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] 기존 모델 사용: {model_path}")

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] searcher 실행")
            X = agent.searcher(ticker, rebuild=rebuild)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] predict 실행")
            target = agent.predict(X)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] reviewer_draft 실행")
            opinion = agent.reviewer_draft(agent.stockdata, target)


            opinions[agent_id] = opinion
            try:
                print(f"  - {agent_id}: next_close={opinion.target.next_close:.4f}")
            except Exception:
                pass

        self.opinions[round] = opinions
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} 의견 수집 완료 ({len(opinions)} agents)")
        return opinions


    def get_rebuttal(self, round: int):
        """
        모든 agent 간 상호 rebuttal 수행
        
        Args:
            round: 라운드 번호
            
        Returns:
            List[Rebuttal]: 생성된 Rebuttal 리스트
            
        Raises:
            ValueError: 이전 라운드의 opinion이 없는 경우
        """
        round_rebuttals = []

        # 이전 라운드의 opinion을 사용 (round=1이면 opinions[0] 사용)
        prev_round = round - 1
        if prev_round not in self.opinions:
            raise ValueError(
                f"get_rebuttal(round={round}) 호출 전에 "
                f"get_opinion(round={prev_round}) 이(가) 먼저 호출되어야 합니다."
            )

        opinions = self.opinions[prev_round]  # 이전 라운드의 opinion 사용

        for agent_id, agent in self.agents.items():
            my_opinion = opinions[agent_id]

            # 나 이외의 에이전트들에 대해 rebuttal 작성
            for other_id, other_op in opinions.items():
                if other_id == agent_id:
                    continue

                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] → [{other_id}] rebuttal 생성 중...")
                rebut = agent.reviewer_rebuttal(
                    my_opinion=my_opinion,
                    other_opinion=other_op,
                    round_index=round,
                )
                round_rebuttals.append(rebut)

        # 필요하면 저장
        self.rebuttals[round] = round_rebuttals
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} rebuttal 완료 ({len(round_rebuttals)}개)")
        return round_rebuttals


    def get_revise(self, round: int):
        """
        모든 agent 간 상호 revise 수행 및 opinions 갱신
        
        각 에이전트가 토론(rebuttal) 이후 자신의 예측을 수정합니다.
        
        Args:
            round: 라운드 번호
            
        Returns:
            Dict[str, Opinion]: 수정된 Opinion 딕셔너리
            
        Raises:
            ValueError: 이전 라운드의 opinion이 없는 경우
        """
        if (round - 1) not in self.opinions:
            raise ValueError(
                f"get_revise(round={round}) 호출 전에 "
                f"get_opinion(round={round-1}) 이(가) 먼저 호출되어야 합니다."
            )

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

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] revise 실행 중...")
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} revise 완료 및 opinions 갱신 ({len(round_revises)} agents)")

        return round_revises

    def run_dataset(self):
        """
        데이터셋 생성 (테스트용)
        
        주의: 이 메서드는 테스트용이며, 일반적으로 run()에서 자동으로 처리됩니다.
        """
        build_dataset(self.ticker)
        self._data_built = True

    def run(self):
        """
        전체 디베이트 프로세스 실행
        
        프로세스:
        1. 데이터셋 생성 (필요시)
        2. Round 0: 초기 Opinion 수집
        3. Round 1~N: Rebuttal → Revise 반복
        4. 최종 Ensemble 예측 생성
        """
        # 데이터셋 생성 (중복 방지)
        if not self._data_built:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터셋 생성 중...")
            build_dataset(self.ticker)
            self._data_built = True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터셋 이미 생성됨, 스킵")

        # Round 0: 초기 Opinion 수집
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round 0: 초기 Opinion 수집 시작")
        print(f"{'='*80}")
        self.get_opinion(0, self.ticker, rebuild=False, force_pretrain=False)

        # Round 1~N: Rebuttal → Revise 반복
        for round in range(1, self.rounds + 1):
            print(f"\n{'='*80}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} 시작")
            print(f"{'='*80}")

            self.get_rebuttal(round)
            self.get_revise(round)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} 토론 완료")

        # 최종 Ensemble 예측
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 최종 Ensemble 예측")
        print(f"{'='*80}")
        ensemble_result = self.get_ensemble()
        print(ensemble_result)

        return ensemble_result


    def get_ensemble(self) -> Dict:
        """
        토론 결과를 바탕으로 ensemble 정보 생성
        
        Returns:
            Dict: Ensemble 예측 정보
                - ticker: 종목 코드
                - agents: 에이전트별 예측가 딕셔너리
                - mean_next_close: 평균 예측가
                - median_next_close: 중앙값 예측가
                - currency: 통화 코드
                - last_price: 현재가
                
        Note:
            현재는 dict를 반환하지만, 향후 Opinion 객체로 변경 가능
        """
        import statistics
        import yfinance as yf

        # 최종 라운드의 의견 가져오기
        final_round = max(self.opinions.keys()) if self.opinions else 0
        final_opinions = self.opinions.get(final_round, {})

        if not final_opinions:
            print("[WARN] 최종 의견이 없습니다.")
            return {
                "ticker": self.ticker,
                "agents": {},
                "mean_next_close": None,
                "median_next_close": None,
                "currency": "USD",
                "last_price": None,
            }

        final_points = [
            float(op.target.next_close)
            for op in final_opinions.values()
            if op and op.target
        ]

        if not final_points:
            print("[WARN] 유효한 예측값이 없습니다.")
            return {
                "ticker": self.ticker,
                "agents": {},
                "mean_next_close": None,
                "median_next_close": None,
                "currency": "USD",
                "last_price": None,
            }

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
            print(f"[WARN] 현재가 정보를 가져올 수 없습니다: {e}")

        return {
            "ticker": self.ticker,
            "agents": agents_data,
            "mean_next_close": statistics.fmean(final_points),
            "median_next_close": statistics.median(final_points),
            "currency": currency,
            "last_price": current_price,
        }
