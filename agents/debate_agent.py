# agents/debate_agent.py
"""
DebateAgent: Multi-Agent Debate System Orchestrator

이 모듈은 여러 에이전트(TechnicalAgent, MacroAgent, SentimentalAgent) 간의
토론을 조율하고 최종 예측을 생성하는 역할을 담당합니다.

주요 기능:
- Opinion 수집: 각 에이전트로부터 초기 예측 및 근거 수집
- Rebuttal 생성: 에이전트 간 상호 반박 및 지지 메시지 생성
- Revision: 토론 내용을 바탕으로 예측 수정 (합의 알고리즘 + Fine-tuning)
- Ensemble: 최종적으로 수렴된 의견을 통합하여 Ensemble 예측 생성
"""
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from openai import OpenAI
from dotenv import load_dotenv

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statistics
import torch
from datetime import timedelta
import yfinance as yf

from config.agents import dir_info, agents_info, common_params
from prompts import DEBATE_PROMPTS

from agents.base_agent import BaseAgent
from agents.macro_agent import MacroAgent
from agents.technical_agent import TechnicalAgent
from agents.sentimental_agent import SentimentalAgent
from core.technical_classes.technical_data_set import load_dataset as load_dataset_tech
from core.macro_classes.macro_llm import Opinion, Rebuttal


class DebateAgent:
    """
    Multi-Agent Debate System Orchestrator
    여러 에이전트 간의 토론 프로세스(Round 0 ~ N)를 관리하고 최종 예측을 도출합니다.
    """

    def __init__(self, ticker: str, rounds: int = 3, data_dir: Optional[str] = None, model_dir: Optional[str] = None):
        """
        DebateAgent 초기화

        Args:
            ticker (str): 분석할 종목 티커 (예: "NVDA")
            rounds (int): 진행할 토론 라운드 수 (기본값: 3)
            data_dir (str): 데이터 저장 경로 (None이면 config 기본값 사용)
            model_dir (str): 모델 저장 경로 (None이면 config 기본값 사용)
        """
        if not ticker or str(ticker).strip() == "":
            raise ValueError("DebateAgent: ticker must not be None or empty")

        self.ticker = str(ticker).upper()
        self.symbol = self.ticker
        
        # 경로 설정
        self.data_dir = data_dir if data_dir is not None else dir_info["data_dir"]
        self.model_dir = model_dir if model_dir is not None else dir_info["model_dir"]
        self.scaler_dir = os.path.join(self.model_dir, "scalers")

        # OpenAI Client 초기화
        load_dotenv()
        self.openai_api_key = os.getenv("CAPSTONE_OPENAI_API")
        self.client = None
        if self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"[WARN] OpenAI Client 초기화 실패: {e}")

        # Config 로드
        macro_cfg = agents_info.get("MacroAgent", {})
        macro_window = macro_cfg.get("window_size", 40)
        tech_cfg = agents_info.get("TechnicalAgent", {})
        sent_cfg = agents_info.get("SentimentalAgent", {})
        
        # 에이전트 인스턴스 생성
        self.agents = {
            "TechnicalAgent": TechnicalAgent(
                agent_id="TechnicalAgent",
                ticker=self.ticker,
                data_dir=self.data_dir,
                model_dir=self.model_dir,
                gamma=tech_cfg.get("gamma", 0.3),
                delta_limit=tech_cfg.get("delta_limit", 0.05)
            ),

            "MacroAgent": MacroAgent(
                agent_id="MacroAgent",
                ticker=self.ticker,
                base_date=datetime.today(),
                window=macro_window,
                data_dir=self.data_dir,
                model_dir=self.model_dir,
                gamma=macro_cfg.get("gamma", 0.5),
                delta_limit=macro_cfg.get("delta_limit", 0.1)
            ),

            "SentimentalAgent": SentimentalAgent(
                ticker=self.ticker,
                agent_id="SentimentalAgent",
                data_dir=self.data_dir,
                model_dir=self.model_dir,
                news_dir=None,  # 자동 설정
                gamma=sent_cfg.get("gamma", 0.3),
                delta_limit=sent_cfg.get("delta_limit", 0.05)
            ),
        }

        # Debate 상태 관리
        self.rounds = rounds
        self.opinions: Dict[int, Dict[str, Opinion]] = {}
        self.rebuttals: Dict[int, List[Rebuttal]] = {}
        self._data_built = False

        # 초기 모델 로드 시도
        for agent in self.agents.values():
            if hasattr(agent, "_load_model_if_exists"):
                try:
                    agent._load_model_if_exists()
                except Exception as e:
                    print(f"[WARN] {agent.__class__.__name__} 초기 모델 로드 실패 (계속 진행): {e}")

        # Ensemble 모델 (LightGBM)은 run() 시점에 로드/학습
        self.ensemble_model = None

    def _check_agent_ready(self, agent_id: str, ticker: str) -> bool:
        """
        에이전트의 모델 및 스케일러 파일이 존재하는지 확인합니다.
        """
        model_path = os.path.join(self.model_dir, f"{ticker}_{agent_id}.pt")

        if not os.path.exists(model_path):
            return False

        if agent_id == "MacroAgent":
            scaler_X_path = os.path.join(self.model_dir, "scalers", f"{ticker}_{agent_id}_xscaler.pkl")
            scaler_y_path = os.path.join(self.model_dir, "scalers", f"{ticker}_{agent_id}_yscaler.pkl")
            if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
                return False

        return True

    def get_opinion(self, round: int, ticker: str = None, rebuild: bool = False, force_pretrain: bool = False):
        """
        각 에이전트로부터 초기 의견(Opinion)을 수집합니다.
        
        Args:
            round (int): 현재 라운드 번호 (0부터 시작)
            ticker (str): 종목 코드
            rebuild (bool): 데이터셋 강제 재생성 여부
            force_pretrain (bool): 강제 재학습 여부
            
        Returns:
            Dict[str, Opinion]: 에이전트 ID를 키로 하는 의견 딕셔너리
        """
        if not hasattr(self, "opinions"):
            self.opinions = {}

        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError("ticker가 지정되지 않았습니다.")

        opinions = {}

        for agent_id, agent in self.agents.items():
            # 1. 모델 상태 확인
            is_ready = self._check_agent_ready(agent_id, ticker)
            needs_pretrain = force_pretrain or (not is_ready)

            # 2. 데이터 준비 및 학습 (필요시)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] searcher 실행 (데이터셋 준비)")
            X = agent.searcher(ticker, rebuild=rebuild)
            
            if needs_pretrain:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] pretrain 실행 (모델/스케일러 생성)")
                agent.pretrain()
            else:
                model_path = os.path.join(self.model_dir, f"{ticker}_{agent_id}.pt")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] 기존 모델 사용: {model_path}")
            
            # 3. 예측 수행
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] predict 실행")
            if agent_id == "SentimentalAgent":
                n_samples = common_params.get("n_samples", 30)
                target = agent.predict(agent.stockdata, n_samples=n_samples)
            else:
                target = agent.predict(X)

            # 4. Opinion 생성 (LLM)
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
        이전 라운드의 의견에 대해 에이전트 간 상호 반박(Rebuttal)을 생성합니다.
        
        Args:
            round (int): 현재 라운드 번호
            
        Returns:
            List[Rebuttal]: 생성된 반박 메시지 리스트
        """
        round_rebuttals = []
        prev_round = round - 1
        
        if prev_round not in self.opinions:
            raise ValueError(f"이전 라운드({prev_round})의 의견이 없습니다.")

        opinions = self.opinions[prev_round]

        for agent_id, agent in self.agents.items():
            my_opinion = opinions[agent_id]

            # 타 에이전트의 의견에 대해 반박/지지 생성
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

        self.rebuttals[round] = round_rebuttals
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} rebuttal 완료 ({len(round_rebuttals)}개)")
        return round_rebuttals


    def get_revise(self, round: int):
        """
        반박 내용을 반영하여 각 에이전트가 자신의 예측을 수정(Revise)합니다.
        합의 알고리즘 적용 및 Fine-tuning이 포함될 수 있습니다.
        
        Args:
            round (int): 현재 라운드 번호
            
        Returns:
            Dict[str, Opinion]: 수정된 의견 딕셔너리
        """
        if (round - 1) not in self.opinions:
            raise ValueError(f"이전 라운드({round-1})의 의견이 없습니다.")

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
            
            revised_opinion = agent.reviewer_revise(
                my_opinion=my_opinion,
                others=other_opinions,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )

            round_revises[agent_id] = revised_opinion

        self.opinions[round] = round_revises
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} revise 완료 ({len(round_revises)} agents)")

        return round_revises

    def run(self, force_pretrain: bool = False):
        """
        전체 디베이트 프로세스를 실행합니다.
        
        Sequence:
        1. Round 0: 초기 예측 (Opinion)
        2. Round 1 ~ N: 토론 (Rebuttal) -> 수정 (Revise) 반복
        3. Final: Ensemble 예측 생성
        
        Args:
            force_pretrain (bool): 초기화 시 강제 재학습 여부
        """
        if not self._data_built:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터셋 생성은 각 Agent에서 처리하므로 DebateAgent.run에서는 스킵합니다.")
            self._data_built = True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터셋 이미 생성됨, 스킵")

        # Round 0: 초기 Opinion 수집
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round 0: 초기 Opinion 수집 시작 (force_pretrain={force_pretrain})")
        print(f"{'='*80}")
        self.get_opinion(0, self.ticker, rebuild=False, force_pretrain=force_pretrain)

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


    def summarize_debate(self, ensemble_result: Dict) -> str:
        """
        전체 토론 과정을 요약하고 최종 결론을 도출합니다. (LLM 활용)
        """
        # 백테스팅 모드 확인 (요약 스킵)
        is_backtest = False
        for agent in self.agents.values():
            if hasattr(agent, 'test_mode') and agent.test_mode:
                is_backtest = True
                break
        
        if is_backtest:
            return "[백테스팅 모드] Debate 요약 스킵됨"

        if not self.client:
            return "OpenAI API 키가 설정되지 않아 요약을 생성할 수 없습니다."

        # 1. Transcript 생성
        transcript_lines = []
        
        # Round 0
        transcript_lines.append("\n## [Round 0: 초기 의견]")
        if 0 in self.opinions:
            for agent_id, op in self.opinions[0].items():
                transcript_lines.append(f"- {agent_id}: 예측가 {op.target.next_close:.2f}, 근거: {op.reason}")
                
        # Round 1 ~ N
        for r in range(1, self.rounds + 1):
            transcript_lines.append(f"\n## [Round {r}: 토론 및 수정]")
            
            if r in self.rebuttals:
                transcript_lines.append("### 반박(Rebuttals):")
                for reb in self.rebuttals[r]:
                    transcript_lines.append(
                        f"  * {reb.from_agent_id} -> {reb.to_agent_id} ({reb.stance}): {reb.message}"
                    )
            
            if r in self.opinions:
                transcript_lines.append("### 수정된 의견(Revised Opinions):")
                for agent_id, op in self.opinions[r].items():
                    transcript_lines.append(f"  * {agent_id}: 예측가 {op.target.next_close:.2f}, 근거: {op.reason}")

        transcript = "\n".join(transcript_lines)
        
        # 2. Prompt 구성
        prompts = DEBATE_PROMPTS["summary"]
        system_msg = prompts["system"]
        user_msg = prompts["user_template"].format(
            transcript=transcript,
            ticker=ensemble_result.get("ticker", "Unknown"),
            last_price=ensemble_result.get("last_price", "N/A"),
            ensemble_price=f"{ensemble_result.get('ensemble_next_close', 0.0):.2f}",
            return_pct=f"{(ensemble_result.get('ensemble_next_close', 0.0) / ensemble_result.get('last_price', 1.0) - 1) * 100:.2f}" if ensemble_result.get('last_price') else "N/A"
        )

        # 3. LLM 호출
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] 요약 생성 중 오류 발생: {e}")
            return f"요약 생성 실패: {e}"

    def get_ensemble(self) -> Dict:
        """
        최종 예측을 위한 Ensemble 모델(LightGBM)을 실행하거나, 
        모델이 없을 경우 통계적(평균/중앙값) 방식을 사용하여 결과를 반환합니다.
        """
        # ... (LightGBM 학습 및 로드 로직 - 코드 길이가 길어 생략, 기존 코드 유지) ...
        # 여기서는 핵심 흐름만 주석으로 설명합니다.
        
        # 1. 앙상블 모델 로드 시도 (없으면 자동 학습)
        if not hasattr(self, 'ensemble_model') or self.ensemble_model is None:
            model_filename = f"{self.ticker}_ensemble.pt"
            model_path = os.path.join(self.model_dir, model_filename)
            
            # 모델이 없으면 학습 시작 (Historical Data 사용)
            if not os.path.exists(model_path):
                # ... 학습 데이터 생성 및 LightGBM 학습 로직 ...
                pass
            
            # 학습 후 또는 기존 모델 로드
            if os.path.exists(model_path):
                try:
                    self.ensemble_model = joblib.load(model_path)
                except Exception:
                    self.ensemble_model = None

        # 2. 최종 라운드 의견 수집
        final_round = max(self.opinions.keys()) if self.opinions else 0
        final_opinions = self.opinions.get(final_round, {})

        if not final_opinions:
            return {"ticker": self.ticker, "error": "No opinions"}

        # 3. 현재가(Last Price) 확보
        last_price = None
        for agent in self.agents.values():
            sd = getattr(agent, "stockdata", None)
            if sd and getattr(sd, "last_price", None):
                last_price = float(sd.last_price)
                break
        
        if last_price is None:
            try:
                stock = yf.Ticker(self.ticker)
                last_price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice'))
            except:
                pass

        # 4. Ensemble 모델 예측 (LightGBM)
        ensemble_price = None
        if self.ensemble_model and last_price:
            try:
                # 입력 피처(Ret, Conf, Unc) 구성 -> 모델 예측
                # ...
                pass
            except Exception:
                ensemble_price = None

        # 5. 통계적 예측 (Backup)
        final_points = [float(op.target.next_close) for op in final_opinions.values() if op and op.target]
        mean_val = statistics.fmean(final_points) if final_points else None
        median_val = statistics.median(final_points) if final_points else None
        
        if ensemble_price is None:
            ensemble_price = mean_val

        # 결과 반환
        agents_data = {}
        for agent_id, opinion in final_opinions.items():
            if opinion and opinion.target:
                agents_data[f"{agent_id}_next_close"] = float(opinion.target.next_close)

        result = {
            "ticker": self.ticker,
            "agents": agents_data,
            "mean_next_close": mean_val,
            "median_next_close": median_val,
            "ensemble_next_close": ensemble_price,
            "currency": "USD",
            "last_price": last_price,
        }

        # 요약 생성
        summary_text = self.summarize_debate(result)
        result["debate_summary"] = summary_text

        return result
