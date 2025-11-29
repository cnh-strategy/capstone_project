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
from config.agents import dir_info, agents_info, common_params
from prompts import DEBATE_PROMPTS

from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from openai import OpenAI
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from agents.macro_agent import MacroAgent
from agents.technical_agent import TechnicalAgent
from agents.sentimental_agent import SentimentalAgent

import yfinance as yf
import statistics



# macro_sercher는 더 이상 사용하지 않음 (MacroAgent.searcher()로 대체)
from core.macro_classes.macro_llm import (
     Opinion, Rebuttal,
)

class DebateAgent:
    """
    Multi-Agent Debate System Orchestrator
    여러 에이전트 간의 토론을 조율하여 최종 예측을 생성합니다.
    """

    def __init__(self, ticker: str, rounds: int = 3):
        """
        DebateAgent 초기화

        Args:
            ticker: 분석할 티커 예: "NVDA"
            rounds: 토론 라운드 수
        """
        if not ticker or str(ticker).strip() == "":
            raise ValueError("DebateAgent: ticker must not be None or empty")

        # ---- 1) ticker 정리 ----
        self.ticker = str(ticker).upper()
        self.symbol = self.ticker

        # ---- 2) OpenAI Client 초기화 ----
        load_dotenv()
        self.openai_api_key = os.getenv("CAPSTONE_OPENAI_API")
        self.client = None
        if self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"[WARN] OpenAI Client 초기화 실패: {e}")

        # ---- 3) config 로부터 window_size 가져오기 ----
        macro_cfg = agents_info.get("MacroAgent", {})
        macro_window = macro_cfg.get("window_size", 40)

        # ---- 3) 각 에이전트 생성 ----
        # 각 에이전트에 config에서 gamma, delta_limit 값을 가져와서 설정
        # gamma: 수렴율 (0~1), 값이 클수록 다른 에이전트의 의견을 더 많이 반영
        tech_cfg = agents_info.get("TechnicalAgent", {})
        macro_cfg = agents_info.get("MacroAgent", {})
        sent_cfg = agents_info.get("SentimentalAgent", {})
        
        self.agents = {
            "TechnicalAgent": TechnicalAgent(
                agent_id="TechnicalAgent",
                ticker=self.ticker,
                gamma=tech_cfg.get("gamma", 0.3),  # config에서 가져오기
                delta_limit=tech_cfg.get("delta_limit", 0.05)  # config에서 가져오기
            ),

            "MacroAgent": MacroAgent(
                agent_id="MacroAgent",
                ticker=self.ticker,
                base_date=datetime.today(),
                window=macro_window,
                gamma=macro_cfg.get("gamma", 0.5),  # config에서 가져오기
                delta_limit=macro_cfg.get("delta_limit", 0.1)  # config에서 가져오기
            ),

            "SentimentalAgent": SentimentalAgent(
                ticker=self.ticker,
                agent_id="SentimentalAgent",
                gamma=sent_cfg.get("gamma", 0.3),  # config에서 가져오기
                delta_limit=sent_cfg.get("delta_limit", 0.05)  # config에서 가져오기
            ),
        }

        # ---- 4) Debate metadata ----
        self.rounds = rounds
        self.opinions: Dict[int, Dict[str, Opinion]] = {}
        self.rebuttals: Dict[int, List[Rebuttal]] = {}

        # 데이터셋 생성는 run()에서 Agent들이 자체적으로 함
        self._data_built = False

        # ---- 5) 각 Agent가 사전 학습된 모델이 있으면 미리 로드 ----
        for agent in self.agents.values():
            if hasattr(agent, "_load_model_if_exists"):
                try:
                    agent._load_model_if_exists()
                except Exception as e:
                    print(f"[WARN] {agent.__class__.__name__} 초기 모델 로드 실패 (계속 진행): {e}")

        # ---- 6) Ensemble Model 로드 (초기화 시점엔 로드하지 않고 run 시점에 확인) ----
        import joblib
        self.ensemble_model = None
        # ensemble_path = os.path.join(dir_info["model_dir"], f"{self.ticker}_ensemble_lightgbm.pkl")
        # if os.path.exists(ensemble_path):
        #     try:
        #         self.ensemble_model = joblib.load(ensemble_path)
        #         print(f"[INFO] Ensemble Model 로드 완료: {ensemble_path}")
        #     except Exception as e:
        #         print(f"[WARN] Ensemble Model 로드 실패: {e}")
        # else:
        #     print("[INFO] Ensemble Model 파일이 없습니다. run() 시점에 학습을 시도합니다.")

    def ensure_ensemble_model(self):
        """
        티커별 앙상블 모델이 존재하는지 확인하고, 없거나 오래되었으면 자동으로 학습합니다.
        """
        import joblib
        # 스크립트 모듈 import (함수형 호출)
        try:
            from scripts.gen_training_data import generate_ensemble_data
            from scripts.train_meta_model import train_meta_model
        except ImportError:
            print("[WARN] 앙상블 학습 스크립트를 import할 수 없어 자동 학습을 건너뜁니다.")
            return

        model_filename = f"{self.ticker}_ensemble_lightgbm.pkl"
        model_path = os.path.join(dir_info["model_dir"], model_filename)
        data_path = os.path.join(dir_info["data_dir"], f"{self.ticker}_ensemble_train.csv")

        # 모델이 없으면 학습 시작
        if not os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"[INFO] {self.ticker} 전용 앙상블 모델이 없습니다. 자동 학습을 시작합니다.")
            print(f"{'='*60}")
            
            try:
                # 1. 학습 데이터 생성
                print(f"[Step 1/2] 학습 데이터 생성 중... ({self.ticker})")
                # DebateAgent 내부 에이전트들을 활용하기보다, 스크립트가 독립적으로 수행하도록 함
                # (메모리 관리 및 독립성 위해)
                # days=None이면 config/agents의 period 사용
                generate_ensemble_data(ticker=self.ticker, days=None, output_path=data_path)
                
                # 2. 모델 학습
                print(f"[Step 2/2] LightGBM 모델 학습 중...")
                train_meta_model(data_path=data_path, model_out_path=model_path)
                
                print(f"[INFO] {self.ticker} 앙상블 모델 학습 완료!")
                
            except Exception as e:
                print(f"[ERROR] 앙상블 모델 자동 학습 중 오류 발생: {e}")
                print("[INFO] 기본 평균 방식을 사용합니다.")
                return

        # 모델 로드 시도
        if os.path.exists(model_path):
            try:
                self.ensemble_model = joblib.load(model_path)
                print(f"[INFO] Ensemble Model 로드 완료: {model_path}")
            except Exception as e:
                print(f"[WARN] Ensemble Model 로드 실패: {e}")
                self.ensemble_model = None
        else:
            print("[WARN] 모델 파일이 생성되지 않았습니다.")

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
            # === 1단계: 모델 준비 확인 ===
            is_ready = self._check_agent_ready(agent_id, ticker)
            needs_pretrain = force_pretrain or (not is_ready)

            # === 2단계: 에이전트별 데이터 수집 및 학습 ===
            if agent_id == "SentimentalAgent":
                # SentimentalAgent: pretrain 먼저 → 이후 run_dataset
                if needs_pretrain:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SentimentalAgent] pretrain 실행 (모델/스케일러 생성)")
                    agent.pretrain()
                else:
                    model_path = os.path.join(dir_info["model_dir"], f"{ticker}_{agent_id}.pt")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SentimentalAgent] 기존 모델 사용: {model_path}")
                
                # pretrain 이후 최신 데이터로 run_dataset (중복 방지)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [SentimentalAgent] run_dataset 실행 (최신 데이터 수집)")
                cfg = agents_info.get(agent_id, {})
                # common_params에서 period 가져오기
                period_str = common_params.get("period", "2y")
                # period 문자열을 일수로 변환
                if period_str.endswith("y"):
                    years = int(period_str[:-1])
                    days = years * 365
                elif period_str.endswith("m"):
                    months = int(period_str[:-1])
                    days = months * 30
                elif period_str.endswith("d"):
                    days = int(period_str[:-1])
                else:
                    days = 2 * 365  # 기본값
                sd = agent.run_dataset(days=days)
                agent.stockdata = sd
                
                # 예측 (config에서 n_samples 가져오기)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [SentimentalAgent] predict 실행 (MC Dropout 포함)")
                n_samples = common_params.get("n_samples", 30)
                target = agent.predict(sd, n_samples=n_samples)
            else:
                # Technical/Macro: searcher 먼저 → 필요시 pretrain
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] searcher 실행 (데이터셋 준비)")
                X = agent.searcher(ticker, rebuild=rebuild)
                
                if needs_pretrain:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] pretrain 실행 (모델/스케일러 생성)")
                    agent.pretrain()
                else:
                    model_path = os.path.join(dir_info["model_dir"], f"{ticker}_{agent_id}.pt")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] 기존 모델 사용: {model_path}")
                
                # 예측
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

    def run(self, force_pretrain: bool = False):
        """
        전체 디베이트 프로세스 실행
        
        Args:
            force_pretrain: 초기 Opinion 수집 시 강제 pretrain 여부
        
        프로세스:
        1. (선택) 공통 데이터셋 생성 – 현재는 각 Agent 내부 pretrain/searcher 에서 처리
        2. 앙상블 모델 준비 (없으면 자동 학습)
        3. Round 0: 초기 Opinion 수집
        4. Round 1~N: Rebuttal → Revise 반복
        5. 최종 Ensemble 예측 생성
        """

        if not self._data_built:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터셋 생성은 각 Agent에서 처리하므로 DebateAgent.run에서는 스킵합니다.")
            self._data_built = True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터셋 이미 생성됨, 스킵")

        # 앙상블 모델 준비 (티커별)
        self.ensure_ensemble_model()

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
        전체 토론 과정(Opinion -> Rebuttal -> Revise)을 요약하고 최종 결론을 도출합니다.
        
        Args:
            ensemble_result: get_ensemble()에서 생성된 수치적 결과
            
        Returns:
            str: LLM이 생성한 요약 및 결론 텍스트
        """
        if not self.client:
            return "OpenAI API 키가 설정되지 않아 요약을 생성할 수 없습니다."

        # 1. Transcript 생성
        transcript_lines = []
        
        # Round 0 (Initial Opinions)
        transcript_lines.append("\n## [Round 0: 초기 의견]")
        if 0 in self.opinions:
            for agent_id, op in self.opinions[0].items():
                transcript_lines.append(f"- {agent_id}: 예측가 {op.target.next_close:.2f}, 근거: {op.reason}")
                
        # Round 1 ~ N
        for r in range(1, self.rounds + 1):
            transcript_lines.append(f"\n## [Round {r}: 토론 및 수정]")
            
            # Rebuttals
            if r in self.rebuttals:
                transcript_lines.append("### 반박(Rebuttals):")
                for reb in self.rebuttals[r]:
                    transcript_lines.append(
                        f"  * {reb.from_agent_id} -> {reb.to_agent_id} ({reb.stance}): {reb.message}"
                    )
            
            # Revised Opinions
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
                model="gpt-4o-mini",  # 또는 적절한 모델
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
        토론 결과를 바탕으로 ensemble 정보 생성 (LightGBM Model 적용)
        
        Returns:
            Dict: Ensemble 예측 정보
                - ticker: 종목 코드
                - agents: 에이전트별 예측가 딕셔너리
                - mean_next_close: (모델 미사용시) 평균 예측가
                - median_next_close: (모델 미사용시) 중앙값 예측가
                - ensemble_next_close: (모델 사용시) 최종 모델 예측가
                - currency: 통화 코드
                - last_price: 현재가
        """
        import statistics
        import pandas as pd
        import numpy as np

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
                "ensemble_next_close": None,
                "currency": "USD",
                "last_price": None,
            }

        # 에이전트별 최종 예측값 추출
        # LightGBM 모델 입력 순서: Tech -> Macro -> Senti
        # feature_cols = [
        #     'Tech_Ret', 'Tech_Conf', 'Tech_Unc',
        #     'Macro_Ret', 'Macro_Conf', 'Macro_Unc',
        #     'Senti_Ret', 'Senti_Conf', 'Senti_Unc'
        # ]
        
        tech_op = final_opinions.get("TechnicalAgent")
        macro_op = final_opinions.get("MacroAgent")
        senti_op = final_opinions.get("SentimentalAgent")
        
        # 현재가(Last Price) 가져오기 (우선 에이전트가 가진 정보 활용)
        last_price = None
        
        # 각 에이전트에서 last_price 찾기 시도
        for agent_id, agent in self.agents.items():
            sd = getattr(agent, "stockdata", None)
            if sd and getattr(sd, "last_price", None):
                last_price = float(sd.last_price)
                break
                
        # 없다면 yfinance 호출 (fallback)
        if last_price is None:
            try:
                stock = yf.Ticker(self.ticker)
                info = stock.info
                last_price = info.get('currentPrice', info.get('regularMarketPrice', None))
            except:
                pass
                
        if last_price is None:
             print("[WARN] 현재가(Last Price)를 찾을 수 없어 Ensemble Model을 실행할 수 없습니다.")
        
        # 1. 모델 기반 예측 시도
        ensemble_price = None
        
        if self.ensemble_model and last_price:
            try:
                # 입력 벡터 구성
                def get_feats(op):
                    if not op or not op.target:
                        return np.nan, 0.0, 0.0
                    pred = float(op.target.next_close)
                    ret = (pred - last_price) / last_price
                    conf = float(op.target.confidence or 0.0)
                    unc = float(op.target.uncertainty or 0.0)
                    return ret, conf, unc

                t_ret, t_conf, t_unc = get_feats(tech_op)
                m_ret, m_conf, m_unc = get_feats(macro_op)
                s_ret, s_conf, s_unc = get_feats(senti_op)
                
                # 입력 데이터프레임 (모델 학습시 feature name과 일치해야 함)
                input_df = pd.DataFrame([{
                    'Tech_Ret': t_ret, 'Tech_Conf': t_conf, 'Tech_Unc': t_unc,
                    'Macro_Ret': m_ret, 'Macro_Conf': m_conf, 'Macro_Unc': m_unc,
                    'Senti_Ret': s_ret, 'Senti_Conf': s_conf, 'Senti_Unc': s_unc
                }])
                
                # 예측 (Target_Ret)
                pred_ret = self.ensemble_model.predict(input_df)[0]
                
                # 가격 변환
                ensemble_price = last_price * (1 + pred_ret)
                print(f"[INFO] Ensemble Model Predict: {ensemble_price:.2f} (Return: {pred_ret*100:.2f}%)")
                
            except Exception as e:
                print(f"[WARN] Ensemble Model 예측 중 오류 발생: {e}")
                ensemble_price = None

        # 2. 기존 통계 기반 집계 (Backup)
        final_points = [
            float(op.target.next_close)
            for op in final_opinions.values()
            if op and op.target
        ]
        
        mean_val = statistics.fmean(final_points) if final_points else None
        median_val = statistics.median(final_points) if final_points else None
        
        # 모델 예측이 실패했거나 없으면 평균값 사용
        if ensemble_price is None:
            ensemble_price = mean_val

        # 결과 구성
        agents_data = {}
        for agent_id, opinion in final_opinions.items():
            if opinion and opinion.target:
                agents_data[f"{agent_id}_next_close"] = float(opinion.target.next_close)

        # 기본 결과 딕셔너리
        result = {
            "ticker": self.ticker,
            "agents": agents_data,
            "mean_next_close": mean_val,
            "median_next_close": median_val,
            "ensemble_next_close": ensemble_price,
            "currency": "USD", # 통화는 일단 USD 고정 (개선 가능)
            "last_price": last_price,
        }

        # 3. Debate Summary 생성 (LLM)
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Debate Summary 생성 중...")
        print(f"{'='*80}")
        summary_text = self.summarize_debate(result)
        result["debate_summary"] = summary_text
        # print(summary_text) # DebateAgent에서의 직접 출력은 끔

        return result
