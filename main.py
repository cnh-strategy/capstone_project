# agent 로드
from agents.macro_agent import FundamentalAgent
from agents.valuation_agent import ValuationAgent
from agents.event_agent import EventAgent
from agents.sentimental_agent import SentimentalAgent
from agents.strategy_agent import StrategyAgent

# 필요 라이브러리 로드
import os
from dotenv import load_dotenv


# .env 파일 로드 (환경변수 로드)
load_dotenv()

# 환경 변수에서 API Key 및 기타 기본 설정 불러오기
OPENAI_API_KEY = os.getenv("CAPSTONE_OPENAI_API")
OPENAI_URL = "https://api.openai.com/v1/responses"
UA = "Mozilla/5.0"  # User-Agent 헤더

ALPHA_API_KEY = os.getenv("ALPHA_API_KEY")


if __name__ == "__main__":
    v_agent = ValuationAgent(model=None)
    e_agent = EventAgent(model=None)
    s_agent = SentimentalAgent(model=None)
    final_agent = StrategyAgent(use_llm_reason=True, model=None, round_decimals=None)
    fundamental_agent = FundamentalAgent(ALPHA_API_KEY, check_years=3, use_llm=True)
    tickers = input("조회할 Ticker를 입력하세요 (공백으로 구분): ").split()

    for tkr in tickers:
        try:
            v_result = v_agent.run(tkr)
            e_result = e_agent.run(tkr)
            s_result = s_agent.run(tkr)

            open_price = float(input(f"{tkr}에 대한 장 시작가격을 float으로 입력하세요: "))
            close_price = float(input(f"{tkr}에 대한 현재 가격을 float으로 입력하세요: "))
            fundamental_result = fundamental_agent.run(tkr,open_price, close_price)

            final_result = final_agent.run(v_result, s_result, fundamental_result)
            
            print(f"\n=== {tkr} 예측 결과 ===")
            print(f"매수가: {final_result[0]}")
            print(f"매도가: {final_result[1]}")
            print(f"사유: {final_result[2]}")

        except Exception as e:
            print(f"{tkr} 처리 중 오류 발생: {e}")
