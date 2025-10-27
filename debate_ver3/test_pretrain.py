# debate_ver3\test_pretrain.py
# ===============================================================

from debate_ver3.agents.sentimental_agent import SentimentalAgent

if __name__ == "__main__":
    agent = SentimentalAgent(agent_id="SentimentalAgent", verbose=True, ticker="TSLA")
    agent.pretrain()  # 80% 학습, 20% 검증 / 모델, 스케일러 저장
    print("✅ pretrain finished")