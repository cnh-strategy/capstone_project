from debate_ver3.agents.sentimental_agent import SentimentalAgent

agent = SentimentalAgent(verbose=True, ticker="TSLA")
opinion = agent.reviewer_draft("TSLA")

print(opinion.agent_id)
print(opinion.target.next_close, opinion.target.uncertainty)
print(opinion.reason)
