from datetime import datetime

def run_debate_rounds(agents, ticker: str, max_rounds=3):
    """LLM 기반 Debate → Rebuttal → Revision"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 💬 Stage 3: Debate Start ({ticker})")
    
    for r in range(1, max_rounds + 1):
        print(f"\n🧭 Round {r}")

        # 1️⃣ 초안 작성
        opinions = {a.agent_id: a.reviewer_draft(ticker) for a in agents.values()}
        
        # 첫 번째 Agent의 stock_data를 공용으로 사용
        first_agent = list(agents.values())[0]
        stock_data = first_agent.stockdata

        # 2️⃣ 반박 생성
        rebuttals = {
            a.agent_id: a.reviewer_rebut(r, opinions[a.agent_id], opinions, stock_data)
            for a in agents.values()
        }

        # 3️⃣ 수정 의견 반영
        revised = {}
        for a in agents.values():
            new_op = a.reviewer_revise(opinions[a.agent_id], opinions, rebuttals[a.agent_id], stock_data)
            a.apply_revision(new_op)
            revised[a.agent_id] = new_op.target.next_close

        # 4️⃣ 합의 출력
        avg_pred = sum(revised.values()) / len(revised)
        print(f"📊 Consensus After Round {r}: {avg_pred:.2f}")
    print(f"✅ Debate Finished.\n")
