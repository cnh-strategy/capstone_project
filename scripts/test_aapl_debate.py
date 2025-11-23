
import os
import sys
from datetime import datetime
import re

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from agents.debate_agent import DebateAgent

def format_output(result):
    """
    사용자 요청 포맷에 맞춰 결과 출력
    
    # 최종 결론 : 000.00 (Return: 0.00%)
     ## [최종 결론 및 제언 내용]

    # 토론 내용 
     - SentimentalAgent : 000.00 → 00.00
     - TechnicalAgent : 000.00 → 00.00
     - MacroAgent : 000.00 → 00.00
     ## [토론 요약]
     ## [주요 쟁점]
    """
    ensemble_price = result.get("ensemble_next_close", 0.0)
    last_price = result.get("last_price", 0.0)
    return_pct = ((ensemble_price / last_price) - 1) * 100 if last_price else 0.0
    
    summary_text = result.get("debate_summary", "")
    
    # LLM 요약 텍스트에서 섹션 추출 (정규표현식 활용)
    # 예상 포맷:
    # ## [토론 요약]
    # ...
    # ## [주요 쟁점]
    # ...
    # ## [최종 결론 및 제언]
    # ...
    
    def extract_section(text, header):
        pattern = f"{re.escape(header)}(.*?)(?=\n## |$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    summary_content = extract_section(summary_text, "## [토론 요약]")
    issues_content = extract_section(summary_text, "## [주요 쟁점]")
    conclusion_content = extract_section(summary_text, "## [최종 결론 및 제언]")
    
    # 만약 정규표현식으로 추출이 안되면 전체 출력 (Fallback)
    if not conclusion_content:
        conclusion_content = summary_text

    print("\n" + "="*60)
    print(f"# 최종 결론 : {ensemble_price:.2f} (Return: {return_pct:.2f}%)")
    print("="*60)
    
    print(f"\n ## [최종 결론 및 제언 내용]")
    print(conclusion_content)
    
    print(f"\n# 토론 내용")
    
    # 에이전트별 초기값 -> 최종값 추출 로직 필요
    # 현재 result 딕셔너리에는 최종값만 있음. DebateAgent 객체에 접근해야 초기값을 알 수 있음.
    # 여기서는 편의상 result에 있는 최종값만 표시하거나, DebateAgent를 수정하여 history를 리턴하게 해야 함.
    # 일단은 최종값만이라도 표시. (초기값은 run 로그에 남아있음)
    # 개선: DebateAgent.run() 리턴값에 round_history 추가 필요. 
    # 현재 구조상으로는 최종값만 출력.
    
    agents_res = result.get("agents", {})
    for key, val in agents_res.items():
        agent_name = key.replace("_next_close", "")
        print(f" - {agent_name} : {val:.2f}")
        
    print(f"\n ## [토론 요약]")
    print(summary_content)
    
    print(f"\n ## [주요 쟁점]")
    print(issues_content)
    print("\n")

def main():
    ticker = "AAPL"
    rounds = 1  # 빠른 검증을 위해 1라운드만 실행
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Debate Check for {ticker}...")

    try:
        debate = DebateAgent(ticker=ticker, rounds=rounds)
        
        # DebateAgent.run()은 이제 내부 print를 줄이고 결과를 리턴함
        result = debate.run()
        
        # 초기 의견(Round 0)과 최종 의견(Round N)을 비교하기 위해
        # debate.opinions에 접근
        initial_ops = debate.opinions.get(0, {})
        final_ops = debate.opinions.get(rounds, {})
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Debate Finished.")
        
        # ---- 커스텀 출력 ----
        ensemble_price = result.get("ensemble_next_close", 0.0)
        last_price = result.get("last_price", 0.0)
        return_pct = ((ensemble_price / last_price) - 1) * 100 if last_price else 0.0
        summary_text = result.get("debate_summary", "")

        # 섹션 추출
        def extract_section(text, header):
            pattern = f"{re.escape(header)}(.*?)(?=\n## |$)"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else ""

        summary_content = extract_section(summary_text, "## [토론 요약]")
        issues_content = extract_section(summary_text, "## [주요 쟁점]")
        conclusion_content = extract_section(summary_text, "## [최종 결론 및 제언]") # 또는 "## [최종 결론]" 등 유동적일 수 있음

        # Fallback: 만약 섹션이 안 잡히면 전체 출력
        if not conclusion_content and not summary_content:
            conclusion_content = summary_text
        
        print("\n" + "="*60)
        print(f"# 최종 결론 : {ensemble_price:.2f} (Return: {return_pct:.2f}%)")
        print("="*60)

        print(f"\n ## [최종 결론 및 제언 내용]")
        print(conclusion_content)

        print(f"\n# 토론 내용")
        # 에이전트별 변화 (초기 -> 최종)
        # 순서 고정: Technical, Macro, Sentimental
        agent_order = ["TechnicalAgent", "MacroAgent", "SentimentalAgent"]
        for agent_id in agent_order:
            init_val = initial_ops.get(agent_id).target.next_close if (initial_ops and agent_id in initial_ops) else 0.0
            final_val = final_ops.get(agent_id).target.next_close if (final_ops and agent_id in final_ops) else 0.0
            
            # 만약 final_ops가 비어있다면(Round 1에서 끝났는데 opinons[1]이 없을 수 있음? get_revise에서 갱신함)
            # 혹시 모를 예외 처리
            if final_val == 0.0 and agent_id in result.get("agents", {}):
                 key = f"{agent_id}_next_close"
                 final_val = result["agents"].get(key, 0.0)

            print(f" - {agent_id} : {init_val:.2f} → {final_val:.2f}")

        print(f"\n ## [토론 요약]")
        print(summary_content)

        print(f"\n ## [주요 쟁점]")
        print(issues_content)
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
