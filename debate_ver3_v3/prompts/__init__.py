# v3 전용 프롬프트 로더 (파일이 없으면 기본값 사용)
from pathlib import Path
import json

def _load_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

# 기본(system/user) 프롬프트 (파일 미존재 시 폴백)
DEFAULT_SYSTEM = (
    "너는 감성/뉴스 기반 단기 주가 분석가다. "
    "아래 ctx 지표만 사용하고 외부추정은 금지한다. "
    "의견은 간결하되 수치/플래그 등 데이터 근거를 포함하고, "
    "trial→검토→최종 3단계 요약 로그를 남긴다."
)
DEFAULT_USER = (
    "[CONTEXT]\n{context}\n\n"
    "[TASK]\n"
    "아래 스키마(JSON)로만 답하라:\n"
    "{schema}\n\n"
    "[CONSTRAINTS]\n"
    "- drivers/contradictions에는 ctx의 실제 키명을 1개 이상 포함\n"
    "- surprise_proxy=true면 과잉반응 가능성 문구 포함\n"
    "- 장황한 chain 금지, 시도/수정은 요약만"
)

OUTPUT_SCHEMA_JSON = {
  "opinion": "상승/하락/보합 + 한 줄 근거",
  "evidence": {
    "drivers": ["ctx의 실제 키명을 포함한 근거(최대3)"],
    "contradictions": ["모순/리스크(최대2)"],
    "data_notes": ["플래그/제약(있으면)"]
  },
  "uncertainty": {"level": "low|medium|high", "pi80": [0, 0]},
  "attempts": [
    {"step":"초안","why":"핵심 지표","edit":"보완점"},
    {"step":"검토","why":"리스크 반영","edit":"확신도 조정"},
    {"step":"최종","why":"균형 판단","edit":"결론 확정"}
  ]
}

# 로컬 txt에서 system/user를 읽어오되, 없으면 DEFAULT 사용
BASE = Path(__file__).parent
system_txt = _load_text(BASE / "sentimental_v3.system.txt") or DEFAULT_SYSTEM
user_txt   = _load_text(BASE / "sentimental_v3.user.txt") or DEFAULT_USER

# 에이전트별 프롬프트 딕셔너리
OPINION_PROMPTS_V3 = {
    "SentimentalAgentV3": {
        "system": system_txt,
        "user":   user_txt,
        "schema": json.dumps(OUTPUT_SCHEMA_JSON, ensure_ascii=False)
    }
}
