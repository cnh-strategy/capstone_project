# agents/base_agent.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# -----------------------------
# ✅ 공용 데이터 구조 정의
# -----------------------------

@dataclass
class Target:
    """모델이 예측한 다음 종가"""
    next_close: float

@dataclass
class Opinion:
    """에이전트의 의견 (LLM 이전에도 사용 가능)"""
    agent_id: str
    target: Target
    reason: str = ""
    confidence: float = 0.0

@dataclass
class Rebuttal:
    """에이전트 간 반박/지지 기록"""
    from_agent_id: str
    to_agent_id: str
    stance: str  # "REBUT" or "SUPPORT"
    message: str

@dataclass
class StockData:
    """에이전트가 사용하는 기본 데이터 구조"""
    sentimental: Dict = field(default_factory=dict)
    fundamental: Dict = field(default_factory=dict)
    technical: Dict = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: str = "USD"

@dataclass
class RoundLog:
    """각 라운드별 토론 로그"""
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]


# -----------------------------
# ✅ 공용 Agent 추상 클래스
# -----------------------------
class BaseAgent:
    """모든 Agent의 기본 인터페이스 (ML 중심)"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.stockdata: Optional[StockData] = None
        self.model = None
        self.opinions: List[Opinion] = []

    # ---- 데이터 준비 ----
    def searcher(self, ticker: str) -> StockData:
        """데이터 수집"""
        raise NotImplementedError

    # ---- 학습 ----
    def train(self, data):
        """모델 학습"""
        raise NotImplementedError

    # ---- 예측 ----
    def predict(self, features) -> Target:
        """다음 종가 예측"""
        raise NotImplementedError
