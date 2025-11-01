# core\sentimental_classes\features.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


FEATURE_COLUMNS = [
    "sentiment_mean",  # 최근 감성 평균
    "sentiment_vol",   # 감성 변동성
    "news_count_7d",   # 최근 7일 뉴스 건수
    "turnover_rate",   # 회전율 (있다면)
]

USER_LABELS = {
    "returns": "최근 주가 흐름",
    "sentiment_mean": "최근 뉴스/커뮤니티 분위기",
    "sentiment_vol": "기사 톤의 들쭉날쭉함",
    "news_count_7d": "최근 7일 이슈 빈도",
    "turnover_rate": "거래 활발도",
}


def build_feature_frame(ticker: str) -> pd.DataFrame:
    # TODO: 가격/뉴스 피처들을 조합해 단일 DF 반환 (기존 파이프라인 호출)
    # 여긴 최소 동작용 스텁: 실제 구현으로 교체
    idx = pd.date_range(end=pd.Timestamp.today(), periods=120, freq="B")
    df = pd.DataFrame(index=idx)
    rng = np.random.RandomState(42)

    df["returns"] = rng.normal(0, 0.01, size=len(idx))
    df["sentiment_mean"] = rng.uniform(-0.5, 0.5, size=len(idx))
    df["sentiment_vol"] = rng.uniform(0.0, 1.0, size=len(idx))
    df["news_count_7d"] = rng.randint(0, 8, size=len(idx))
    df["turnover_rate"] = rng.uniform(0.2, 1.5, size=len(idx))

    return df


def user_reason_labels() -> Dict[str, str]:
    return USER_LABELS.copy()


def _score_window(window_df: pd.DataFrame) -> List[Tuple[str, float]]:
    """사용자 노출용 이유 점수.
    복잡한 모델 중요도 대신 최근 윈도우 통계(절대 변화/추세/수준)를 간단 점수로 산출.
    """
    scores: List[Tuple[str, float]] = []
    for col in FEATURE_COLUMNS:
        if col not in window_df.columns:
            continue
        s = window_df[col].dropna()
        if s.empty:
            continue
        # (예시) 절대 평균 + 추세 기여의 합으로 간단 점수화
        level = float(s.tail(5).mean())
        trend = float((s.tail(5).iloc[-1] - s.tail(5).iloc[0]))
        weight = abs(level) + abs(trend)
        scores.append((col, weight))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def pick_top_reason_texts(window_df: pd.DataFrame, label_map: Dict[str, str]) -> List[Tuple[str, float]]:
    ranked = _score_window(window_df)
    results: List[Tuple[str, float]] = []

    for feat, w in ranked[:4]:
        label = label_map.get(feat, feat)

        # 사용자 문장 템플릿 (개발자 용어 제거)
        if feat == "sentiment_mean":
            text = "최근 기사/커뮤니티의 전체 분위기가 방향성에 힘을 보탭니다"
        elif feat == "sentiment_vol":
            text = "기사 톤의 변동이 커서 단기 변동성 가능성을 키웁니다"
        elif feat == "news_count_7d":
            text = "최근 1주간 이슈가 빈번하게 등장했습니다"
        elif feat == "turnover_rate":
            text = "거래가 활발해 가격 움직임이 확대되기 쉬운 환경입니다"
        elif feat == "returns":
            text = "최근 가격 흐름이 현재 방향을 이어갈 가능성을 시사합니다"
        else:
            text = f"{label} 신호가 유의합니다"

        results.append((text, float(w)))

    return results
