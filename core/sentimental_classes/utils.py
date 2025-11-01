# core\sentimental_classes\utils.py

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def build_price_sentiment_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    예시: 가격 + 감성 집계 결합 (세부 로직은 로컬 변경분 이식)
    - 반드시 DataFrame 반환 (BaseAgent 파이프라인과 호환)
    """
    df = raw_df.copy()
    # TODO: 로컬 수정분의 전처리/특성 계산을 이 블록에 이식
    # ex) df['returns'] = df['Close'].pct_change()
    return df.dropna()


def build_sequences(df: pd.DataFrame, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    윈도우 시퀀스 생성 (X:[N,window,F], y:[N,1] 또는 [N])
    """
    cols = [c for c in df.columns if c not in ('y', 'target', 'label')]
    Xs, ys = [], []
    vals = df[cols].values
    # TODO: 로컬 규칙에 맞는 타깃 생성 반영
    y_raw = df['returns'].shift(-1) if 'returns' in df else df.iloc[:, 0].shift(-1)
    y_vals = y_raw.values
    for i in range(len(df) - window - 1):
        Xs.append(vals[i:i+window, :])
        ys.append(y_vals[i+window])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def rank_user_factors(context: Dict[str, Any], topk: int = 4):
    """
    사용자 친화적 표현을 위한 상위 요인 선정.
    - 개발자용 점수/어텐션 대신, plain language로 사유를 구성
    """
    # TODO: 로컬에서 계산한 해석 요소(예: 모멘텀, 뉴스 톤, 거래대금 변화 등)를 받아 정렬
    items = context.get("user_factors", [])
    items = sorted(items, key=lambda x: x.get("impact", 0), reverse=True)
    texts = [it.get("text", "") for it in items[:topk]]
    return [t for t in texts if t]
