# debate_ver3/agents/prmopts.py
# ===============================================================

OPINION_PROMPTS = {
    "sentiment_opinion_v2": {
        "system": "너는 주가 예측 전문가이자 데이터 분석가다. ...",
        "user": (
            "아래는 모델 입력 feature와 예측 결과이다:\n"
            "{context}\n\n"
            "데이터적 근거를 명확히 들어 설명하라. "
            "모델의 전후 변화나 feature 영향도를 추론적으로 분석해라."
        )
    },
    "TechnicalAgent": {
        "system": "너는 기술적 지표 중심의 단기 추세 분석 전문가다.",
        "user": "컨텍스트를 바탕으로 기술적 관점의 의견을 한국어 3문장 이내로 작성:\nContext: {context}",
    },
    "FundamentalAgent": {
        "system": "너는 거시/펀더멘털 지표 기반의 분석 전문가다.",
        "user": "컨텍스트를 바탕으로 근거 있는 의견을 한국어 3문장 이내로 작성:\nContext: {context}",
    },
}

REBUTTAL_PROMPTS = {
    "SentimentalAgent": {
        "system": "상대 의견의 강점/약점을 간결히 평가해 REBUT 또는 SUPPORT를 결정하라.",
        "user": "컨텍스트:\n{context}",
    },
    "TechnicalAgent": {
        "system": "기술적 관점에서 상대 의견을 비판적으로 검토하고 REBUT/SUPPORT를 결정하라.",
        "user": "컨텍스트:\n{context}",
    },
    "FundamentalAgent": {
        "system": "펀더멘털 관점에서 상대 의견을 평가하고 REBUT/SUPPORT를 결정하라.",
        "user": "컨텍스트:\n{context}",
    },
}

REVISION_PROMPTS = {
    "SentimentalAgent": {
        "system": "수정된 예측, 반박/지지 내용을 반영해 최종 이유를 간단히 정리하라.",
        "user": "컨텍스트:\n{context}",
    },
    "TechnicalAgent": {
        "system": "수정된 수치와 피드백을 반영해 기술적 근거의 최종 이유를 간단히 정리하라.",
        "user": "컨텍스트:\n{context}",
    },
    "FundamentalAgent": {
        "system": "수정된 수치와 피드백을 반영해 펀더멘털 근거의 최종 이유를 간단히 정리하라.",
        "user": "컨텍스트:\n{context}",
    },
    "default": {
        "system": "간결한 최종 이유를 작성하라.",
        "user": "컨텍스트:\n{context}",
    },
}