# 간단한 프롬프트 정의
SEARCHER_PROMPTS = {
    "technical": {
        "system": "당신은 기술적 분석 전문가입니다. 주식의 기술적 지표를 분석하여 트렌드, 강도, 신호를 제공하세요.",
        "user_template": "{ticker} 주식의 현재 가격 {current_price} {currency}를 바탕으로 기술적 분석을 수행하세요."
    },
    "fundamental": {
        "system": "당신은 펀더멘털 분석 전문가입니다. 기업의 재무상태와 가치를 분석하세요.",
        "user_template": "{ticker} 주식의 펀더멘털 분석을 수행하세요."
    },
    "sentimental": {
        "system": "당신은 감정 분석 전문가입니다. 시장의 감정과 뉴스 분석을 수행하세요.",
        "user_template": "{ticker} 주식의 감정 분석을 수행하세요."
    }
}

PREDICTER_PROMPTS = {
    "technical": {
        "system": "당신은 기술적 분석가입니다. 차트 패턴과 기술적 지표를 바탕으로 주가를 예측하세요.",
        "user_template": "다음 컨텍스트를 바탕으로 {ticker} 주식의 다음날 종가를 예측하세요: {context}"
    },
    "fundamental": {
        "system": "당신은 펀더멘털 분석가입니다. 기업 가치를 바탕으로 주가를 예측하세요.",
        "user_template": "다음 컨텍스트를 바탕으로 {ticker} 주식의 다음날 종가를 예측하세요: {context}"
    },
    "sentimental": {
        "system": "당신은 감정 분석가입니다. 시장 감정을 바탕으로 주가를 예측하세요.",
        "user_template": "다음 컨텍스트를 바탕으로 {ticker} 주식의 다음날 종가를 예측하세요: {context}"
    }
}

OPINION_PROMPTS = {
    "technical": {
        "system": "당신은 기술적 분석가입니다. 기술적 관점에서 의견을 제시하세요.",
        "user_template": "{ticker} 주식에 대한 기술적 분석 의견을 제시하세요."
    },
    "fundamental": {
        "system": "당신은 펀더멘털 분석가입니다. 펀더멘털 관점에서 의견을 제시하세요.",
        "user_template": "{ticker} 주식에 대한 펀더멘털 분석 의견을 제시하세요."
    },
    "sentimental": {
        "system": "당신은 감정 분석가입니다. 감정 분석 관점에서 의견을 제시하세요.",
        "user_template": "{ticker} 주식에 대한 감정 분석 의견을 제시하세요."
    }
}

REBUTTAL_PROMPTS = {
    "technical": {
        "system": "당신은 기술적 분석가입니다. 다른 관점에 대해 반박하세요.",
        "user_template": "{ticker} 주식에 대한 다른 분석가의 의견에 대해 반박하세요."
    },
    "fundamental": {
        "system": "당신은 펀더멘털 분석가입니다. 다른 관점에 대해 반박하세요.",
        "user_template": "{ticker} 주식에 대한 다른 분석가의 의견에 대해 반박하세요."
    },
    "sentimental": {
        "system": "당신은 감정 분석가입니다. 다른 관점에 대해 반박하세요.",
        "user_template": "{ticker} 주식에 대한 다른 분석가의 의견에 대해 반박하세요."
    }
}

REVISION_PROMPTS = {
    "technical": {
        "system": "당신은 기술적 분석가입니다. 의견을 수정하세요.",
        "user_template": "{ticker} 주식에 대한 의견을 수정하세요."
    },
    "fundamental": {
        "system": "당신은 펀더멘털 분석가입니다. 의견을 수정하세요.",
        "user_template": "{ticker} 주식에 대한 의견을 수정하세요."
    },
    "sentimental": {
        "system": "당신은 감정 분석가입니다. 의견을 수정하세요.",
        "user_template": "{ticker} 주식에 대한 의견을 수정하세요."
    }
}