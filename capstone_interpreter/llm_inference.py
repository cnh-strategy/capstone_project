import os
import json
from openai import OpenAI

# OpenAI API 키 환경변수 설정 필요(terminal에서 실행 전)
# set OPENAI_API_KEY=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


# (1) SHAP 프롬프트 예시 텍스트 생성 함수

def make_prompt():
    prompt_json = {
        "timestamp": "3827",
        "top_features": ["ret", "prob_negative", "Close"],
        "sample_values": {
            "prob_positive": 0.51852566,
            "prob_negative": 0.23568296,
            "prob_neutral": 0.24579139,
            "n_news": 40.0,
            "ret": 0.0076799,
            "Close": 124.62285614,
            "eod_sentiment": 0.0
        },
        "shap_values": {
            "prob_positive": 0.00975,
            "prob_negative": 0.01304,
            "prob_neutral": 0.01143,
            "n_news": 0.01979,
            "ret": -0.0009,
            "Close": 0.00177,
            "eod_sentiment": 0.0032
        }
    }

    prompt_text = f"""
다음은 주가 예측 모델(LSTM+SHAP) 분석 결과입니다.

예측 시점: {prompt_json['timestamp']}
가장 영향력이 강한 피처: {', '.join(prompt_json['top_features'])}
입력 값: {json.dumps(prompt_json['sample_values'], ensure_ascii=False)}
SHAP 영향 점수: {json.dumps(prompt_json['shap_values'], ensure_ascii=False)}

이 데이터를 바탕으로:
1. 모델이 왜 이런 예측을 했는지 설명하세요.
2. 어떤 요인이 예측에 가장 크게 작용했나요?
3. 뉴스/감성 피처와 기술적 지표 중 어떤 경향성이 두드러졌나요?
4. 투자나 의사결정 참고 시 주의점이 있으면 알려주세요.

-- 상세 해석은 (금융/AI 전문가 기준 서술, 한국어)
"""
    return prompt_text


# (2) OpenAI API로 프롬프트 전달하고 답변 받는 함수

def get_llm_response(prompt_text):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response.choices[0].message.content


# (3) 실행부
if __name__ == "__main__":
    prompt = make_prompt()
    print("=== LLM에게 보낼 프롬프트 ===")
    print(prompt)
    print("=== LLM 응답 시작 ===")
    explanation = get_llm_response(prompt)
    print(explanation)
