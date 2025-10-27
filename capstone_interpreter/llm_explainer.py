import os
import pandas as pd
import numpy as np
import json

# 데이터 로드
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed_data_with_both_sentiment_V2.csv")
df = pd.read_csv(DATA_PATH)
FEATURES = ['prob_positive','prob_negative','prob_neutral','n_news','ret','Close','eod_sentiment']

# SHAP 값 예시 (shap_analyzer.py에서 결과 CSV로 저장해도 되고, 직접 불러도 됨)
# 예시: 이미 메모리에서 shap_values와 X를 불러온 경우
# 여기서는 dummy random 값, 실제로는 SHAP 결과를 넘겨받아야 함
np.random.seed(42)
shap_values = np.random.normal(0, 0.01, size=(len(df), len(FEATURES)))  # ★ 실제 값으로 대체할 것
X = df[FEATURES].values

sample_idx = -1
feature_importance = np.abs(shap_values).mean(axis=0)
sorted_indices = np.argsort(feature_importance)[::-1]
top_features = [FEATURES[i] for i in sorted_indices[:3]]

sample_shap = shap_values[sample_idx]
sample_input = X[sample_idx]
sample_dict = {FEATURES[i]: float(sample_input[i]) for i in range(len(FEATURES))}

prompt_json = {
    "timestamp": str(df.index[sample_idx]),
    "top_features": top_features,
    "sample_values": sample_dict,
    "shap_values": {f: float(sample_shap[i]) for i, f in enumerate(FEATURES)}
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
print(prompt_text)
