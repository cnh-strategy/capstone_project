import torch
import pandas as pd
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from .model_def import StockSentimentLSTM


# 모델 로드
model = StockSentimentLSTM(hidden_dim=128, num_layers=2, input_size=7)
model.load_state_dict(torch.load("model_lstm_bothsentiment_V2.pt", map_location='cpu'))
model.eval()


# 데이터 경로 안전하게 지정
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed_data_with_both_sentiment_V2.csv")
df = pd.read_csv(DATA_PATH)

FEATURES = ['prob_positive','prob_negative','prob_neutral','n_news','ret','Close','eod_sentiment']
X = df[FEATURES].values.astype(np.float32)

# 샘플 텐서 생성
sample = torch.tensor(X[:100]).view(100, -1, len(FEATURES))


# SHAP GradientExplainer 사용 (회귀 모델에 적합)
explainer = shap.GradientExplainer(model, sample)
shap_values = explainer.shap_values(sample)

# SHAP 값 차원 축소
shap_values = np.array(shap_values).squeeze()

print("SHAP values shape after squeeze:", shap_values.shape)
print("Input features shape:", X[:100].shape)

assert shap_values.shape == X[:100].shape, "SHAP values와 입력 데이터의 shape가 같아야 합니다."

shap.summary_plot(shap_values, features=X[:100], feature_names=FEATURES)

# 시각화
shap.summary_plot(shap_values, features=X[:100], feature_names=FEATURES)
plt.savefig(os.path.join(os.path.dirname(__file__), "explain_results", "shap_summary.png"), dpi=200)

print("✅ SHAP summary plot saved.")