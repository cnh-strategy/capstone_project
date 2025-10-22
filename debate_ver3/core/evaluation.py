import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------
# 1️⃣ 기본 회귀 지표
# --------------------------------------------
def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE(%)": mape}

# --------------------------------------------
# 2️⃣ Agent별 예측 성능 비교
# --------------------------------------------

# --------------------------------------------
# 3️⃣ β 신뢰도 계산
# --------------------------------------------
def compute_beta_confidences(errors_dict):
    """MSE 기반 신뢰도 가중치 계산"""
    betas = {k: np.exp(-v) for k, v in errors_dict.items()}
    s = sum(betas.values())
    return {k: v / s for k, v in betas.items()}

# --------------------------------------------
# 4️⃣ Debate 합의 결과 평가
# --------------------------------------------
def evaluate_consensus(agents):
    preds = [a.last_pred for a in agents.values() if a.last_pred is not None]
    avg_pred = np.mean(preds)
    std_pred = np.std(preds)
    print(f"\n🧾 Consensus Result → Mean: {avg_pred:.4f} | Std: {std_pred:.4f}")
    return {"mean": avg_pred, "std": std_pred}
