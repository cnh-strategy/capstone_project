import torch
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
# from prompts import DEBATE_PROMPTS
import json

# ==========================================================
# 🧩 Utility Functions
# ==========================================================

def apply_normalization(y, method="none"):
    """출력값 정규화 방법 선택"""
    y = np.array(y)
    eps = 1e-8

    if method == "none":
        return y
    elif method == "zscore":
        return (y - np.mean(y)) / (np.std(y) + eps)
    elif method == "minmax":
        return (y - np.min(y)) / (np.max(y) - np.min(y) + eps)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_metric(y_true, y_pred, metric="mse"):
    """metric별 계산 (mse / rmse / mape)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-8

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "rmse":
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    elif metric == "mape":
        return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_confidence_per_timestep(preds_t_dict, y_t, metric="mse"):
    """시점별 β-confidence 계산"""
    metric_values = []
    for name, preds in preds_t_dict.items():
        val = compute_metric(np.array([y_t]), np.array([preds]), metric)
        metric_values.append(val)

    metric_values = np.array(metric_values)
    inv_loss = np.exp(-metric_values / (metric_values.std() + 1e-8))
    betas = inv_loss / np.sum(inv_loss)
    return {name: betas[i] for i, name in enumerate(preds_t_dict.keys())}

# ==========================================================
# 🧠 Dynamic Mutual Learning (Stable + Visualization)
# ==========================================================

from core.utils import weighted_mean_std  # 우리가 정의할 간단한 유틸 (아래 예시 있음)

def aggregate_consensus(agents: dict, opinions: dict, llm_client) -> dict:
    """
    각 Agent의 예측(next_close)과 신뢰도(β)를 기반으로 
    가중평균 예측 및 합성 reasoning 생성
    """
    # -----------------------------------
    # 1️⃣ 가중 평균 계산
    # -----------------------------------
    preds = np.array([op.target.next_close for op in opinions.values()])
    confs = np.array([max(op.target.confidence or 0.01, 1e-3) for op in opinions.values()])  # confidence β

    weights = confs / np.sum(confs)
    mean_pred = np.sum(preds * weights)
    std_pred = np.sqrt(np.sum(weights * (preds - mean_pred) ** 2))

    # -----------------------------------
    # 2️⃣ Reasoning 합성 (LLM)
    # -----------------------------------
    debate_prompt = DEBATE_PROMPTS["strategy_reason"]
    context_json = json.dumps({
        "predictions": [
            {
                "agent_id": op.agent_id,
                "predicted_next_close": op.target.next_close,
                "confidence_beta": op.target.confidence,
                "reason": op.reason
            }
            for op in opinions.values()
        ],
        "weighted_mean": round(mean_pred, 3),
        "weighted_std": round(std_pred, 4)
    }, ensure_ascii=False, indent=2)

    sys_text = debate_prompt["system"]
    user_text = debate_prompt["user_template"].format(context=context_json)

    # Schema: reasoning만 반환
    schema = {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}

    parsed = llm_client.ask_with_schema(
        messages=[{"role": "system", "content": sys_text},
                  {"role": "user", "content": user_text}],
        schema=schema
    )

    final_reason = parsed.get("reason", "(최종 reasoning 생성 실패)")

    # -----------------------------------
    # 3️⃣ 최종 결과 리턴
    # -----------------------------------
    return {
        "consensus_next_close": round(mean_pred, 3),
        "std": round(std_pred, 4),
        "reason": final_reason,
        "weights": {k: round(w, 3) for k, w in zip(opinions.keys(), weights)},
    }
