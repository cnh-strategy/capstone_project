import torch
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from core.utils import weighted_mean_std  # 우리가 정의할 간단한 유틸 (아래 예시 있음)
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

def mutual_learning_with_individual_data(
    agents,
    datasets,
    rounds=5,
    metric="mse",
    normalize="none",   # "none", "zscore", "minmax"
    save_models=True,
    model_dir="models",
):
    """
    Selective Mutual Learning (Time-step β)
    - 각 시점마다 β-confidence를 별도로 계산
    - metric 및 normalization 방식 선택 가능
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Stage 2-Dynamic v2: Time-step Mutual Learning "
          f"(metric={metric}, normalize={normalize}) Start")

    metric_log = {name: [] for name in agents.keys()}
    beta_mean_log = {name: [] for name in agents.keys()}
    beta_matrix = []  # 각 라운드별 β_t 저장

    for r in range(rounds):
        print(f"\n🧭 Round {r+1}/{rounds}")

        # 1️⃣ 모든 Agent의 예측 계산
        all_preds = {}
        for name, agent in agents.items():
            X_agent, _ = datasets[name]
            preds = []
            with torch.no_grad():
                for t in range(X_agent.shape[0]):
                    preds.append(agent.forward(X_agent[t:t+1]).detach().numpy())
            all_preds[name] = np.vstack(preds)

        # 2️⃣ 공통 타깃 + 정규화
        y_true = next(iter(datasets.values()))[1].detach().numpy()
        y_true_proc = apply_normalization(y_true, method=normalize)
        all_preds_proc = {k: apply_normalization(v, method=normalize) for k, v in all_preds.items()}

        # 3️⃣ 시점별 β 계산
        T = len(y_true_proc)
        betas_t = []
        for t in range(T):
            preds_t = {n: all_preds_proc[n][t] for n in agents.keys()}
            betas_t.append(compute_confidence_per_timestep(preds_t, y_true_proc[t], metric=metric))
        beta_matrix.append(betas_t)

        # 4️⃣ Mutual Learning 업데이트
        for name, agent in agents.items():
            X_agent, y_agent = datasets[name]
            y_true_local = y_agent.detach().numpy()
            y_true_local_proc = apply_normalization(y_true_local, method=normalize)

            y_mutual_list = []
            for t in range(T):
                # 시점별 β 사용
                beta_i_t = np.clip(betas_t[t][name], 0.2, 0.8)
                other_agents = [k for k in agents.keys() if k != name]
                other_preds = np.array([all_preds_proc[o][t] for o in other_agents])
                other_weights = np.array([betas_t[t][o] for o in other_agents])
                other_avg_t = np.average(other_preds, axis=0, weights=other_weights)

                y_mutual_t = beta_i_t * y_true_local_proc[t] + (1 - beta_i_t) * other_avg_t
                y_mutual_t = 0.9 * y_true_local_proc[t] + 0.1 * y_mutual_t
                y_mutual_list.append(y_mutual_t)

            y_mutual_tensor = torch.tensor(np.vstack(y_mutual_list), dtype=torch.float32)
            agent.update_parameters(X_agent, y_mutual_tensor)

        # 5️⃣ 라운드별 metric / β 평균 기록
        round_betas_mean = {n: np.mean([bt[n] for bt in betas_t]) for n in agents.keys()}
        beta_mean_log = {k: v + [round_betas_mean[k]] for k, v in beta_mean_log.items()}

        for n in agents.keys():
            val = compute_metric(y_true_proc, all_preds_proc[n], metric)
            metric_log[n].append(val)

        print(f"📉 Round {r+1} Summary:")
        for n in agents.keys():
            print(f"   - {n:20s} | {metric.upper()}={metric_log[n][-1]:.6f} | β̄={beta_mean_log[n][-1]:.3f}")

    # =====================================================
    # ✅ 그래프 저장 (Metric / β / Heatmap)
    # =====================================================
    save_dir = os.path.join(model_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    rounds_list = list(range(1, rounds + 1))

    # (1) Metric per Round
    plt.figure(figsize=(8, 5))
    for n in agents.keys():
        plt.plot(rounds_list, metric_log[n], marker="o", label=f"{n} {metric.upper()}")
    plt.xlabel("Round")
    plt.ylabel(metric.upper())
    plt.title(f"📉 {metric.upper()} per Round ({normalize}-normalized)")
    plt.legend()
    plt.grid(True)
    path1 = os.path.join(save_dir, f"{metric.upper()}_per_round_v2.png")
    plt.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path1}")

    # (2) Round별 평균 β 변화
    plt.figure(figsize=(8, 5))
    for n in agents.keys():
        plt.plot(rounds_list, beta_mean_log[n], marker="o", label=f"{n} β̄")
    plt.xlabel("Round")
    plt.ylabel("Mean β")
    plt.title("🧠 Mean β per Round (Time-step β Version)")
    plt.legend()
    plt.grid(True)
    path2 = os.path.join(save_dir, "Mean_Beta_per_round_v2.png")
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🧠 Saved: {path2}")

    # (3) 마지막 라운드 β heatmap
    betas_last = beta_matrix[-1]
    beta_arr = np.array([[bt[n] for n in agents.keys()] for bt in betas_last])
    plt.figure(figsize=(7, 4))
    plt.imshow(beta_arr.T, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="β-confidence")
    plt.yticks(range(len(agents.keys())), list(agents.keys()))
    plt.xlabel("Time-step")
    plt.title("🔥 Time-step β Heatmap (Final Round)")
    path3 = os.path.join(save_dir, "Beta_timestep_heatmap_v2.png")
    plt.savefig(path3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🔥 Saved: {path3}")

    # =====================================================
    # ✅ 모델 저장
    # =====================================================
    if save_models:
        print("\n💾 Stage 2 Time-step Mutual Learning 완료 → 모델 저장 중...")
        os.makedirs(model_dir, exist_ok=True)
        for name, agent in agents.items():
            agent_name = agent.agent_id.replace("Agent", "").lower()
            model_path = os.path.join(model_dir, f"{agent_name}_agent_dynamic_v2.pt")
            torch.save(agent.state_dict(), model_path)
            print(f"💾 Saved: {model_path}")

    print(f"\n✅ Time-step β Mutual Learning Finished.\n")

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

# def mutual_learning_with_individual_data(agents, datasets, rounds=5, save_models=True, model_dir="models"):
#     """각 에이전트별로 적절한 데이터를 사용하는 상호 학습"""
#     print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔁 Stage 2: Mutual Learning Start")
    
#     for r in range(rounds):
#         preds = {}
#         for name, agent in agents.items():
#             with torch.no_grad():
#                 # 각 에이전트별로 적절한 데이터 사용
#                 X_agent, y_agent = datasets[name]
#                 if hasattr(agent, 'forward') and hasattr(agent, 'parameters'):
#                     preds[name] = agent.forward(X_agent).detach().numpy()
#                 else:
#                     preds[name] = agent.model(X_agent).detach().numpy()
        
#         # 공통 타겟 사용 (모든 에이전트의 타겟은 동일)
#         y_true = next(iter(datasets.values()))[1].detach().numpy()
#         betas = compute_confidence(preds, y_true)

#         for name, agent in agents.items():
#             # 수정된 Mutual Learning 공식: 가중치 합이 1이 되도록 정규화
#             other_agents = [pn for pn in agents.keys() if pn != name]
#             if other_agents:
#                 # 다른 에이전트들의 가중평균
#                 other_preds = np.array([preds[pn] for pn in other_agents])
#                 other_weights = np.array([betas[pn] for pn in other_agents])
#                 other_weights = other_weights / (other_weights.sum() + 1e-8)  # 정규화
#                 other_avg = np.average(other_preds, axis=0, weights=other_weights)
                
#                 # 현재 에이전트의 신뢰도와 다른 에이전트들의 평균을 결합
#                 y_mutual = (betas[name] * y_true) + ((1 - betas[name]) * other_avg)
#             else:
#                 y_mutual = y_true
            
#             y_mutual = torch.tensor(y_mutual, dtype=torch.float32)
#             # 각 에이전트별로 적절한 입력 데이터 사용
#             X_agent, _ = datasets[name]
#             agent.update_parameters(X_agent, y_mutual)
#         print(f"  🔹 Round {r+1}/{rounds} completed.")
    
#     # 상호학습 완료 후 모델 저장
#     if save_models:
#         print("💾 상호학습 완료된 모델들을 저장합니다...")
#         for name, agent in agents.items():
#             agent_name = agent.agent_id.replace("Agent", "").lower()
#             model_path = os.path.join(model_dir, f"{agent_name}_agent_finetuned.pt")
            
#             model_state = {
#                 "model_state_dict": agent.state_dict(),
#                 "agent_id": agent.agent_id,
#                 "model_type": type(agent).__name__,
#                 "timestamp": datetime.now().isoformat(),
#                 "training_stage": "mutual_learning"
#             }
            
#             os.makedirs(model_dir, exist_ok=True)
#             torch.save(model_state, model_path)
#             print(f"💾 {agent.agent_id} 상호학습 모델 저장됨: {model_path}")
    
#     print(f"✅ Mutual Learning finished.\n")

