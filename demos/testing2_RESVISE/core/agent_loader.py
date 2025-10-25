import os
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 외부에서 import 가능한 AgentLoader
class AgentLoader:
    # 에이전트 로더 초기화
    """
    Args:
        config_path (str): 설정 파일 경로
        class_map (dict): {class_name: class_reference} 형태의 외부 클래스 매핑
        device (torch.device): 'cuda' or 'cpu' 자동 선택
    """

    def __init__(self, config_path="config/agents.yaml", class_map=None, device=None):
        self.config = self._load_config(config_path)
        self.models_dir = self.config["models_dir"]
        self.data_dir = self.config["data_dir"]
        self.agents_config = self.config["agents"]
        self.agents = {}

        # 클래스 매핑 정보
        self.class_map = class_map or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------
    # Config 불러오기
    # -----------------------------------------
    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # -----------------------------------------
    # 개별 에이전트 로드
    # -----------------------------------------
    def load_agent(self, agent_name):
        if agent_name in self.agents:
            return self.agents[agent_name]

        cfg = self.agents_config.get(agent_name)
        if cfg is None:
            raise ValueError(f"Unknown agent: {agent_name}")

        model_path = os.path.join(self.models_dir, cfg["model_file"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # 외부에서 전달된 클래스 매핑 사용
        class_name = cfg["class_name"]
        if class_name not in self.class_map:
            raise ValueError(f"Class '{class_name}' not found in class_map")

        model_class = self.class_map[class_name]
        model = model_class(input_dim=checkpoint["input_dim"], dropout=0.1)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Agent 정보 구성
        self.agents[agent_name] = {
            "model": model,
            "feature_cols": cfg["feature_cols"],
            "scaler_X": checkpoint["scaler_X"],
            "scaler_y": checkpoint["scaler_y"],
            "window_size": checkpoint["window_size"],
            "train_losses": checkpoint.get("train_losses", []),
            "val_losses": checkpoint.get("val_losses", []),
        }
        return self.agents[agent_name]

    # -----------------------------------------
    # 모든 에이전트 로드
    # -----------------------------------------
    def load_all(self):
        for name in self.agents_config.keys():
            try:
                self.load_agent(name)
                print(f"✅ Loaded {name}")
            except Exception as e:
                print(f"❌ {name} load failed: {e}")
        return self.agents

    # -----------------------------------------
    # 예측
    # -----------------------------------------
    def predict(self, agent_name, data: pd.DataFrame):
        agent = self.agents.get(agent_name) or self.load_agent(agent_name)
        model = agent["model"]
        scaler_X = agent["scaler_X"]
        scaler_y = agent["scaler_y"]
        feature_cols = agent["feature_cols"]
        window_size = agent["window_size"]

        features = data[feature_cols].values
        scaled = scaler_X.transform(features)

        # 시퀀스 구성
        if len(scaled) >= window_size:
            seq = scaled[-window_size:].reshape(1, window_size, -1)
        else:
            pad = np.zeros((window_size, scaled.shape[1]))
            pad[-len(scaled):] = scaled
            seq = pad.reshape(1, window_size, -1)

        with torch.no_grad():
            pred_scaled = model(torch.tensor(seq, dtype=torch.float32)).squeeze().numpy()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return float(pred[0])

    # -----------------------------------------
    # Agent 정보 조회
    # -----------------------------------------
    def get_info(self, agent_name):
        agent = self.agents.get(agent_name) or self.load_agent(agent_name)
        model = agent["model"]
        return {
            "model_class": model.__class__.__name__,
            "params": sum(p.numel() for p in model.parameters()),
            "features": len(agent["feature_cols"]),
            "window": agent["window_size"],
        }
