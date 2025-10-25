import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

def pretrain_agent(agent, train_X, train_y, val_X=None, val_y=None, epochs=50, lr=1e-3, batch_size=32):
    """개별 Agent 사전학습"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 Pretraining {agent.agent_id}")
    
    # Agent가 nn.Module을 상속받는지 확인
    if hasattr(agent, 'forward') and hasattr(agent, 'parameters'):
        # Agent 자체가 모델인 경우
        agent.train()
        optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        
        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            total_loss = 0
            for Xb, yb in train_loader:
                y_pred = agent.forward(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")
    else:
        # 기존 방식 (agent.model 사용)
        agent.model.train()
        optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            total_loss = 0
            for Xb, yb in train_loader:
                y_pred = agent.model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")
    
    print(f"✅ {agent.agent_id} pretraining finished.\n")

def save_agent_model(agent, model_dir="models"):
    """Agent 모델을 .pt 파일로 저장"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Agent 이름에서 Agent 접미사 제거
    agent_name = agent.agent_id.replace("Agent", "").lower()
    model_path = os.path.join(model_dir, f"{agent_name}_agent.pt")
    
    # 모델 상태 저장
    model_state = {
        "model_state_dict": agent.state_dict(),
        "agent_id": agent.agent_id,
        "model_type": type(agent).__name__,
        "timestamp": datetime.now().isoformat()
    }
    
    torch.save(model_state, model_path)
    print(f"💾 {agent.agent_id} 모델 저장됨: {model_path}")

def load_agent_model(agent, model_dir="models"):
    """저장된 Agent 모델을 로드"""
    agent_name = agent.agent_id.replace("Agent", "").lower()
    model_path = os.path.join(model_dir, f"{agent_name}_agent.pt")
    
    if os.path.exists(model_path):
        try:
            model_state = torch.load(model_path, map_location="cpu")
            agent.load_state_dict(model_state["model_state_dict"])
            print(f"📂 {agent.agent_id} 모델 로드됨: {model_path}")
            return True
        except Exception as e:
            print(f"⚠️ {agent.agent_id} 모델 로드 실패: {e}")
            return False
    else:
        print(f"⚠️ {agent.agent_id} 모델 파일 없음: {model_path}")
        return False

def pretrain_all_agents(agents, datasets, epochs=50, lr=1e-3, save_models=True, model_dir="models"):
    """모든 Agent에 대해 사전학습"""
    for name, agent in agents.items():
        X_train, y_train = datasets[name]
        pretrain_agent(agent, X_train, y_train, epochs=epochs, lr=lr)
        
        # 모델 저장
        if save_models:
            save_agent_model(agent, model_dir)
