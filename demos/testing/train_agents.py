# ======================================================
# Stage 1: Pre-training of Agents
# 각 Agent가 자신의 전문 도메인 패턴을 학습
# ======================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 1️⃣ 모델 정의
# ------------------------------

class TechnicalAgent(nn.Module):
    """Technical Analysis Agent - TCN based"""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Temporal Convolutional Network
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # 필수 - Stage 3 불확실성 추정용
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim//2, output_dim)
        
    def forward(self, x):
        # x shape: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

class FundamentalAgent(nn.Module):
    """Fundamental Analysis Agent - LSTM based"""
    def __init__(self, input_dim=16, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)  # 필수 - Stage 3 불확실성 추정용
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, time, features)
        lstm_out, _ = self.lstm(x)
        # Use the last time step output
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

class SentimentalAgent(nn.Module):
    """Sentimental Analysis Agent - Transformer based"""
    def __init__(self, input_dim=8, d_model=64, nhead=4, num_layers=2, output_dim=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)  # 필수 - Stage 3 불확실성 추정용
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: (batch, time, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Use the last time step output
        last_output = x[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

# ------------------------------
# 2️⃣ 데이터 로더
# ------------------------------

def load_dataset(dataset_type, phase='pretrain', ticker='TSLA'):
    """Load dataset for specific agent type"""
    file_path = f'data/{ticker}_{dataset_type}_{phase}.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prepare features and target
    if dataset_type == 'technical':
        # Technical features: Open, High, Low, Close, Volume, returns, sma_5, sma_20, rsi, volume_z
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'sma_5', 'sma_20', 'rsi', 'volume_z']
        target_col = 'Close'
    elif dataset_type == 'fundamental':
        # Fundamental features: Close, USD_KRW, NASDAQ, VIX, priceEarningsRatio, forwardPE, 
        # debtEquityRatio, returnOnAssets, returnOnEquity, profitMargins, grossMargins
        feature_cols = ['Close', 'USD_KRW', 'NASDAQ', 'VIX', 'priceEarningsRatio', 'forwardPE', 
                       'priceToBook', 'debtEquityRatio', 'returnOnAssets', 'returnOnEquity', 
                       'profitMargins', 'grossMargins']
        target_col = 'Close'
    elif dataset_type == 'sentimental':
        # Sentimental features: Close, returns, sentiment_mean, sentiment_vol
        feature_cols = ['Close', 'returns', 'sentiment_mean', 'sentiment_vol']
        target_col = 'Close'
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Extract features and target
    features = df[feature_cols].values
    target = df[target_col].values
    
    return features, target, feature_cols

def create_sequences(features, target, window_size=14):
    """Create time series sequences"""
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])
    
    return np.array(X), np.array(y)

# ------------------------------
# 3️⃣ 훈련 함수
# ------------------------------

def train_agent(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, batch_size=32):
    """Train a single agent"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam(lr=0.001)
    criterion = nn.MSELoss()  # MSE Loss
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20  # Early stopping patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return model, train_losses, val_losses

# ------------------------------
# 4️⃣ 메인 실행
# ------------------------------

def main():
    print("🚀 Stage 1: Pre-training of Agents")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Agent configurations
    agents_config = {
        'technical': {
            'model_class': TechnicalAgent,
            'input_dim': 10,
            'window_size': 14,
            'lr': 0.001
        },
        'fundamental': {
            'model_class': FundamentalAgent,
            'input_dim': 12,
            'window_size': 14,
            'lr': 0.001
        },
        'sentimental': {
            'model_class': SentimentalAgent,
            'input_dim': 4,
            'window_size': 14,
            'lr': 0.001
        }
    }
    
    trained_agents = {}
    
    for agent_name, config in agents_config.items():
        print(f"\n🎯 Training {agent_name.upper()} Agent")
        print("-" * 40)
        
        try:
            # Load data
            features, target, feature_cols = load_dataset(agent_name, 'pretrain')
            print(f"✅ Loaded {len(features)} samples with {len(feature_cols)} features")
            
            # Create sequences
            X, y = create_sequences(features, target, config['window_size'])
            print(f"✅ Created {len(X)} sequences with window size {config['window_size']}")
            
            # Normalize features and target
            scaler_X = MinMaxScaler()  # 입력 정규화
            scaler_y = MinMaxScaler()  # 출력 정규화
            
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler_X.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Train-validation split
            split_idx = int(0.8 * len(X_scaled))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
            
            print(f"✅ Train: {len(X_train)} samples, Val: {len(X_val)} samples")
            
            # Create and train model
            model = config['model_class'](
                input_dim=config['input_dim'],
                dropout=0.1  # 필수 - Stage 3 불확실성 추정용
            )
            
            print(f"✅ Model created: {model.__class__.__name__}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train
            trained_model, train_losses, val_losses = train_agent(
                model, X_train, y_train, X_val, y_val,
                epochs=100, lr=config['lr'], batch_size=32
            )
            
            # Save model with all required information
            model_path = f'models/{agent_name}_agent.pt'
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'model_class_name': config['model_class'].__name__,
                'input_dim': config['input_dim'],
                'window_size': config['window_size'],
                'scaler_X': scaler_X,  # 스케일러 저장
                'scaler_y': scaler_y,  # 스케일러 저장
                'feature_cols': feature_cols,  # feature meta 정보 저장
                'train_losses': train_losses,
                'val_losses': val_losses
            }, model_path)
            
            print(f"✅ Model saved: {model_path}")
            
            # Final evaluation
            trained_model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_pred = trained_model(X_val_tensor).squeeze().numpy()
                y_true = y_val
                
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                
                print(f"📊 Final Performance:")
                print(f"   MSE: {mse:.6f}")
                print(f"   MAE: {mae:.6f}")
                print(f"   RMSE: {np.sqrt(mse):.6f}")
            
            trained_agents[agent_name] = {
                'model': trained_model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'feature_cols': feature_cols,
                'window_size': config['window_size']
            }
            
        except Exception as e:
            print(f"❌ Error training {agent_name} agent: {e}")
            continue
    
    print(f"\n🎉 Stage 1 Training Complete!")
    print("=" * 60)
    print(f"✅ Trained agents: {list(trained_agents.keys())}")
    print(f"📁 Models saved in: models/")
    print(f"🔧 Each model has dropout=0.1 and is saved as .pt file")
    print(f"📊 Models include: model weights + scalers + feature meta info")
    
    return trained_agents

if __name__ == "__main__":
    trained_agents = main()
