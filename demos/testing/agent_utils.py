# ======================================================
# Agent Utilities for New Debating System
# Load and use pretrained agents
# ======================================================

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Define model classes directly to avoid import issues
import torch.nn as nn

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
        self.dropout = nn.Dropout(dropout)
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
        self.dropout = nn.Dropout(dropout)
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
        self.dropout = nn.Dropout(dropout)
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

class AgentLoader:
    """Load and manage pretrained agents"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.agents = {}
        self.model_classes = {
            'technical': TechnicalAgent,
            'fundamental': FundamentalAgent,
            'sentimental': SentimentalAgent
        }
    
    def load_agent(self, agent_name):
        """Load a specific pretrained agent"""
        if agent_name in self.agents:
            return self.agents[agent_name]
        
        model_path = os.path.join(self.models_dir, f'{agent_name}_agent.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model instance
        model_class_name = checkpoint['model_class_name']
        model_class = self.model_classes[agent_name]
        model = model_class(
            input_dim=checkpoint['input_dim'],
            dropout=0.1
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Store agent info
        agent_info = {
            'model': model,
            'scaler_X': checkpoint['scaler_X'],
            'scaler_y': checkpoint['scaler_y'],
            'feature_cols': checkpoint['feature_cols'],
            'window_size': checkpoint['window_size'],
            'train_losses': checkpoint['train_losses'],
            'val_losses': checkpoint['val_losses']
        }
        
        self.agents[agent_name] = agent_info
        return agent_info
    
    def load_all_agents(self):
        """Load all available agents"""
        available_agents = ['technical', 'fundamental', 'sentimental']
        
        for agent_name in available_agents:
            try:
                self.load_agent(agent_name)
                print(f"✅ Loaded {agent_name} agent")
            except Exception as e:
                print(f"❌ Failed to load {agent_name} agent: {e}")
        
        return self.agents
    
    def predict(self, agent_name, data):
        """Make prediction using a specific agent"""
        if agent_name not in self.agents:
            self.load_agent(agent_name)
        
        agent_info = self.agents[agent_name]
        model = agent_info['model']
        scaler_X = agent_info['scaler_X']
        scaler_y = agent_info['scaler_y']
        window_size = agent_info['window_size']
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            # Extract features based on agent type
            if agent_name == 'technical':
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'sma_5', 'sma_20', 'rsi', 'volume_z']
            elif agent_name == 'fundamental':
                feature_cols = ['Close', 'USD_KRW', 'NASDAQ', 'VIX', 'priceEarningsRatio', 'forwardPE', 
                               'priceToBook', 'debtEquityRatio', 'returnOnAssets', 'returnOnEquity', 
                               'profitMargins', 'grossMargins']
            elif agent_name == 'sentimental':
                feature_cols = ['Close', 'returns', 'sentiment_mean', 'sentiment_vol']
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")
            
            features = data[feature_cols].values
        else:
            features = data
        
        # Normalize features
        features_scaled = scaler_X.transform(features)
        
        # Create sequence if needed
        if len(features_scaled) >= window_size:
            sequence = features_scaled[-window_size:].reshape(1, window_size, -1)
        else:
            # Pad with zeros if sequence is too short
            padded = np.zeros((window_size, features_scaled.shape[1]))
            padded[-len(features_scaled):] = features_scaled
            sequence = padded.reshape(1, window_size, -1)
        
        # Make prediction
        with torch.no_grad():
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            prediction_scaled = model(sequence_tensor).squeeze().numpy()
            
            # Inverse transform prediction
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        
        return prediction[0]
    
    def get_agent_info(self, agent_name):
        """Get information about a specific agent"""
        if agent_name not in self.agents:
            self.load_agent(agent_name)
        
        agent_info = self.agents[agent_name]
        
        info = {
            'model_class': agent_info['model'].__class__.__name__,
            'parameters': sum(p.numel() for p in agent_info['model'].parameters()),
            'window_size': agent_info['window_size'],
            'feature_cols': agent_info['feature_cols'],
            'train_losses': agent_info['train_losses'],
            'val_losses': agent_info['val_losses']
        }
        
        return info

def test_agents():
    """Test all agents with sample data"""
    print("🧪 Testing Pretrained Agents")
    print("=" * 50)
    
    # Load agent loader
    loader = AgentLoader()
    loader.load_all_agents()
    
    # Load test data
    test_data = {}
    for agent_type in ['technical', 'fundamental', 'sentimental']:
        try:
            df = pd.read_csv(f'data/TSLA_{agent_type}_test.csv')
            test_data[agent_type] = df
            print(f"✅ Loaded {agent_type} test data: {len(df)} samples")
        except Exception as e:
            print(f"❌ Failed to load {agent_type} test data: {e}")
    
    # Test predictions
    print(f"\n🔮 Making Predictions")
    print("-" * 30)
    
    for agent_name in loader.agents.keys():
        try:
            # Get sample data
            df = test_data[agent_name]
            sample_data = df.head(20)  # Use first 20 samples
            
            # Make prediction
            prediction = loader.predict(agent_name, sample_data)
            
            # Get actual value for comparison
            actual = sample_data['Close'].iloc[-1]
            
            print(f"{agent_name.upper():>12}: Predicted={prediction:.2f}, Actual={actual:.2f}, Error={abs(prediction-actual):.2f}")
            
        except Exception as e:
            print(f"{agent_name.upper():>12}: Error - {e}")
    
    # Show agent info
    print(f"\n📊 Agent Information")
    print("-" * 30)
    
    for agent_name in loader.agents.keys():
        info = loader.get_agent_info(agent_name)
        print(f"{agent_name.upper():>12}: {info['model_class']} ({info['parameters']:,} params)")
        print(f"{'':>12}   Window: {info['window_size']}, Features: {len(info['feature_cols'])}")

if __name__ == "__main__":
    test_agents()
