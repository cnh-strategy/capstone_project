# ======================================================
# Ticker Input System
# 사용자가 원하는 주식을 입력하면 전체 시스템 실행
# ======================================================

import os
import sys
import pandas as pd
import torch
from single_ticker_builder import SingleTickerBuilder
from train_agents import main as train_agents_main, load_dataset, create_sequences, train_agent
from train_agents import TechnicalAgent, FundamentalAgent, SentimentalAgent
from stage2_trainer import Stage2Trainer
from debate_system import DebateSystem
import warnings
warnings.filterwarnings("ignore")

class TickerInputSystem:
    """System that takes user input for ticker and runs complete pipeline"""
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.builder = SingleTickerBuilder(data_dir)
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    def get_ticker_from_user(self):
        """Get ticker input from user"""
        print("🚀 3-Stage Debating System")
        print("=" * 50)
        print("📊 Popular tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX")
        print("💼 Finance: JPM, BAC, WFC, GS, MS, C, AXP, BLK")
        print("🏥 Healthcare: JNJ, PFE, UNH, ABBV, MRK, TMO, ABT, DHR")
        print("🛒 Consumer: KO, PEP, WMT, PG, MCD, NKE, SBUX")
        print("⛽ Energy: XOM, CVX, COP, EOG, SLB, PXD, MPC, VLO")
        print("🇰🇷 Korean: 005930.KS, 000660.KS, 035420.KS, 207940.KS, 006400.KS")
        print("=" * 50)
        
        while True:
            ticker = input("\nEnter ticker symbol (or 'quit' to exit): ").upper().strip()
            
            if ticker.lower() == 'quit':
                print("👋 Goodbye!")
                return None
            
            if not ticker:
                print("❌ Please enter a valid ticker symbol")
                continue
            
            # Validate ticker
            is_valid, message = self.builder.validate_ticker(ticker)
            if is_valid:
                print(f"✅ Valid ticker: {ticker}")
                return ticker
            else:
                print(f"❌ Invalid ticker {ticker}: {message}")
                print("Please try again or enter 'quit' to exit")
    
    def build_datasets(self, ticker):
        """Build datasets for the ticker"""
        print(f"\n📊 Step 1: Building datasets for {ticker}")
        print("-" * 40)
        
        success = self.builder.build_datasets_for_ticker(ticker)
        if not success:
            print(f"❌ Failed to build datasets for {ticker}")
            return False
        
        print(f"✅ Datasets created successfully for {ticker}")
        return True
    
    def train_agents(self, ticker):
        """Train agents for the ticker"""
        print(f"\n🎯 Step 2: Training agents for {ticker}")
        print("-" * 40)
        
        try:
            # Import and modify the training function
            from train_agents import load_dataset, create_sequences, train_agent
            
            # Agent configurations
            agents_config = {
                'technical': {
                    'input_dim': 10,
                    'window_size': 14,
                    'lr': 0.001
                },
                'fundamental': {
                    'input_dim': 12,
                    'window_size': 14,
                    'lr': 0.001
                },
                'sentimental': {
                    'input_dim': 4,
                    'window_size': 14,
                    'lr': 0.001
                }
            }
            
            trained_agents = {}
            
            for agent_name, config in agents_config.items():
                print(f"\n🎯 Training {agent_name.upper()} Agent for {ticker}")
                print("-" * 30)
                
                try:
                    # Load data with ticker prefix
                    features, target, feature_cols = load_dataset(agent_name, 'pretrain', ticker)
                    print(f"✅ Loaded {len(features)} samples with {len(feature_cols)} features")
                    
                    # Create sequences
                    X, y = create_sequences(features, target, config['window_size'])
                    print(f"✅ Created {len(X)} sequences with window size {config['window_size']}")
                    
                    # Create model instance
                    if agent_name == 'technical':
                        model = TechnicalAgent(input_dim=config['input_dim'], dropout=0.1)
                    elif agent_name == 'fundamental':
                        model = FundamentalAgent(input_dim=config['input_dim'], dropout=0.1)
                    elif agent_name == 'sentimental':
                        model = SentimentalAgent(input_dim=config['input_dim'], dropout=0.1)
                    
                    # Split data for training
                    split_idx = int(len(X) * 0.8)
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    # Train agent
                    trained_model, train_losses, val_losses = train_agent(
                        model, X_train, y_train, X_val, y_val, 
                        epochs=100, lr=config['lr'], batch_size=32
                    )
                    
                    # Save model with ticker prefix
                    model_path = f"{self.models_dir}/{ticker}_{agent_name}_agent.pt"
                    
                    # Save model checkpoint
                    torch.save({
                        'model_state_dict': trained_model.state_dict(),
                        'model_class_name': trained_model.__class__.__name__,
                        'input_dim': config['input_dim'],
                        'window_size': config['window_size'],
                        'feature_cols': feature_cols,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                    }, model_path)
                    
                    print(f"✅ {agent_name.title()} agent trained and saved to {model_path}")
                    trained_agents[agent_name] = True
                    
                except Exception as e:
                    print(f"❌ Error training {agent_name} agent for {ticker}: {e}")
                    continue
            
            return len(trained_agents) == 3
            
        except Exception as e:
            print(f"❌ Training failed for {ticker}: {e}")
            return False
    
    def run_mutual_learning(self, ticker):
        """Run mutual learning for the ticker"""
        print(f"\n🔄 Step 3: Running mutual learning for {ticker}")
        print("-" * 40)
        
        try:
            # Load mutual data
            data_dict = {}
            for agent_type in ['technical', 'fundamental', 'sentimental']:
                data_dict[agent_type] = pd.read_csv(f'{self.data_dir}/{ticker}_{agent_type}_mutual.csv')
            
            # Initialize trainer
            trainer = Stage2Trainer(self.models_dir)
            
            # Run mutual learning
            results = trainer.train_mutual_learning(data_dict, start_idx=0, end_idx=100)
            
            # Save results with ticker prefix
            trainer.save_beta_log(f'{self.data_dir}/{ticker}_beta_log.csv')
            
            print(f"✅ Mutual learning completed for {ticker}")
            return True
            
        except Exception as e:
            print(f"❌ Mutual learning failed for {ticker}: {e}")
            return False
    
    def run_debate_system(self, ticker):
        """Run debate system for the ticker"""
        print(f"\n🎭 Step 4: Running debate system for {ticker}")
        print("-" * 40)
        
        try:
            # Load test data
            data_dict = {}
            for agent_type in ['technical', 'fundamental', 'sentimental']:
                data_dict[agent_type] = pd.read_csv(f'{self.data_dir}/{ticker}_{agent_type}_test.csv')
            
            # Initialize debate system
            debate_system = DebateSystem(self.models_dir)
            
            # Test multiple predictions
            test_points = [50, 100, 150, 180]
            
            for i, point in enumerate(test_points):
                print(f"\n🎯 Prediction {i+1}/{len(test_points)} - Point {point}")
                
                # Get data slice
                data_slice = {}
                actual_price = None
                
                for agent_name in data_dict.keys():
                    data_slice[agent_name] = data_dict[agent_name].iloc[:point+1]
                    if actual_price is None:
                        actual_price = data_dict[agent_name].iloc[point]['Close']
                
                # Make prediction
                result = debate_system.predict_next_day(data_slice, actual_price)
            
            # Save results with ticker prefix
            debate_system.save_debate_summary(f'{self.data_dir}/{ticker}_debate_summary.csv')
            
            print(f"✅ Debate system completed for {ticker}")
            return True
            
        except Exception as e:
            print(f"❌ Debate system failed for {ticker}: {e}")
            return False
    
    def run_complete_pipeline(self, ticker):
        """Run complete pipeline for a ticker"""
        print(f"\n🚀 Running complete pipeline for {ticker}")
        print("=" * 60)
        
        # Step 1: Build datasets
        if not self.build_datasets(ticker):
            return False
        
        # Step 2: Train agents
        if not self.train_agents(ticker):
            return False
        
        # Step 3: Mutual learning
        if not self.run_mutual_learning(ticker):
            return False
        
        # Step 4: Debate system
        if not self.run_debate_system(ticker):
            return False
        
        print(f"\n🎉 Complete pipeline finished successfully for {ticker}!")
        print(f"\n💡 Next steps:")
        print(f"   1. Launch dashboard: streamlit run streamlit_dashboard.py")
        print(f"   2. Check results in: {self.data_dir}/{ticker}_debate_summary.csv")
        print(f"   3. View beta evolution: {self.data_dir}/{ticker}_beta_log.csv")
        
        return True
    
    def run(self):
        """Main run function"""
        ticker = self.get_ticker_from_user()
        
        if ticker is None:
            return
        
        success = self.run_complete_pipeline(ticker)
        
        if success:
            print(f"\n🎉 System completed successfully for {ticker}!")
        else:
            print(f"\n❌ System failed for {ticker}")

def main():
    """Main function"""
    system = TickerInputSystem()
    system.run()

if __name__ == "__main__":
    main()
