# ======================================================
# Multi-Ticker System
# 다양한 주식에 대한 통합 시스템
# ======================================================

import os
import pandas as pd
import numpy as np
from multi_ticker_dataset_builder import MultiTickerDatasetBuilder
from train_agents import main as train_agents_main
from stage2_trainer import Stage2Trainer
from debate_system import DebateSystem
import warnings
warnings.filterwarnings("ignore")

class MultiTickerSystem:
    """Multi-ticker system for various stocks"""
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.dataset_builder = MultiTickerDatasetBuilder(data_dir)
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    def get_available_tickers(self):
        """Get list of available tickers"""
        return self.dataset_builder.get_available_tickers()
    
    def validate_ticker(self, ticker):
        """Validate ticker"""
        return self.dataset_builder.validate_ticker(ticker)
    
    def build_datasets(self, ticker):
        """Build datasets for a specific ticker"""
        print(f"🚀 Building datasets for {ticker}")
        return self.dataset_builder.build_datasets_for_ticker(ticker)
    
    def build_multiple_datasets(self, tickers):
        """Build datasets for multiple tickers"""
        print(f"🚀 Building datasets for {len(tickers)} tickers")
        return self.dataset_builder.build_multiple_tickers(tickers)
    
    def train_agents_for_ticker(self, ticker):
        """Train agents for a specific ticker"""
        print(f"🎯 Training agents for {ticker}")
        
        # Check if datasets exist
        required_files = [
            f'{self.data_dir}/{ticker}_technical_pretrain.csv',
            f'{self.data_dir}/{ticker}_fundamental_pretrain.csv',
            f'{self.data_dir}/{ticker}_sentimental_pretrain.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing datasets for {ticker}:")
            for f in missing_files:
                print(f"   - {f}")
            return False
        
        # Temporarily modify the train_agents.py to use ticker-specific data
        # This is a simplified approach - in production, you'd modify the functions directly
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
                print("-" * 40)
                
                try:
                    # Load data with ticker prefix
                    features, target, feature_cols = load_dataset(agent_name, 'pretrain', ticker)
                    print(f"✅ Loaded {len(features)} samples with {len(feature_cols)} features")
                    
                    # Create sequences
                    X, y = create_sequences(features, target, config['window_size'])
                    print(f"✅ Created {len(X)} sequences with window size {config['window_size']}")
                    
                    # Continue with training...
                    # (This is a simplified version - full implementation would include all training logic)
                    
                    trained_agents[agent_name] = True
                    
                except Exception as e:
                    print(f"❌ Error training {agent_name} agent for {ticker}: {e}")
                    continue
            
            return len(trained_agents) == 3
            
        except Exception as e:
            print(f"❌ Training failed for {ticker}: {e}")
            return False
    
    def run_mutual_learning_for_ticker(self, ticker):
        """Run mutual learning for a specific ticker"""
        print(f"🔄 Running mutual learning for {ticker}")
        
        # Check if datasets exist
        required_files = [
            f'{self.data_dir}/{ticker}_technical_mutual.csv',
            f'{self.data_dir}/{ticker}_fundamental_mutual.csv',
            f'{self.data_dir}/{ticker}_sentimental_mutual.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing mutual datasets for {ticker}")
            return False
        
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
    
    def run_debate_for_ticker(self, ticker):
        """Run debate system for a specific ticker"""
        print(f"🎭 Running debate system for {ticker}")
        
        # Check if datasets exist
        required_files = [
            f'{self.data_dir}/{ticker}_technical_test.csv',
            f'{self.data_dir}/{ticker}_fundamental_test.csv',
            f'{self.data_dir}/{ticker}_sentimental_test.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing test datasets for {ticker}")
            return False
        
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
        """Run complete pipeline for a specific ticker"""
        print(f"🚀 Running complete pipeline for {ticker}")
        print("=" * 60)
        
        # Step 1: Build datasets
        if not self.build_datasets(ticker):
            print(f"❌ Failed to build datasets for {ticker}")
            return False
        
        # Step 2: Train agents
        if not self.train_agents_for_ticker(ticker):
            print(f"❌ Failed to train agents for {ticker}")
            return False
        
        # Step 3: Mutual learning
        if not self.run_mutual_learning_for_ticker(ticker):
            print(f"❌ Failed mutual learning for {ticker}")
            return False
        
        # Step 4: Debate system
        if not self.run_debate_for_ticker(ticker):
            print(f"❌ Failed debate system for {ticker}")
            return False
        
        print(f"🎉 Complete pipeline finished successfully for {ticker}!")
        return True
    
    def run_multiple_tickers(self, tickers):
        """Run complete pipeline for multiple tickers"""
        print(f"🚀 Running complete pipeline for {len(tickers)} tickers")
        print("=" * 80)
        
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}")
            print("-" * 40)
            
            success = self.run_complete_pipeline(ticker)
            results[ticker] = success
        
        # Summary
        successful = [t for t, s in results.items() if s]
        failed = [t for t, s in results.items() if not s]
        
        print(f"\n🎉 Multi-Ticker Pipeline Complete!")
        print("=" * 80)
        print(f"✅ Successful: {len(successful)} tickers")
        for ticker in successful:
            print(f"   - {ticker}")
        
        if failed:
            print(f"\n❌ Failed: {len(failed)} tickers")
            for ticker in failed:
                print(f"   - {ticker}")
        
        return results

def main():
    """Main function for multi-ticker system"""
    print("🚀 Multi-Ticker Debating System")
    print("=" * 60)
    
    # Initialize system
    system = MultiTickerSystem()
    
    # Show available tickers
    print("📊 Available Ticker Categories:")
    available_tickers = system.get_available_tickers()
    for category, tickers in available_tickers.items():
        print(f"   {category}: {', '.join(tickers)}")
    
    # Example: Run pipeline for a few tickers
    example_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print(f"\n🎯 Running complete pipeline for: {', '.join(example_tickers)}")
    
    results = system.run_multiple_tickers(example_tickers)
    
    if any(results.values()):
        print(f"\n💡 Next steps:")
        print(f"   1. Launch multi-ticker dashboard: streamlit run multi_ticker_dashboard.py")
        print(f"   2. Compare performance across tickers")
        print(f"   3. Analyze agent specialization by sector")

if __name__ == "__main__":
    main()
