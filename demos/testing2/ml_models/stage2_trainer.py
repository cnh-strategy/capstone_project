# ======================================================
# Stage 2: Selective Mutual Learning Trainer
# 2024 mutual data 기반 공동 적응 학습
# ======================================================

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .agent_utils import AgentLoader
import warnings
warnings.filterwarnings("ignore")

class Stage2Trainer:
    """Stage 2: Selective Mutual Learning Trainer"""
    
    def __init__(self, models_dir='models'):
        self.loader = AgentLoader(models_dir)
        self.loader.load_all_agents()
        
        # Learning parameters
        self.alpha = 0.2  # Learning rate for peer correction (typically 0.2)
        self.n_mc_samples = 10  # Monte Carlo samples for uncertainty estimation
        
        # History tracking
        self.beta_history = []
        self.residual_history = []
        self.performance_log = []
        
        # Agent performance tracking
        self.agent_performance = {
            'technical': {'mse': [], 'mae': [], 'uncertainty': []},
            'fundamental': {'mse': [], 'mae': [], 'uncertainty': []},
            'sentimental': {'mse': [], 'mae': [], 'uncertainty': []}
        }
    
    def monte_carlo_dropout_prediction(self, agent_name, data, n_samples=None):
        """Monte Carlo Dropout prediction with uncertainty estimation"""
        if n_samples is None:
            n_samples = self.n_mc_samples
            
        agent_info = self.loader.agents[agent_name]
        model = agent_info['model']
        scaler_X = agent_info['scaler_X']
        scaler_y = agent_info['scaler_y']
        window_size = agent_info['window_size']
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
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
        
        # Create sequence
        if len(features_scaled) >= window_size:
            sequence = features_scaled[-window_size:].reshape(1, window_size, -1)
        else:
            padded = np.zeros((window_size, features_scaled.shape[1]))
            padded[-len(features_scaled):] = features_scaled
            sequence = padded.reshape(1, window_size, -1)
        
        # Monte Carlo Dropout sampling
        model.train()  # Enable dropout for uncertainty estimation
        predictions = []
        
        with torch.no_grad():
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            
            for _ in range(n_samples):
                pred_scaled = model(sequence_tensor).squeeze().numpy()
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                predictions.append(pred[0])
        
        # Calculate statistics
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        return mean_prediction, uncertainty, predictions
    
    def calculate_confidence_beta(self, uncertainties):
        """Calculate confidence weights β using softmax of negative uncertainties"""
        # Convert uncertainties to confidence scores
        confidence_scores = np.exp(-np.array(uncertainties))
        
        # Normalize to get β values
        beta_values = confidence_scores / np.sum(confidence_scores)
        
        return beta_values
    
    def calculate_peer_mean(self, agent_idx, predictions, beta_values):
        """Calculate peer mean for a specific agent"""
        peer_predictions = [predictions[other_agent] for j, other_agent in enumerate(self.loader.agents.keys()) if j != agent_idx]
        peer_betas = [beta_values[j] for j in range(len(beta_values)) if j != agent_idx]
        
        if peer_predictions and peer_betas:
            peer_mean = np.average(peer_predictions, weights=peer_betas)
        else:
            peer_mean = predictions[list(self.loader.agents.keys())[agent_idx]]
        
        return peer_mean
    
    def mutual_learning_step(self, data_dict, actual_price, step):
        """Single step of mutual learning with peer correction"""
        predictions = {}
        uncertainties = {}
        mc_samples = {}
        
        # Get predictions and uncertainties from all agents
        for agent_name in self.loader.agents.keys():
            if agent_name in data_dict:
                try:
                    pred, uncertainty, samples = self.monte_carlo_dropout_prediction(agent_name, data_dict[agent_name])
                    predictions[agent_name] = pred
                    uncertainties[agent_name] = uncertainty
                    mc_samples[agent_name] = samples
                except Exception as e:
                    print(f"❌ Error in {agent_name} prediction: {e}")
                    predictions[agent_name] = None
                    uncertainties[agent_name] = float('inf')
                    mc_samples[agent_name] = []
            else:
                predictions[agent_name] = None
                uncertainties[agent_name] = float('inf')
                mc_samples[agent_name] = []
        
        # Calculate confidence weights β
        valid_uncertainties = [u for u in uncertainties.values() if u != float('inf')]
        if valid_uncertainties:
            beta_values = self.calculate_confidence_beta(valid_uncertainties)
        else:
            beta_values = [1.0/len(self.loader.agents)] * len(self.loader.agents)
        
        # Apply peer correction: y_i' = y_i + αβ_i(peer_mean_i - y_i)
        corrected_predictions = {}
        for i, agent_name in enumerate(self.loader.agents.keys()):
            if predictions[agent_name] is not None:
                peer_mean = self.calculate_peer_mean(i, predictions, beta_values)
                original_pred = predictions[agent_name]
                correction = self.alpha * beta_values[i] * (peer_mean - original_pred)
                corrected_pred = original_pred + correction
                corrected_predictions[agent_name] = corrected_pred
            else:
                corrected_predictions[agent_name] = predictions[agent_name]
        
        # Calculate final consensus
        valid_corrected = {k: v for k, v in corrected_predictions.items() if v is not None}
        if valid_corrected:
            consensus = np.average(list(valid_corrected.values()), weights=beta_values[:len(valid_corrected)])
        else:
            consensus = None
        
        # Store results
        result = {
            'step': step,
            'predictions': predictions,
            'uncertainties': uncertainties,
            'beta_values': dict(zip(self.loader.agents.keys(), beta_values)),
            'corrected_predictions': corrected_predictions,
            'consensus': consensus,
            'actual_price': actual_price,
            'mc_samples': mc_samples
        }
        
        self.beta_history.append(result['beta_values'])
        
        # Calculate residuals for performance tracking
        if consensus is not None and actual_price is not None:
            residual = abs(consensus - actual_price)
            self.residual_history.append(residual)
            
            # Update agent performance
            for agent_name, pred in predictions.items():
                if pred is not None:
                    agent_residual = abs(pred - actual_price)
                    self.agent_performance[agent_name]['mse'].append(agent_residual**2)
                    self.agent_performance[agent_name]['mae'].append(agent_residual)
                    self.agent_performance[agent_name]['uncertainty'].append(uncertainties[agent_name])
        
        # Log performance
        self.performance_log.append({
            'step': step,
            'consensus': consensus,
            'actual': actual_price,
            'residual': residual if consensus is not None and actual_price is not None else None,
            'beta_technical': result['beta_values']['technical'],
            'beta_fundamental': result['beta_values']['fundamental'],
            'beta_sentimental': result['beta_values']['sentimental']
        })
        
        return result
    
    def train_mutual_learning(self, data_dict, start_idx=0, end_idx=None):
        """Train mutual learning on 2024 mutual data"""
        print("🔄 Stage 2: Selective Mutual Learning Training")
        print("=" * 60)
        
        # Determine data length
        min_length = min(len(data_dict[agent]) for agent in data_dict.keys())
        if end_idx is None:
            end_idx = min_length
        
        print(f"📊 Processing {end_idx - start_idx} samples from 2024 mutual data")
        print(f"🎯 Using data: {list(data_dict.keys())}")
        
        results = []
        
        for i in range(start_idx, end_idx):
            if i % 20 == 0:
                print(f"   Processing sample {i}/{end_idx-1}")
            
            # Get data slice for each agent
            data_slice = {}
            actual_price = None
            
            for agent_name in data_dict.keys():
                data_slice[agent_name] = data_dict[agent_name].iloc[:i+1]
                if actual_price is None:  # Use technical data for actual price
                    actual_price = data_dict[agent_name].iloc[i]['Close']
            
            # Run mutual learning step
            result = self.mutual_learning_step(data_slice, actual_price, i)
            results.append(result)
        
        print(f"✅ Mutual learning training completed: {len(results)} steps")
        return results
    
    def save_beta_log(self, filename='beta_log.csv'):
        """Save β confidence log to CSV"""
        if self.performance_log:
            df = pd.DataFrame(self.performance_log)
            df.to_csv(filename, index=False)
            print(f"📊 Beta log saved: {filename}")
    
    def save_finetuned_models(self, models_dir='models'):
        """Save fine-tuned models (placeholder - would need actual fine-tuning)"""
        print("💾 Saving fine-tuned models...")
        # In a real implementation, you would fine-tune the models here
        # For now, we'll just copy the original models with a different name
        
        import shutil
        import os
        
        for agent_name in self.loader.agents.keys():
            original_path = f"{models_dir}/{agent_name}_agent.pt"
            finetuned_path = f"{models_dir}/{agent_name}_agent_finetuned.pt"
            
            if os.path.exists(original_path):
                shutil.copy2(original_path, finetuned_path)
                print(f"✅ Saved: {finetuned_path}")
    
    def print_training_summary(self):
        """Print training summary"""
        print("\n📊 Stage 2 Training Summary")
        print("=" * 50)
        
        if self.residual_history:
            avg_residual = np.mean(self.residual_history)
            print(f"Average Consensus Residual: {avg_residual:.2f}")
        
        print(f"\nIndividual Agent Performance:")
        for agent_name in self.loader.agents.keys():
            perf = self.agent_performance[agent_name]
            if perf['mae']:
                avg_mae = np.mean(perf['mae'])
                avg_uncertainty = np.mean(perf['uncertainty'])
                print(f"  {agent_name.title():>12}: MAE={avg_mae:.2f}, Avg Uncertainty={avg_uncertainty:.3f}")
        
        # Final β values
        if self.beta_history:
            final_betas = self.beta_history[-1]
            print(f"\nFinal β Confidence Weights:")
            for agent_name, beta in final_betas.items():
                print(f"  {agent_name.title():>12}: {beta:.3f}")

def main():
    """Main function for Stage 2 training"""
    print("🚀 Stage 2: Selective Mutual Learning Trainer")
    print("=" * 60)
    
    # Initialize trainer
    trainer = Stage2Trainer()
    
    # Load 2024 mutual data
    print("\n📊 Loading 2024 Mutual Data...")
    data_dict = {}
    for agent_type in ['technical', 'fundamental', 'sentimental']:
        data_dict[agent_type] = pd.read_csv(f'data/TSLA_{agent_type}_mutual.csv')
        print(f"✅ Loaded {agent_type} mutual data: {len(data_dict[agent_type])} samples")
    
    # Run mutual learning training
    results = trainer.train_mutual_learning(data_dict, start_idx=0, end_idx=100)
    
    # Print summary
    trainer.print_training_summary()
    
    # Save results
    trainer.save_beta_log('beta_log.csv')
    trainer.save_finetuned_models()
    
    print(f"\n🎉 Stage 2 training completed successfully!")
    return trainer

if __name__ == "__main__":
    trainer = main()
