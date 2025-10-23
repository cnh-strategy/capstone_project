# ======================================================
# Stage 3: Online Debate & Consensus Prediction System
# 실시간 시장 데이터 기반 Debate 구조로 합의(consensus) 형성
# ======================================================

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agent_utils import AgentLoader
import warnings
warnings.filterwarnings("ignore")

class DebateSystem:
    """Stage 3: Online Debate & Consensus Prediction System"""
    
    def __init__(self, models_dir='models'):
        self.loader = AgentLoader(models_dir)
        self.loader.load_all_agents()
        
        # Debate parameters
        self.alpha = 0.2  # Learning rate for peer correction
        self.n_mc_samples = 10  # Number of Monte Carlo samples
        self.max_debate_rounds = 5  # Maximum debate rounds
        
        # EMA parameters for β updates
        self.ema_lambda = 0.8  # EMA smoothing factor (λ≈0.8)
        self.beta_ema = {agent: 1.0/len(self.loader.agents) for agent in self.loader.agents.keys()}
        
        # History tracking
        self.debate_history = []
        self.performance_history = []
        self.debate_summary = []
        
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
    
    def debate_round(self, data_dict, round_num):
        """Single debate round with peer correction"""
        predictions = {}
        uncertainties = {}
        mc_samples = {}
        
        # Step ①: Monte Carlo Dropout 수행
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
        
        # Step ②: 신뢰도 β 계산
        valid_uncertainties = [u for u in uncertainties.values() if u != float('inf')]
        if valid_uncertainties:
            beta_values = self.calculate_confidence_beta(valid_uncertainties)
        else:
            beta_values = [1.0/len(self.loader.agents)] * len(self.loader.agents)
        
        # Step ③: 상호보정 (보정식)
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
        
        # Step ④: 합의 산출
        valid_corrected = {k: v for k, v in corrected_predictions.items() if v is not None}
        if valid_corrected:
            consensus = np.average(list(valid_corrected.values()), weights=beta_values[:len(valid_corrected)])
        else:
            consensus = None
        
        return {
            'round': round_num,
            'predictions': predictions,
            'uncertainties': uncertainties,
            'beta_values': dict(zip(self.loader.agents.keys(), beta_values)),
            'corrected_predictions': corrected_predictions,
            'consensus': consensus,
            'mc_samples': mc_samples
        }
    
    def online_debate_prediction(self, data_dict, max_rounds=None):
        """Full online debate prediction process"""
        if max_rounds is None:
            max_rounds = self.max_debate_rounds
        
        debate_results = []
        current_predictions = {}
        
        # Initial predictions
        initial_round = self.debate_round(data_dict, 0)
        debate_results.append(initial_round)
        current_predictions = initial_round['corrected_predictions']
        
        # Iterative debate rounds
        for round_num in range(1, max_rounds):
            debate_round = self.debate_round(data_dict, round_num)
            debate_results.append(debate_round)
            
            # Update current predictions
            current_predictions = debate_round['corrected_predictions']
            
            # Check for convergence (optional early stopping)
            if round_num > 1:
                prev_consensus = debate_results[-2]['consensus']
                curr_consensus = debate_round['consensus']
                if prev_consensus is not None and curr_consensus is not None:
                    if abs(prev_consensus - curr_consensus) < 0.1:  # Convergence threshold
                        print(f"🔄 Debate converged at round {round_num}")
                        break
        
        # Final consensus
        final_result = debate_results[-1]
        
        # Step ⑤: EMA 피드백
        self.update_beta_ema(final_result['beta_values'])
        
        return debate_results, final_result
    
    def update_beta_ema(self, new_beta_values):
        """Update β weights using EMA: βᵢ ← λβᵢ + (1−λ)βᵢ(new)"""
        for agent_name in self.loader.agents.keys():
            if agent_name in new_beta_values:
                old_beta = self.beta_ema[agent_name]
                new_beta = new_beta_values[agent_name]
                self.beta_ema[agent_name] = self.ema_lambda * old_beta + (1 - self.ema_lambda) * new_beta
    
    def predict_next_day(self, data_dict, actual_price=None):
        """Predict next day's price using online debate"""
        print(f"🎯 Online Debate Prediction")
        print("-" * 40)
        
        # Run debate process
        debate_results, final_result = self.online_debate_prediction(data_dict)
        
        # Store in history
        self.debate_history.append({
            'debate_results': debate_results,
            'final_result': final_result,
            'actual_price': actual_price
        })
        
        # Log to summary
        summary_entry = {
            'round': len(self.debate_history),
            'consensus': final_result['consensus'],
            'actual': actual_price,
            'error': abs(final_result['consensus'] - actual_price) if final_result['consensus'] is not None and actual_price is not None else None,
            'beta_technical': final_result['beta_values']['technical'],
            'beta_fundamental': final_result['beta_values']['fundamental'],
            'beta_sentimental': final_result['beta_values']['sentimental'],
            'uncertainty_technical': final_result['uncertainties']['technical'],
            'uncertainty_fundamental': final_result['uncertainties']['fundamental'],
            'uncertainty_sentimental': final_result['uncertainties']['sentimental']
        }
        self.debate_summary.append(summary_entry)
        
        # Print results
        print(f"Final Consensus: ${final_result['consensus']:.2f}")
        print(f"\nIndividual Predictions:")
        for agent_name, pred in final_result['corrected_predictions'].items():
            if pred is not None:
                uncertainty = final_result['uncertainties'][agent_name]
                beta = final_result['beta_values'][agent_name]
                print(f"  {agent_name.title():>12}: ${pred:.2f} (σ={uncertainty:.3f}, β={beta:.3f})")
        
        print(f"\nCurrent EMA β Weights:")
        for agent_name, beta in self.beta_ema.items():
            print(f"  {agent_name.title():>12}: {beta:.3f}")
        
        if actual_price is not None:
            error = abs(final_result['consensus'] - actual_price)
            print(f"\nPrediction Error: ${error:.2f}")
            self.performance_history.append(error)
        
        return final_result
    
    def save_debate_summary(self, filename='debate_summary.csv'):
        """Save debate summary to CSV"""
        if self.debate_summary:
            df = pd.DataFrame(self.debate_summary)
            df.to_csv(filename, index=False)
            print(f"📊 Debate summary saved: {filename}")
    
    def plot_debate_results(self, save_path=None):
        """Plot debate results and visualization"""
        if not self.debate_history:
            print("No debate history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stage 3: Online Debate & Consensus Results', fontsize=16)
        
        # Extract data for plotting
        all_rounds = []
        all_consensus = []
        all_uncertainties = []
        all_betas = []
        
        for debate_session in self.debate_history:
            for round_result in debate_session['debate_results']:
                all_rounds.append(round_result['round'])
                all_consensus.append(round_result['consensus'])
                all_uncertainties.append(round_result['uncertainties'])
                all_betas.append(round_result['beta_values'])
        
        # 1. Consensus Predictions
        axes[0, 0].plot(all_consensus, marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Consensus Predictions')
        axes[0, 0].set_xlabel('Debate Round')
        axes[0, 0].set_ylabel('Consensus Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Agent Predictions
        agent_names = list(self.loader.agents.keys())
        colors = ['blue', 'red', 'green']
        
        for i, agent_name in enumerate(agent_names):
            predictions = []
            for d in all_rounds:
                if isinstance(d, dict) and 'predictions' in d:
                    predictions.append(d['predictions'].get(agent_name, 0))
                else:
                    predictions.append(0)
            valid_predictions = [p for p in predictions if p is not None and p != 0]
            valid_indices = [j for j, p in enumerate(predictions) if p is not None and p != 0]
            
            if valid_predictions:
                axes[0, 1].plot(valid_indices, valid_predictions, 
                               label=agent_name.title(), color=colors[i], linewidth=2)
        
        axes[0, 1].set_title('Individual Agent Predictions')
        axes[0, 1].set_xlabel('Debate Round')
        axes[0, 1].set_ylabel('Price Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Weight Evolution
        for i, agent_name in enumerate(agent_names):
            betas = [b.get(agent_name, 0) for b in all_betas]
            axes[1, 0].plot(betas, label=agent_name.title(), color=colors[i], linewidth=2)
        
        axes[1, 0].set_title('Confidence Weights (β) Evolution')
        axes[1, 0].set_xlabel('Debate Round')
        axes[1, 0].set_ylabel('β Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Variance
        variances = []
        for d in all_rounds:
            if isinstance(d, dict) and 'predictions' in d:
                preds = [v for v in d['predictions'].values() if v is not None]
                if len(preds) > 1:
                    variance = np.var(preds)
                    variances.append(variance)
                else:
                    variances.append(0)
            else:
                variances.append(0)
        
        axes[1, 1].plot(variances, marker='^', color='purple', linewidth=2, markersize=6)
        axes[1, 1].set_title('Prediction Variance (Disagreement)')
        axes[1, 1].set_xlabel('Debate Round')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Debate results plot saved: {save_path}")
        
        plt.show()
    
    def print_performance_summary(self):
        """Print overall performance summary"""
        print("\n📊 Online Debate Performance Summary")
        print("=" * 50)
        
        if self.performance_history:
            avg_error = np.mean(self.performance_history)
            std_error = np.std(self.performance_history)
            print(f"Average Prediction Error: ${avg_error:.2f} ± ${std_error:.2f}")
            print(f"Total Predictions: {len(self.performance_history)}")
        
        print(f"\nFinal EMA β Weights:")
        for agent_name, beta in self.beta_ema.items():
            print(f"  {agent_name.title():>12}: {beta:.3f}")

def main():
    """Main function for Stage 3 debate system"""
    print("🎭 Stage 3: Online Debate & Consensus Prediction System")
    print("=" * 60)
    
    # Initialize debate system
    ds = DebateSystem()
    
    # Load test data
    print("\n📊 Loading Test Data...")
    data_dict = {}
    for agent_type in ['technical', 'fundamental', 'sentimental']:
        data_dict[agent_type] = pd.read_csv(f'data/TSLA_{agent_type}_test.csv')
        print(f"✅ Loaded {agent_type} test data: {len(data_dict[agent_type])} samples")
    
    # Test multiple predictions
    test_points = [50, 100, 150, 180]
    
    for i, point in enumerate(test_points):
        print(f"\n{'='*60}")
        print(f"🎯 Prediction {i+1}/{len(test_points)} - Using data up to point {point}")
        print(f"{'='*60}")
        
        # Get data slice
        data_slice = {}
        actual_price = None
        
        for agent_name in data_dict.keys():
            data_slice[agent_name] = data_dict[agent_name].iloc[:point+1]
            if actual_price is None:  # Use technical data for actual price
                actual_price = data_dict[agent_name].iloc[point]['Close']
        
        # Make prediction
        result = ds.predict_next_day(data_slice, actual_price)
    
    # Print final summary
    ds.print_performance_summary()
    
    # Save results
    ds.save_debate_summary('debate_summary.csv')
    
    # Plot results
    print(f"\n📊 Generating plots...")
    ds.plot_debate_results('debate_results.png')
    
    print(f"\n🎉 Stage 3 debate system completed successfully!")
    return ds

if __name__ == "__main__":
    ds = main()
