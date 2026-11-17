import torch
import numpy as np
from bayesian_group_lasso import BayesianGroupLasso, train_model
from visualization import (
    plot_weight_comparison,
    plot_training_progress,
    plot_group_sparsity,
    plot_prediction_scatter
)

def generate_synthetic_data(n_samples=1000, input_dim=20, num_groups=4, noise_level=0.1):
    """Generate synthetic data with group structure."""
    # Create group structure
    group_size = input_dim // num_groups
    X = torch.randn(n_samples, input_dim)
    
    # True weights with group sparsity
    w_true = torch.zeros(input_dim)
    active_groups = [0, 2]  # Only first and third groups are active
    for group in active_groups:
        start_idx = group * group_size
        end_idx = start_idx + group_size
        w_true[start_idx:end_idx] = torch.randn(group_size)
    
    # Generate target variable with noise
    y = X @ w_true + torch.randn(n_samples) * noise_level
    
    return X, y, w_true

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with uncertainty estimation."""
    mean_pred, std_pred = model.predict(X_test)
    mse = torch.mean((mean_pred - y_test)**2)
    
    # Calculate coverage of 95% prediction intervals
    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred
    coverage = torch.mean(((y_test >= lower) & (y_test <= upper)).float())
    
    return {
        'mse': mse.item(),
        'coverage': coverage.item()
    }

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    input_dim = 20
    num_groups = 4
    X, y, w_true = generate_synthetic_data(n_samples, input_dim, num_groups)
    
    # Split data into train and test sets
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    model = BayesianGroupLasso(input_dim=input_dim, num_groups=num_groups)
    history = train_model(model, X_train, y_train, num_epochs=1000, group_penalty=0.1)
    
    # Get the learned weights
    with torch.no_grad():
        w_learned = model.sample_weights()
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("\nTest Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"95% CI Coverage: {metrics['coverage']:.4f}")
    
    # Generate predictions with uncertainty
    mean_pred, std_pred = model.predict(X_test)
    
    # Visualizations
    plot_weight_comparison(w_true, w_learned, num_groups)
    plot_training_progress(history['loss'])
    plot_group_sparsity(w_learned, num_groups)
    plot_prediction_scatter(y_test, mean_pred)

if __name__ == "__main__":
    main() 