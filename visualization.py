import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

def plot_weight_comparison(w_true, w_learned, num_groups):
    """Plot true vs learned weights with group structure visualization."""
    plt.figure(figsize=(12, 6))
    
    # Plot true weights
    plt.subplot(2, 1, 1)
    plt.title('True Weights')
    sns.heatmap(w_true.reshape(1, -1), cmap='RdBu_r', center=0)
    plt.colorbar()
    
    # Add group separators
    group_size = len(w_true) // num_groups
    for i in range(1, num_groups):
        plt.axvline(x=i*group_size, color='black', linestyle='--', alpha=0.3)
    
    # Plot learned weights
    plt.subplot(2, 1, 2)
    plt.title('Learned Weights')
    sns.heatmap(w_learned.reshape(1, -1), cmap='RdBu_r', center=0)
    plt.colorbar()
    
    # Add group separators
    for i in range(1, num_groups):
        plt.axvline(x=i*group_size, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_training_progress(losses):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def plot_group_sparsity(w_learned, num_groups):
    """Plot group-level sparsity patterns."""
    group_size = len(w_learned) // num_groups
    group_norms = []
    
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_norm = torch.norm(w_learned[start_idx:end_idx]).item()
        group_norms.append(group_norm)
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(num_groups), group_norms)
    plt.title('Group-Level Sparsity')
    plt.xlabel('Group Index')
    plt.ylabel('L2 Norm')
    plt.show()

def plot_prediction_scatter(y_true, y_pred):
    """Plot scatter plot of true vs predicted values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true.numpy(), y_pred.detach().numpy(), alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show() 