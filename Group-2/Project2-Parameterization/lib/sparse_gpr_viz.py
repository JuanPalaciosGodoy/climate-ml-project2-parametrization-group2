# visualizations from sparse GPR analysis
# in separate library for organization

import matplotlib as plt
import numpy as np

# Cell 1: Overall Performance Metrics Comparison
def plot_performance_metrics(results):
    """
    Compare overall performance metrics between NN and NN+GP models
    
    Parameters:
    -----------
    results : dict
        Results dictionary from compare_models
    """
    # Create metrics dictionary
    metrics = {
        'Mean Absolute Error': [results['nn_mae'], results['gp_mae']],
        'Root Mean Square Error': [results['nn_rmse'], results['gp_rmse']]
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create grouped bar chart
    ax.bar(x - width/2, [metrics[m][0] for m in metrics], width, label='Neural Network')
    ax.bar(x + width/2, [metrics[m][1] for m in metrics], width, label='NN + GP')
    
    # Add labels and styling
    ax.set_ylabel('Error')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate improvement percentages
    mae_improvement = (1 - results['gp_mae']/results['nn_mae']) * 100
    rmse_improvement = (1 - results['gp_rmse']/results['nn_rmse']) * 100
    
    print(f"MAE Improvement: {mae_improvement:.2f}%")
    print(f"RMSE Improvement: {rmse_improvement:.2f}%")

# Cell 2: Error by Vertical Level
def plot_error_by_level(results):
    """
    Compare error across different vertical levels
    
    Parameters:
    -----------
    results : dict
        Results dictionary from compare_models
    """
    # Extract predictions
    nn_pred = results['nn_pred']
    gp_pred = results['gp_pred']
    true_vals = results['true_values']
    
    # Calculate MAE by level
    nn_mae_by_level = np.mean(np.abs(nn_pred - true_vals), axis=0)
    gp_mae_by_level = np.mean(np.abs(gp_pred - true_vals), axis=0)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 12))
    levels = range(16)
    
    ax.barh(levels, nn_mae_by_level, alpha=0.6, label='Neural Network')
    ax.barh(levels, gp_mae_by_level, alpha=0.6, label='NN + GP')
    
    # Add labels and styling
    ax.set_yticks(levels)
    ax.set_yticklabels([f'Level {16-i}' for i in levels])
    ax.set_ylabel('Vertical Level')
    ax.set_xlabel('Mean Absolute Error')
    ax.set_title('Error by Vertical Level')
    ax.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate improvement by level
    improvement_by_level = (1 - gp_mae_by_level/nn_mae_by_level) * 100
    
    # Print levels with most improvement
    max_improvement_idx = np.argmax(improvement_by_level)
    print(f"Level with greatest improvement: Level {16-max_improvement_idx} ({improvement_by_level[max_improvement_idx]:.2f}%)")
    
    # Print levels with least improvement
    min_improvement_idx = np.argmin(improvement_by_level)
    print(f"Level with least improvement: Level {16-min_improvement_idx} ({improvement_by_level[min_improvement_idx]:.2f}%)")

# Cell 3: Sample Profile Predictions with Uncertainty
def plot_sample_profiles(results):
    """
    Plot sample vertical profiles with uncertainty bands
    
    Parameters:
    -----------
    results : dict
        Results dictionary from compare_models
    """
    # Extract predictions
    nn_pred = results['nn_pred']
    gp_pred = results['gp_pred']
    true_vals = results['true_values']
    
    # Select random samples to visualize
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(nn_pred.shape[0], 4, replace=False)
    
    # Define vertical levels
    sig_levels = np.linspace(0, 1, nn_pred.shape[1] + 2)[1:-1]  # Remove endpoints
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Plot true values
        ax.plot(true_vals[idx], sig_levels, 'ko-', label='True')
        
        # Plot NN prediction
        ax.plot(nn_pred[idx], sig_levels, 'b.-', label='NN')
        
        # Plot GP prediction with uncertainty
        ax.plot(gp_pred[idx], sig_levels, 'r.-', label='NN+GP')
        ax.fill_betweenx(sig_levels, 
                        results['gp_lower'][idx], 
                        results['gp_upper'][idx], 
                        color='r', alpha=0.2, label='95% CI')
        
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Diffusivity')
        if i % 2 == 0:
            ax.set_ylabel('Normalized Depth (0=bottom, 1=top)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate how often true values fall within confidence intervals
    in_ci = ((results['true_values'] >= results['gp_lower']) & 
             (results['true_values'] <= results['gp_upper']))
    ci_coverage = np.mean(in_ci) * 100
    print(f"True values fall within 95% confidence intervals {ci_coverage:.2f}% of the time")

# Cell 4: Parity Plots (Predicted vs Actual)
def plot_parity_comparison(results):
    """
    Create parity plots comparing model predictions with true values
    
    Parameters:
    -----------
    results : dict
        Results dictionary from compare_models
    """
    # Extract predictions
    nn_pred = results['nn_pred']
    gp_pred = results['gp_pred']
    true_vals = results['true_values']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Flatten arrays for scatter plots
    nn_flat = nn_pred.flatten()
    gp_flat = gp_pred.flatten()
    true_flat = true_vals.flatten()
    
    # Calculate limits to use same scale for both plots
    min_val = min(nn_flat.min(), gp_flat.min(), true_flat.min())
    max_val = max(nn_flat.max(), gp_flat.max(), true_flat.max())
    
    # NN parity plot
    axes[0].scatter(true_flat, nn_flat, alpha=0.3, s=10)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--')
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Neural Network')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # NN+GP parity plot
    axes[1].scatter(true_flat, gp_flat, alpha=0.3, s=10)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('NN + GP')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate R² values
    from sklearn.metrics import r2_score
    nn_r2 = r2_score(true_flat, nn_flat)
    gp_r2 = r2_score(true_flat, gp_flat)
    
    print(f"Neural Network R²: {nn_r2:.4f}")
    print(f"NN+GP R²: {gp_r2:.4f}")
    print(f"R² Improvement: {(gp_r2 - nn_r2):.4f}")

# Cell 5: Uncertainty Distribution by Vertical Level
def plot_uncertainty_distribution(results):
    """
    Visualize the uncertainty distribution across vertical levels
    
    Parameters:
    -----------
    results : dict
        Results dictionary from compare_models
    """
    # Calculate uncertainty as the width of the confidence interval
    gp_uncertainty = results['gp_upper'] - results['gp_lower']
    
    # Prepare data for violin plot
    level_data = [gp_uncertainty[:, i] for i in range(16)]
    
    # Create violin plot
    plt.figure(figsize=(14, 8))
    violin_parts = plt.violinplot(level_data, showmeans=True, showmedians=True)
    
    # Customize plot appearance
    plt.xticks(range(1, 17), [f"Level {16-i}" for i in range(16)], rotation=45)
    plt.ylabel('Uncertainty (95% CI Width)')
    plt.title('Uncertainty Distribution by Vertical Level')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print statistics about uncertainty
    mean_uncertainty = np.mean(gp_uncertainty, axis=0)
    max_uncertainty_level = np.argmax(mean_uncertainty)
    min_uncertainty_level = np.argmin(mean_uncertainty)
    
    print(f"Level with highest uncertainty: Level {16-max_uncertainty_level} (mean width: {mean_uncertainty[max_uncertainty_level]:.4f})")
    print(f"Level with lowest uncertainty: Level {16-min_uncertainty_level} (mean width: {mean_uncertainty[min_uncertainty_level]:.4f})")
    
    # Calculate correlation between uncertainty and depth
    depth_values = np.repeat(np.arange(16)/15, gp_uncertainty.shape[0]).reshape(16, -1).T
    corr = np.corrcoef(depth_values.flatten(), gp_uncertainty.flatten())[0, 1]
    print(f"Correlation between depth and uncertainty: {corr:.4f}")


# Cell 6: Uncertainty vs. Error Relationship
def plot_uncertainty_vs_error(results):
    """
    Analyze the relationship between model uncertainty and prediction errors
    
    Parameters:
    -----------
    results : dict
        Results dictionary from compare_models
    """
    # Calculate error and uncertainty
    gp_errors = np.abs(results['gp_pred'] - results['true_values'])
    gp_uncertainty = results['gp_upper'] - results['gp_lower']
    
    # Flatten arrays for scatter plot
    flat_uncertainty = gp_uncertainty.flatten()
    flat_errors = gp_errors.flatten()
    
    # Calculate correlation
    from scipy import stats
    correlation, p_value = stats.pearsonr(flat_uncertainty, flat_errors)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(flat_uncertainty, flat_errors, alpha=0.3, s=5)
    plt.xlabel('Uncertainty (95% CI Width)')
    plt.ylabel('Absolute Error')
    plt.title('Relationship between Uncertainty and Prediction Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trendline
    z = np.polyfit(flat_uncertainty, flat_errors, 1)
    p = np.poly1d(z)
    plt.plot(np.unique(flat_uncertainty), p(np.unique(flat_uncertainty)), "r--", 
             label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}, r={correlation:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Correlation between uncertainty and error: {correlation:.4f} (p-value: {p_value:.4e})")
    
    # Additional analysis - uncertainty calibration
    # Calculate what percentage of true values fall within confidence intervals
    in_ci = ((results['true_values'] >= results['gp_lower']) & 
             (results['true_values'] <= results['gp_upper']))
    coverage = np.mean(in_ci) * 100
    
    print(f"Overall, {coverage:.2f}% of true values fall within the 95% confidence intervals")
    
    # Calculate coverage by level
    coverage_by_level = np.mean(in_ci, axis=0) * 100
    for i in range(16):
        print(f"Level {16-i}: {coverage_by_level[i]:.2f}% coverage")