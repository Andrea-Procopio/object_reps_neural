import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# SegFormer model parameter counts (calculated from actual models)
# These are the exact parameter counts from the Hugging Face models
segformer_params = {
    'b0': 3752694,    # 3.8M parameters
    'b1': 13715798,   # 13.7M parameters  
    'b2': 27461974,   # 27.5M parameters
    'b3': 47337814,   # 47.3M parameters
    'b4': 64108374,   # 64.1M parameters
    'b5': 84708694,   # 84.7M parameters
}

def load_correlation_results(model_dir):
    """Load correlation results from a model directory."""
    results_file = os.path.join(model_dir, 'correlation_results.json')
    if not os.path.exists(results_file):
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    return df

def get_max_correlation(df, correlation_type='spearman'):
    """Get the maximum positive correlation value."""
    if df is None or df.empty:
        return None
    
    # Get the maximum positive correlation value
    max_corr = df[correlation_type].max()
    return max_corr

def main():
    # Path to exp3b results
    results_dir = Path('exp3b_results')
    
    # Model directories we have results for
    model_dirs = [
        'nvidia_segformer-b0-finetuned-ade-512-512',
        'nvidia_segformer-b1-finetuned-ade-512-512', 
        'nvidia_segformer-b2-finetuned-ade-512-512',
        'nvidia_segformer-b3-finetuned-ade-512-512',
        'nvidia_segformer-b4-finetuned-ade-512-512',
        'nvidia_segformer-b5-finetuned-ade-640-640'
    ]
    
    # Collect data
    data = []
    all_correlation_data = []
    
    for model_dir in model_dirs:
        # Extract model variant (b0, b1, etc.)
        if 'segformer-b0' in model_dir:
            variant = 'b0'
        elif 'segformer-b1' in model_dir:
            variant = 'b1'
        elif 'segformer-b2' in model_dir:
            variant = 'b2'
        elif 'segformer-b3' in model_dir:
            variant = 'b3'
        elif 'segformer-b4' in model_dir:
            variant = 'b4'
        elif 'segformer-b5' in model_dir:
            variant = 'b5'
        else:
            continue
            
        # Load correlation results
        full_path = results_dir / model_dir
        df = load_correlation_results(full_path)
        
        if df is not None:
            # Get max correlation for both Pearson and Spearman
            max_pearson = get_max_correlation(df, 'pearson')
            max_spearman = get_max_correlation(df, 'spearman')
            
            data.append({
                'model': f'SegFormer-{variant.upper()}',
                'variant': variant,
                'parameters': segformer_params[variant],
                'max_pearson': max_pearson,
                'max_spearman': max_spearman
            })
            
            # Store all correlation data for detailed analysis
            df['model'] = f'SegFormer-{variant.upper()}'
            df['variant'] = variant
            df['parameters'] = segformer_params[variant]
            all_correlation_data.append(df)
            
            print(f"SegFormer-{variant.upper()}: {segformer_params[variant]/1e6:.1f}M params, "
                  f"max Pearson: {max_pearson:.3f}, max Spearman: {max_spearman:.3f}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(data)
    df_all_correlations = pd.concat(all_correlation_data, ignore_index=True)
    
    # Sort by parameter count
    df_results = df_results.sort_values('parameters')
    
    # Create the main plot
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Main correlation vs parameters plot
    ax1.plot(df_results['parameters'] / 1e6, df_results['max_pearson'], 
             'o-', label='Max Pearson Correlation', linewidth=2, markersize=8, color='blue')
    ax1.plot(df_results['parameters'] / 1e6, df_results['max_spearman'], 
             's-', label='Max Spearman Correlation', linewidth=2, markersize=8, color='red')
    
    # Add model labels
    for _, row in df_results.iterrows():
        ax1.annotate(row['model'], 
                    (row['parameters'] / 1e6, row['max_spearman']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left')
    
    ax1.set_xlabel('Number of Parameters (Millions)', fontsize=12)
    ax1.set_ylabel('Maximum Positive Correlation', fontsize=12)
    ax1.set_title('SegFormer Model Performance: Correlation vs Parameter Count', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df_results['parameters'] / 1e6)
    ax1.set_xticklabels([f'{p/1e6:.1f}M' for p in df_results['parameters']], rotation=45)
    
    # Plot 2: Correlation heatmap across thresholds
    # Create pivot table for heatmap
    pivot_pearson = df_all_correlations.pivot(index='threshold', columns='model', values='pearson')
    pivot_spearman = df_all_correlations.pivot(index='threshold', columns='model', values='spearman')
    
    # Plot Pearson heatmap
    im1 = ax2.imshow(pivot_pearson.T, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax2.set_title('Pearson Correlation Heatmap', fontsize=14)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Model', fontsize=12)
    ax2.set_yticks(range(len(pivot_pearson.columns)))
    ax2.set_yticklabels(pivot_pearson.columns)
    
    # Set x-axis ticks for thresholds (show every 10th threshold)
    threshold_indices = np.arange(0, len(pivot_pearson.index), 10)
    ax2.set_xticks(threshold_indices)
    ax2.set_xticklabels([f'{pivot_pearson.index[i]:.2f}' for i in threshold_indices], rotation=45)
    
    plt.colorbar(im1, ax=ax2, label='Correlation')
    
    # Plot 3: Spearman heatmap
    im2 = ax3.imshow(pivot_spearman.T, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax3.set_title('Spearman Correlation Heatmap', fontsize=14)
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Model', fontsize=12)
    ax3.set_yticks(range(len(pivot_spearman.columns)))
    ax3.set_yticklabels(pivot_spearman.columns)
    
    # Set x-axis ticks for thresholds
    ax3.set_xticks(threshold_indices)
    ax3.set_xticklabels([f'{pivot_spearman.index[i]:.2f}' for i in threshold_indices], rotation=45)
    
    plt.colorbar(im2, ax=ax3, label='Correlation')
    
    # Plot 4: Performance comparison bar chart
    x_pos = np.arange(len(df_results))
    width = 0.35
    
    ax4.bar(x_pos - width/2, df_results['max_pearson'], width, label='Max Pearson', color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, df_results['max_spearman'], width, label='Max Spearman', color='red', alpha=0.7)
    
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('Maximum Correlation', fontsize=12)
    ax4.set_title('Maximum Correlation by Model', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([row['model'] for _, row in df_results.iterrows()], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plt.savefig('segformer_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also save the original simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['parameters'] / 1e6, df_results['max_pearson'], 
             'o-', label='Max Pearson Correlation', linewidth=2, markersize=8)
    plt.plot(df_results['parameters'] / 1e6, df_results['max_spearman'], 
             's-', label='Max Spearman Correlation', linewidth=2, markersize=8)
    
    for _, row in df_results.iterrows():
        plt.annotate(row['model'], 
                    (row['parameters'] / 1e6, row['max_spearman']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left')
    
    plt.xlabel('Number of Parameters (Millions)', fontsize=12)
    plt.ylabel('Maximum Positive Correlation', fontsize=12)
    plt.title('SegFormer Model Performance: Correlation vs Parameter Count', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(df_results['parameters'] / 1e6, 
               [f'{p/1e6:.1f}M' for p in df_results['parameters']], 
               rotation=45)
    plt.tight_layout()
    plt.savefig('segformer_correlation_vs_params.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Number of models analyzed: {len(df_results)}")
    print(f"Parameter range: {df_results['parameters'].min()/1e6:.1f}M - {df_results['parameters'].max()/1e6:.1f}M")
    print(f"Max Pearson correlation range: {df_results['max_pearson'].min():.3f} - {df_results['max_pearson'].max():.3f}")
    print(f"Max Spearman correlation range: {df_results['max_spearman'].min():.3f} - {df_results['max_spearman'].max():.3f}")
    
    # Calculate correlation between parameter count and performance
    pearson_corr = np.corrcoef(df_results['parameters'], df_results['max_pearson'])[0, 1]
    spearman_corr = np.corrcoef(df_results['parameters'], df_results['max_spearman'])[0, 1]
    
    print(f"\nCorrelation between parameter count and performance:")
    print(f"Pearson: {pearson_corr:.3f}")
    print(f"Spearman: {spearman_corr:.3f}")
    
    # Find best performing model
    best_pearson_idx = df_results['max_pearson'].idxmax()
    best_spearman_idx = df_results['max_spearman'].idxmax()
    
    print(f"\nBest performing models:")
    print(f"Best Pearson: {df_results.loc[best_pearson_idx, 'model']} ({df_results.loc[best_pearson_idx, 'max_pearson']:.3f})")
    print(f"Best Spearman: {df_results.loc[best_spearman_idx, 'model']} ({df_results.loc[best_spearman_idx, 'max_spearman']:.3f})")
    
    # Save detailed results to CSV
    df_results.to_csv('segformer_correlation_results.csv', index=False)
    print(f"\nDetailed results saved to 'segformer_correlation_results.csv'")

if __name__ == "__main__":
    main()



