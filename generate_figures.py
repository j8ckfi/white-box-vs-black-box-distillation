import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_data(csv_file="latestsummary.csv"):
    df = pd.read_csv(csv_file)
    return df

def plot_learning_curves(df, save_path="figures/"):
    """Figure 1: Learning curves for all methods"""
    Path(save_path).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['black_box', 'hidden_state', 'attention', 'combined']
    method_labels = ['Black-Box', 'White-Box (Hidden)', 'White-Box (Attention)', 'White-Box (Combined)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for method, label, color in zip(methods, method_labels, colors):
        method_data = df[df['distill_type'] == method]
        epochs = method_data.groupby('epoch')['validation_accuracy'].agg(['mean', 'std']).reset_index()
        
        ax.plot(epochs['epoch'], epochs['mean'], marker='o', label=label, color=color, linewidth=2)
        ax.fill_between(epochs['epoch'], 
                        epochs['mean'] - epochs['std'],
                        epochs['mean'] + epochs['std'],
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontweight='bold')
    ax.set_title('Learning Curves: Validation Accuracy Over Training', fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_ylim([0.85, 1.0])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}learning_curves.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}learning_curves.png", bbox_inches='tight')
    print(f"Saved: {save_path}learning_curves.pdf")
    plt.close()

def plot_final_accuracy_comparison(df, save_path="figures/"):
    """Figure 2: Final accuracy comparison with confidence intervals"""
    Path(save_path).mkdir(exist_ok=True)
    
    final_results = df[df['epoch'] == 2].copy()
    
    methods = ['black_box', 'hidden_state', 'attention', 'combined']
    method_labels = ['Black-Box', 'White-Box\n(Hidden)', 'White-Box\n(Attention)', 'White-Box\n(Combined)']
    
    means = []
    stds = []
    ci_lowers = []
    ci_uppers = []
    
    for method in methods:
        scores = final_results[final_results['distill_type'] == method]['validation_accuracy'].values
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)
        se = std / np.sqrt(n)
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se
        
        means.append(mean)
        stds.append(std)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(method_labels))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(x_pos, means, yerr=[np.array(means) - np.array(ci_lowers), 
                                      np.array(ci_uppers) - np.array(means)],
                  capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Validation Accuracy', fontweight='bold')
    ax.set_title('Final Validation Accuracy Comparison (Epoch 2)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels)
    ax.set_ylim([0.90, 0.97])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}final_accuracy_comparison.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}final_accuracy_comparison.png", bbox_inches='tight')
    print(f"Saved: {save_path}final_accuracy_comparison.pdf")
    plt.close()

def plot_task_breakdown(df, save_path="figures/"):
    """Figure 3: Task-specific breakdown"""
    Path(save_path).mkdir(exist_ok=True)
    
    final_results = df[df['epoch'] == 2].copy()
    
    tasks = ['sst2', 'mmlu', 'gsm8k']
    task_labels = ['SST-2', 'MMLU', 'GSM8K']
    methods = ['black_box', 'hidden_state', 'attention', 'combined']
    method_labels = ['Black-Box', 'Hidden', 'Attention', 'Combined']
    
    x = np.arange(len(task_labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = []
        for task in tasks:
            col = f'validation_accuracy_{task}'
            scores = final_results[final_results['distill_type'] == method][col].values
            means.append(np.mean(scores))
        
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Task-Specific Performance Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend()
    ax.set_ylim([0.85, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}task_breakdown.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}task_breakdown.png", bbox_inches='tight')
    print(f"Saved: {save_path}task_breakdown.pdf")
    plt.close()

def plot_loss_curves(df, save_path="figures/"):
    """Figure 4: Validation loss curves"""
    Path(save_path).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['black_box', 'hidden_state', 'attention', 'combined']
    method_labels = ['Black-Box', 'White-Box (Hidden)', 'White-Box (Attention)', 'White-Box (Combined)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for method, label, color in zip(methods, method_labels, colors):
        method_data = df[df['distill_type'] == method]
        # Filter out NaN values
        method_data = method_data[method_data['validation_loss'].notna()]
        epochs = method_data.groupby('epoch')['validation_loss'].agg(['mean', 'std']).reset_index()
        
        if len(epochs) > 0:
            ax.plot(epochs['epoch'], epochs['mean'], marker='o', label=label, color=color, linewidth=2)
            ax.fill_between(epochs['epoch'], 
                            epochs['mean'] - epochs['std'],
                            epochs['mean'] + epochs['std'],
                            alpha=0.2, color=color)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Learning Curves: Validation Loss Over Training', fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}loss_curves.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}loss_curves.png", bbox_inches='tight')
    print(f"Saved: {save_path}loss_curves.pdf")
    plt.close()

def main():
    print("Generating figures for arXiv paper...")
    df = load_data()
    
    plot_learning_curves(df)
    plot_final_accuracy_comparison(df)
    plot_task_breakdown(df)
    plot_loss_curves(df)
    
    print("\nAll figures generated successfully!")
    print("Figures saved in 'figures/' directory")

if __name__ == "__main__":
    main()

