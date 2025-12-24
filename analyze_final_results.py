import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results(csv_file="latestsummary.csv"):
    df = pd.read_csv(csv_file)
    
    final_results = df[df['epoch'] == 2].copy()
    
    print("=" * 80)
    print("WHITE-BOX VS BLACK-BOX KNOWLEDGE DISTILLATION - FINAL RESULTS")
    print("=" * 80)
    print()
    
    stats = final_results.groupby('distill_type').agg({
        'validation_accuracy': ['mean', 'std', 'min', 'max', 'count'],
        'validation_accuracy_sst2': ['mean', 'std'],
        'validation_accuracy_mmlu': ['mean', 'std'],
        'validation_accuracy_gsm8k': ['mean', 'std'],
        'validation_loss': ['mean', 'std']
    }).round(4)
    
    print("OVERALL VALIDATION ACCURACY (Final Epoch, N=7 seeds per method)")
    print("-" * 80)
    overall = stats['validation_accuracy']
    overall.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'N']
    print(overall)
    print()
    
    print("TASK-SPECIFIC BREAKDOWN")
    print("-" * 80)
    for task in ['sst2', 'mmlu', 'gsm8k']:
        col = f'validation_accuracy_{task}'
        task_stats = stats[col]
        print(f"\n{task.upper()}:")
        print(f"  {'Method':<20} {'Mean':<10} {'Std Dev':<10}")
        for method in ['black_box', 'hidden_state', 'attention', 'combined']:
            mean_val = task_stats.loc[method, 'mean']
            std_val = task_stats.loc[method, 'std']
            print(f"  {method:<20} {mean_val:<10.4f} {std_val:<10.4f}")
    
    print()
    print("VALIDATION LOSS (Lower is better)")
    print("-" * 80)
    loss_stats = stats['validation_loss']
    loss_stats.columns = ['Mean', 'Std Dev']
    print(loss_stats)
    print()
    
    print("KEY FINDINGS")
    print("-" * 80)
    
    black_box_mean = overall.loc['black_box', 'Mean']
    hidden_mean = overall.loc['hidden_state', 'Mean']
    attention_mean = overall.loc['attention', 'Mean']
    combined_mean = overall.loc['combined', 'Mean']
    
    best_method = overall['Mean'].idxmax()
    best_score = overall.loc[best_method, 'Mean']
    
    print(f"1. Best Overall Method: {best_method} ({best_score:.4f})")
    print(f"2. Black-Box Performance: {black_box_mean:.4f} ± {overall.loc['black_box', 'Std Dev']:.4f}")
    print(f"3. White-Box (Hidden) Performance: {hidden_mean:.4f} ± {overall.loc['hidden_state', 'Std Dev']:.4f}")
    print(f"4. White-Box (Attention) Performance: {attention_mean:.4f} ± {overall.loc['attention', 'Std Dev']:.4f}")
    print(f"5. White-Box (Combined) Performance: {combined_mean:.4f} ± {overall.loc['combined', 'Std Dev']:.4f}")
    print()
    
    diff_hidden = black_box_mean - hidden_mean
    diff_attention = black_box_mean - attention_mean
    diff_combined = black_box_mean - combined_mean
    
    print("COMPARISON TO BLACK-BOX BASELINE:")
    print(f"  Hidden State: {diff_hidden:+.4f} ({'worse' if diff_hidden > 0 else 'better'})")
    print(f"  Attention: {diff_attention:+.4f} ({'worse' if diff_attention > 0 else 'better'})")
    print(f"  Combined: {diff_combined:+.4f} ({'worse' if diff_combined > 0 else 'better'})")
    print()
    
    print("TASK-SPECIFIC INSIGHTS:")
    print("-" * 80)
    for task in ['sst2', 'mmlu', 'gsm8k']:
        col = f'validation_accuracy_{task}'
        task_data = final_results.groupby('distill_type')[col].mean()
        best_task = task_data.idxmax()
        best_task_score = task_data.max()
        print(f"{task.upper()}: Best = {best_task} ({best_task_score:.4f})")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    analyze_results()

