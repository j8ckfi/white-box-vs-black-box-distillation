import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def compute_statistical_significance(df, csv_file="latestsummary.csv"):
    """
    Compute statistical significance tests comparing white-box methods vs black-box baseline.
    """
    df = pd.read_csv(csv_file)
    final_results = df[df['epoch'] == 2].copy()
    
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    print()
    
    black_box = final_results[final_results['distill_type'] == 'black_box']['validation_accuracy'].values
    hidden_state = final_results[final_results['distill_type'] == 'hidden_state']['validation_accuracy'].values
    attention = final_results[final_results['distill_type'] == 'attention']['validation_accuracy'].values
    combined = final_results[final_results['distill_type'] == 'combined']['validation_accuracy'].values
    
    methods = {
        'Hidden State': hidden_state,
        'Attention': attention,
        'Combined': combined
    }
    
    print("COMPARISON TO BLACK-BOX BASELINE")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean Diff':<15} {'t-statistic':<15} {'p-value':<15} {'Significant':<15} {'Effect Size (d)':<15}")
    print("-" * 80)
    
    results = []
    
    for method_name, method_scores in methods.items():
        # Paired t-test (assuming same seeds across methods)
        t_stat, p_value = stats.ttest_rel(method_scores, black_box)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(method_scores) + np.var(black_box)) / 2)
        cohens_d = (np.mean(method_scores) - np.mean(black_box)) / pooled_std
        
        mean_diff = np.mean(method_scores) - np.mean(black_box)
        
        # Significance levels
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"{method_name:<20} {mean_diff:+.4f}          {t_stat:>8.3f}        {p_value:>8.4f}        {sig:<15} {cohens_d:>8.3f}")
        
        results.append({
            'method': method_name,
            'mean_diff': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': sig,
            'cohens_d': cohens_d
        })
    
    print()
    print("NOTE: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print()
    
    # Confidence intervals
    print("95% CONFIDENCE INTERVALS")
    print("-" * 80)
    
    all_methods = {
        'Black-Box': black_box,
        'Hidden State': hidden_state,
        'Attention': attention,
        'Combined': combined
    }
    
    ci_results = []
    for method_name, scores in all_methods.items():
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)
        se = std / np.sqrt(n)
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se
        
        print(f"{method_name:<20} {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        ci_results.append({
            'method': method_name,
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
    
    print()
    
    # Task-specific significance
    print("TASK-SPECIFIC STATISTICAL TESTS")
    print("-" * 80)
    
    for task in ['sst2', 'mmlu', 'gsm8k']:
        col = f'validation_accuracy_{task}'
        print(f"\n{task.upper()}:")
        print(f"{'Method':<20} {'vs Black-Box':<15} {'p-value':<15} {'Significant':<15}")
        print("-" * 65)
        
        bb_task = final_results[final_results['distill_type'] == 'black_box'][col].values
        
        for method_name, method_key in [('Hidden State', 'hidden_state'), 
                                        ('Attention', 'attention'), 
                                        ('Combined', 'combined')]:
            method_task = final_results[final_results['distill_type'] == method_key][col].values
            t_stat, p_value = stats.ttest_rel(method_task, bb_task)
            
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            mean_diff = np.mean(method_task) - np.mean(bb_task)
            print(f"{method_name:<20} {mean_diff:+.4f}          {p_value:>8.4f}        {sig:<15}")
    
    print()
    print("=" * 80)
    
    return results, ci_results

if __name__ == "__main__":
    compute_statistical_significance(None)

