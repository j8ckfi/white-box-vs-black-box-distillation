import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("results_summary.csv")

# Filter for the last epoch (epoch 2) of each trial to get final performance
final_results = df[df['epoch'] == 2]

# Group by distillation type and calculate stats
stats = final_results.groupby('distill_type').agg({
    'validation_accuracy': ['mean', 'std', 'count'],
    'validation_accuracy_sst2': ['mean', 'std'],
    'validation_accuracy_mmlu': ['mean', 'std'],
    'validation_accuracy_gsm8k': ['mean', 'std']
}).round(4)

print("=== Final Validation Accuracy (Epoch 2) ===")
print(stats['validation_accuracy'])
print("\n=== Breakdown by Task ===")
print(stats[['validation_accuracy_sst2', 'validation_accuracy_mmlu', 'validation_accuracy_gsm8k']])


