import numpy as np
from scipy.stats import chi2_contingency, ttest_ind_from_stats

# 5 topics x 180 QA x 3 runs = 2700 samples
N = 2700 

def check_accuracy_significance(acc_baseline, acc_prompt, label):
    # Convert percentages to raw counts
    correct_base = round((acc_baseline / 100) * N)
    incorrect_base = N - correct_base
    
    correct_prompt = round((acc_prompt / 100) * N)
    incorrect_prompt = N - correct_prompt
    
    # Create the contingency table
    table = [
        [correct_base, incorrect_base],
        [correct_prompt, incorrect_prompt]
    ]
    
    # Run Chi-Squared test
    chi2, p_value, dof, expected = chi2_contingency(table)
    
    print(f"--- Baseline vs {label} ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: SIGNIFICANT improvement/change! (p < 0.05)")
    else:
        print("Result: Not statistically significant. (p >= 0.05)")
    print()

# Example using Qwen3-1.7B data from your image
# [jp1.6B, jp7B, Qwen1.7B, Qwen4B]
# baselines = [35.89, 39.33, 35.26, 39.56]
# labels = ["Female", "Male", "Nanako", "Rintaro"]

all_data = {"jp1_6B": {"Baseline": 35.89,
                        "Female": 38.22, 
                        "Male": 38.48, 
                        "Nanako": 37.93, 
                        "Rintaro": 37.52},
       
       "jp7B": { "Baseline": 39.33,
                 "Female": 40.30, 
                 "Male": 38.78, 
                 "Nanako": 38.00, 
                 "Rintaro": 38.41},
       
       "qwen1_7B": {"Baseline": 35.26,
                    "Female":39.89, 
                    "Male": 39.11, 
                    "Nanako": 42.00, 
                    "Rintaro": 41.33},
       
        "qwen4B": { "Baseline": 39.56,
                    "Female":39.67, 
                    "Male": 39.93, 
                    "Nanako": 39.15, 
                    "Rintaro": 40.85},
        }



for model, info in all_data.items():
    for label, acc in info.items():
        baseline = info["Baseline"]
        print(f"Model:{model} - Baseline: {baseline}")
        print(label, acc)
        if label != "Baseline": 
            check_accuracy_significance(acc_baseline=baseline, acc_prompt=acc, label=label)
    print(f"-"*40)