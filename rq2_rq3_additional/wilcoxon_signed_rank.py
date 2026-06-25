# for RQ-2: Conv(t) significant improvement or not from baseline vs gender prompts
import numpy as np
from scipy.stats import wilcoxon


def calc_wilcoxon_signed_rank(baseline_scores, prompt_scores, prompt_name="Gender Prompt"):
    """
    Performs a Wilcoxon signed-rank test between baseline and a gender prompt
    using averaged scores across 3 runs (900 pairs total).
    
    Parameters:
    - baseline_scores (list or np.array): 900 master scores for the baseline (-)
    - prompt_scores (list or np.array): 900 master scores for the target prompt
    - prompt_name (str): Name of the prompt for printing results cleanly
    """
    # arrays - baseline vs gender prompt
    baseline = np.array(baseline_scores)
    target = np.array(prompt_scores)

    # if len(baseline) != len(target):
    #     raise ValueError(f"Array length mismatch! Baseline has {len(baseline)} elements, but {prompt_name} has {len(target)}.")
        
    #  Wilcoxon signed-rank test
    # 'pratt' -> includes ranks where exact 0
    statistic, p_value = wilcoxon(baseline, target, zero_method='pratt')
    
    # Bonferroni-corrected alpha threshold (0.05 / 4 tests) 
    # to tighten probability rules and drastically lower the risk of reporting false positives.
    num_tests = 4
    alpha_corrected = 0.05 / num_tests
    
    # basically Conv(t)
    mean_baseline = np.mean(baseline)
    mean_target = np.mean(target)
    
    print(f"=== Statistical Analysis for: {prompt_name} ===")
    print(f"Baseline Conv(t): {mean_baseline:.4f}")
    print(f"{prompt_name} Conv(t): {mean_target:.4f}")
    print(f"Wilcoxon Statistic:   {statistic}")
    print(f"p-value:              {p_value:.6f}")
    print(f"Significance Threshold (Bonferroni): {alpha_corrected}")
    print("-" * 50)
    
    if p_value < alpha_corrected:
        if mean_target > mean_baseline:
            print(f"SIGNIFICANT & BETTER than baseline.")
        else:
            print(f"SIGNIFICANT & WORSE than baseline.")
    else:
        print(f"NOT statistically significant.")
    print("=" * 50 + "\n")
    
    # return statistic, p_value
