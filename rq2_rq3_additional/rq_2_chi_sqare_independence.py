"""
This file calculates Chi2 of independence of accuracy scores, comparing baseline prompt and 4 gendered prompts
to see if there is significant difference or not.
"""
from scipy.stats import chi2_contingency

# 5 topics x 180 QA x 3 runs = 2700 samples
N = 2700 

def check_accuracy_significance(acc_baseline, acc_prompt, label):
    """
        Calculates Chi2 of independence of accuracy scores, comparing baseline prompt and 4 gendered prompts.
    """
    # convert percentages to raw counts
    correct_base = round((acc_baseline / 100) * N)
    incorrect_base = N - correct_base
    
    correct_prompt = round((acc_prompt / 100) * N)
    incorrect_prompt = N - correct_prompt
    
    # contingency table
    table = [
        [correct_base, incorrect_base],
        [correct_prompt, incorrect_prompt]
    ]
    
    # Run Chi-Squared test
    chi2, p_value, dof, expected = chi2_contingency(table)
    
    print(f"--- Baseline vs {label} ---")
    print(f"Chi2: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("SIGNIFICANT improvement/change! (p < 0.05)")
    else:
        print("NOT statistically significant. (p >= 0.05)")
    print()



if __name__=="__main__":
    # run on accuracies (manually added)
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


    # run baseline prompt of each model vs gendered prompts
    for model, info in all_data.items():
        baseline = info["Baseline"]
        print(f"Model:{model}")
        for label, acc in info.items():
            print(f"{label}: {acc}")
            if label != "Baseline": 
                check_accuracy_significance(acc_baseline=baseline, acc_prompt=acc, label=label)
        print(f"-"*40)
        