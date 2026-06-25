"""
This file calculateds Chi2 of homogenity to see the distribution shifts between heatmaps.
Specifically, it examines if the distributions of all prompt types (neutral and gendered)
are shifting based on the prompt type. It shows if the distributions are different or similar.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def calc_chi2(data:dict):
    """
        Calculates Chi2 of homogenity and returns results in a list of strings,
        in order to write them in files in get_chi2(data:dict, report_dir:str).
    """
    
    lines = []

    # Convert to a 5x9 Matrix
    obs_matrix = np.array(list(data.values()))
    prompts = list(data.keys())

    # global Chi2 of Homogeneity
    # drop columns with only 0 if both heatmaps have 0
    global_valid_cols = np.sum(obs_matrix, axis=0) > 0
    clean_global_matrix = obs_matrix[:, global_valid_cols]
    
    chi2, p, dof, expected = chi2_contingency(clean_global_matrix)
    # chi2, p, dof, expected = chi2_contingency(obs_matrix)
    
    lines.append(f"Global Chi-Square: {chi2:.2f} | P-Value: {p:.4e}")
    lines.append(f"dof: {dof}")
    print(f"Global Chi-Square: {chi2:.2f} | P-Value: {p:.4e} | df: {dof}")

    # pairwise comparison
    pairwise_results = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            pair_obs = np.array([obs_matrix[i], obs_matrix[j]])
            
            pair_valid_cols = np.sum(pair_obs, axis=0) > 0
            clean_pair_obs = pair_obs[:, pair_valid_cols]
            
            chi2_individ, p_pair_raw, dof_individ, expected_individ = chi2_contingency(clean_pair_obs)
            
            p_pair = round(p_pair_raw, 4)
            words = ""
            if p_pair < 0.001: words = "different***" 
            elif p_pair < 0.01: words = "different**"       
            elif p_pair < 0.05: words = "different*"         
            else: words = "similar"         
            pairwise_results.append((prompts[i], prompts[j], p_pair, words, round(chi2_individ, 2), round(dof_individ, 4), expected_individ))

    df_pairs = pd.DataFrame(pairwise_results, columns=["Prompt A", "Prompt B", "P-Value", "Word", "Chi-2", "df", "E"])
    table_header = f"\nPairwise Similarity (p > 0.05 means they are effectively identical):\n"
    lines.append(table_header)
    lines.append(df_pairs.to_string(index=False))
    print(table_header)
    print(df_pairs.to_string(index=False))
    
    return lines
    

def get_chi2(data:dict, report_dir:str):
    """
        Calculates Chi2 of homogenity on data and writes it in a file.
    """
    
    lines = calc_chi2(data=data)
    # save
    with open(report_dir, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n[INFO] Chi2 saved in {report_dir}")