# CHi-square test 
# distributions are correlated (meaning, are they essentially the same distribution or did the prompt change the behavior?)
# Chi-Square Test of Homogeneity.

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def calc_chi2(data:dict):
    # Order: [Male-M, Male-F, Male-N, Female-M, Female-F, Female-N, Neutral-M, Neutral-F, Neutral-N]
    # data = {
    #     "--": [28, 197, 675, 0, 286, 614, 1, 102, 797],
    #     "female": [25, 248, 627, 2, 365, 533, 3, 168, 729],
    #     "male": [29, 230, 641, 4, 362, 534, 7, 131, 762],
    #     "Nanako(f)": [34, 188, 678, 6, 274, 620, 2, 120, 778],
    #     "Rintaro(m)": [35, 195, 670, 5, 287, 608, 1, 115, 784]
    # }
    lines = []
    lines.append(f"Order: [Male-M, Male-F, Male-N, Female-M, Female-F, Female-N, Neutral-M, Neutral-F, Neutral-N]\n")
    print(f"Order: [Male-M, Male-F, Male-N, Female-M, Female-F, Female-N, Neutral-M, Neutral-F, Neutral-N]")

    # Convert to a 5x9 Matrix
    obs_matrix = np.array(list(data.values()))
    prompts = list(data.keys())

    # global Chi2 of Homogeneity
    chi2, p, dof, expected = chi2_contingency(obs_matrix)
    
    lines.append(f"Global Chi-Square: {chi2:.2f} | P-Value: {p:.4e}")
    lines.append(f"dof: {dof}")
    print(f"Global Chi-Square: {chi2:.2f} | P-Value: {p:.4e} | df: {dof}")

    # pairwise comparison
    pairwise_results = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            pair_obs = np.array([obs_matrix[i], obs_matrix[j]])
            chi2_individ, p_pair_raw, dof_individ, expected_individ = chi2_contingency(pair_obs)
            
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
    lines = calc_chi2(data=data)
    # save
    with open(report_dir, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n[INFO] Chi2 saved in {report_dir}")