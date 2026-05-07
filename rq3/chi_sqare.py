# CHi-square test 
# distributions are correlated (meaning, are they essentially the same distribution or did the prompt change the behavior?)
# Chi-Square Test of Homogeneity.

# NOTE: testing for understanding (7.5.26)
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Flattening the heatmaps based on your screenshot
# Order: [Male-M, Male-F, Male-N, Female-M, Female-F, Female-N, Neutral-M, Neutral-F, Neutral-N]
data = {
    "--": [28, 197, 675, 0, 286, 614, 1, 102, 797],
    "female": [25, 248, 627, 2, 365, 533, 3, 168, 729],
    "male": [29, 230, 641, 4, 362, 534, 7, 131, 762],
    "Nanako(f)": [34, 188, 678, 6, 274, 620, 2, 120, 778],
    "Rintaro(m)": [35, 195, 670, 5, 287, 608, 1, 115, 784]
}

# Convert to a 5x9 Matrix
obs_matrix = np.array(list(data.values()))
prompts = list(data.keys())

# Run the Chi-Square Test of Homogeneity
chi2, p, dof, expected = chi2_contingency(obs_matrix)

print(f"Global Chi-Square: {chi2:.2f}")
print(f"P-Value: {p:.4e}")

# Detailed Similarity: Pairwise comparison
pairwise_results = []
for i in range(len(prompts)):
    for j in range(i + 1, len(prompts)):
        pair_obs = np.array([obs_matrix[i], obs_matrix[j]])
        _, p_pair, _, _ = chi2_contingency(pair_obs)
        pairwise_results.append((prompts[i], prompts[j], round(p_pair, 4)))

df_pairs = pd.DataFrame(pairwise_results, columns=["Prompt A", "Prompt B", "P-Value"])
print("\nPairwise Similarity (p > 0.05 means they are effectively identical):")
print(df_pairs)