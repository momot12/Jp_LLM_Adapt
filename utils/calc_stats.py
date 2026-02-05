import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# basic stats: len/#char, freq --> only formality, then all



# Assuming input is: [{in: [formality, gender], out: [formality, gender]}, {}...]
example_out = [
                {"input": ["1", "M"], "output": ["2", "M"]},
                {"input": ["1", "F"], "output": ["3", "F"]},
                {"input": ["2", "M"], "output": ["1", "M"]},
                {"input": ["1", "M"], "output": ["2", "M"]},
                {"input": ["2", "M"], "output": ["2", "M"]},
              ]

LABELS_formality = [1, 2, 3]


def count_formality(info_list:list):
    # convert list of dict -> list of tuples for easier counting
    pairs = [(int(item["input"][0]), int(item["output"][0])) for item in info_list]

    # convert to pandas table
    df = pd.DataFrame(pairs, columns=["user", "model"])
    
    # group by input and output, count occurrences, and fill missing with 0
    count_matrix = (
        df
        .groupby(["model", "user"])     # group by input-output pair
        .size()                         # count number of occurrences
        .unstack(fill_value=0)          # group into table, fill 0 counts with 0
        .reindex(index=LABELS_formality, columns=LABELS_formality, fill_value=0)  # ensure square matrix
    )

    return count_matrix

def plot_heatmap(matrix, filename:str="testplot"):
    # (Optional) set fig size
    # plt.figure(figsize=(6,5)) 
    
    # plot heatmap
    sb.heatmap(
        matrix,               # df
        annot=True,           # show counts
        fmt="d",              # integer
        cmap="Reds",         # color palette
        cbar_kws={"label": "Count"}  # colorbar label
    )
    
    # axis labels
    plt.xlabel("User")       
    plt.ylabel("Model")
    
    # save file 
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    # show plot
    # plt.show()


# matrix = count_formality(info_list=example_out)
# plot_heatmap(matrix=matrix)