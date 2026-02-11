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
LABELS_gender = ["male", "female", "neutral"]
# ['1-m', '1-f', '1-n', '2-m'...]
COMBINED_LABELS = [f"{f}-{g[0].lower()}" for f in LABELS_formality for g in LABELS_gender]
# LABELS_gender = ["M", "F", "N"]



def count_formGend(info_list: list):
    # get input formality & gender from input and output
    # ex.: '1-m', '2-f' etc.
    data = [
        (
            # Input: if formality 3 and male --> then neutral
            f"{item['input'][0]}-n" if str(item['input'][0]) == "3" and item['input'][1][0].lower() == "m" 
            else f"{item['input'][0]}-{item['input'][1][0].lower()}",
            # Output
            f"{item['output'][0]}-{item['output'][1][0].lower()}"
        ) 
        for item in info_list
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["User", "Model"])

    # create the matrix
    # formality-gender (model vs. user)
    count_matrix = (
        df
        .groupby(["Model", "User"])
        .size()
        .unstack(fill_value=0)
        # Ensure all formality levels and genders are present even if count is 0
        .reindex(index=COMBINED_LABELS, columns=COMBINED_LABELS, fill_value=0)
    )

    return count_matrix


def count_formality_gender(info_list: list):
    # Extract only the output data: (Formality, Gender)
    data = [(int(item["output"][0]), item["output"][1]) for item in info_list]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Formality", "Gender"])
    
    # Create the matrix
    # Index = Formality (Rows), Columns = Gender
    count_matrix = (
        df
        .groupby(["Formality", "Gender"])
        .size()
        .unstack(fill_value=0)
        # Ensure all formality levels and genders are present even if count is 0
        .reindex(index=LABELS_formality, columns=LABELS_gender, fill_value=0)
    )

    return count_matrix

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

def plot_heatmap(matrix, max_val:int,filename:str="testplot"):
    # (Optional) set fig size
    # plt.figure(figsize=(6,5)) 
    
    # plot heatmap
    map = sb.heatmap(
        matrix,               # df
        annot=True,           # show counts
        fmt="d",              # integer
        cmap="Reds",          # color palette
        vmax=max_val,
        cbar_kws={"label": "Count"}  # colorbar label
    )
    
    # axis labels
    plt.xlabel("User")       
    plt.ylabel("Model")
    
    # set title
    # map.set_title('Passenger Distribution Map', fontsize=16, pad=20)
    
    # save file 
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    # show plot
    # plt.show()


def matrix_plot(info_list:list, max_val:int, filename:str="testplot", mode:int=1):
    if mode == 1:
        matrix = count_formality(info_list=info_list)
    elif mode == 2:
        matrix = count_formality_gender(info_list=info_list)
    elif mode == 3:
        matrix = count_formGend(info_list=info_list)
    
    # plot
    plot_heatmap(matrix=matrix, max_val=max_val, filename=filename)
    plt.close()





# matrix_plot(info_list=example_out, max_val=10, mode=3)
