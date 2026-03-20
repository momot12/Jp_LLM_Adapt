import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt


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
            # Input: e.g. 1-m, 1-f, etc.
            f"{item['input'][0]}-{item['input'][1][0].lower()}",
            # Output: e.g. 1-m, 1-f, etc.
            f"{item['output'][0]}-{item['output'][1][0].lower()}"
        ) 
        for item in info_list]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["user", "model"])

    # create the matrix
    # formality-gender (model vs. user)
    count_matrix = (
        df
        .groupby(["user", "model"])
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
    """
        Gets count matrix for formalities.
        Args:
            info_list (list): [{"input": [formality, gender], "output": [formality, gender]}, {}]
    """
    
    # convert list of dict -> list of tuples for easier counting
    pairs = [(int(item["input"][0]), int(item["output"][0])) for item in info_list]

    # convert to pandas table
    df = pd.DataFrame(pairs, columns=["user", "model"])
    
    # group by input and output, count occurrences, and fill missing with 0
    count_matrix = (
        df
        .groupby(["user", "model"])     # group by input-output pair
        .size()                         # count number of occurrences
        .unstack(fill_value=0)          # group into table, fill 0 counts with 0
        .reindex(index=LABELS_formality,
                 columns=LABELS_formality, fill_value=0)  # ensure square matrix
    )
    
    return count_matrix

def plot_heatmap(matrix, label_dict:dict, max_val:int=None, filename:str="testplot"):
    """
        Takes in a matrix created earlier, and creates a heatmap.
        Args:
            matrix    : df.Dataframe matrix
            label_dict: labels for axis and title {x: .., y: ..., title: ...}
            max_val   : maximum count on the scale (to keep plots unified) TODO
            filename  : filename to save plot
    """
    # plot heatmap
    map = sb.heatmap(
            matrix,               # df
            annot=True,           # show counts
            fmt="d",              # integer
            cmap="Reds",          # color palette
            vmax=max_val,
            # cbar_kws={"label": "Count"}  # colorbar label
        )
    
    # axis labels & title
    if label_dict:
        plt.xlabel(label_dict.get('x', "Model"), fontsize=12)       
        plt.ylabel(label_dict.get('y', "User"), fontsize=12)
        plt.title(label_dict.get('title', ""), fontsize=12)

    # save file 
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    plt.close()


def matrix_plot(info_list:list, max_val:int=None, filename:str="testplot"): #, mode:int=1):
    """
        Depending on the mode, creates respective matrix and plots heatmap.
        Mode 1: Formality adaptation between input and output.
        Mode 2: Formality vs gender in output.
        Mode 3: Formality & gender adaptation between input and output.
    """
    plot_configs = [
        {
            "func": count_formality,
            "suffix": "f",
            "labels": {
                'title': "Formality Adaptation (Input vs Output)",
                'x': "Model (Formality)", 'y': "User (Formality)"
            }
        },
        {
            "func": count_formality_gender,
            "suffix": "f_g",
            "labels": {
                'title': "Output: Formality vs Gender",
                'x': "Gender", 'y': "Formality"
            }
        },
        {
            "func": count_formGend,
            "suffix": "FG",
            "labels": {
                'title': "Formality & Gender Adaptation (Input vs Output)",
                'x': "Model", 'y': "User"
            }
        }
    ]

    # 2. Loop through and generate each plot
    for config in plot_configs:
        # Call the specific counting function
        matrix = config["func"](info_list=info_list)
        
        # Create a unique filename for this mode (e.g., "run1_formality.png")
        # We strip .png if it's already there to avoid "run1.png_formality.png"
        base_name = filename.replace(".png", "")
        mode_filename = f"{base_name}_{config['suffix']}.png"

        #plot
        plot_heatmap(
            matrix=matrix, 
            label_dict=config["labels"], 
            max_val=max_val, 
            filename=mode_filename
        )

def plot_boxplot(data:list[int], plot_title:str, filename:str="distribution"):
    df = pd.DataFrame(data, columns=['Value'])
    
    mean_val = np.mean(data)
    max_val = np.max(data)
    min_val = np.min(data)
    std_val = np.std(data)

    # Creating the plot
    plt.figure(figsize=(8, 6))

    # Boxplot: showmeans=True adds a green circle for the mean
    plt.boxplot(data, vert=True, patch_artist=True, showmeans=True,
                boxprops=dict(facecolor='#DDEBF7', color='#2F5597'),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(marker='o', markerfacecolor='green', markeredgecolor='black'))

    # Add labels for the stats on the plot
    plt.text(1.1, max_val, f'Max: {max_val}', verticalalignment='center')
    plt.text(1.1, min_val, f'Min: {min_val}', verticalalignment='center')
    plt.text(1.1, mean_val, f'Mean: {mean_val:.2f}', verticalalignment='center', color='green')

    plt.title(plot_title)
    # plt.ylabel('Values')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save and show
    plt.savefig(f"{filename}.png")
    plt.close()
    


# Example data
run1_counts = [120, 150, 130, 170, 140]  
run2_counts = [110, 160, 125, 180, 135]
run3_counts = [130, 145, 140, 165, 150]

def bar_plots_stacked(run1:list, run2:list, run3:list, filename:str="stacked_plot"):
    # calculate averages
    avg1 = np.mean(run1)
    avg2 = np.mean(run2)
    avg3 = np.mean(run3)

    # Calculate the portions of the mean
    s1, s2, s3 = avg1/3, avg2/3, avg3/3
    mean_of_three = s1 + s2 + s3

    plt.figure(figsize=(6, 8))
    
    blues = ['#aed6f1', '#3498db', '#21618c']

    # plot each portion stacked so the total height equals the mean
    p1 = plt.bar(0, s1, color=blues[0], label=f'Run 1 Portion')
    p2 = plt.bar(0, s2, bottom=s1, color=blues[1], label=f'Run 2 Portion')
    p3 = plt.bar(0, s3, bottom=s1+s2, color=blues[2], label=f'Run 3 Portion')

    # plot 1x 
    p4 = plt.bar(1, avg1, color=blues[0], edgecolor='black', alpha=0.9, width=0.6)
    
    plt.xticks([0, 1], ['3xRun', '1xRun'])

    # labels for the total height
    plt.text(0, mean_of_three+3, f'{mean_of_three:.1f}', ha='center', fontweight='bold')
    plt.text(1, avg1+3, f'{avg1:.1f}', ha='center', fontweight='bold')

    # labels for each stacked portion
    plt.text(0, s1/2, f'{s1:.1f}', ha='center', color='black', fontsize=9)
    plt.text(0, s1 + (s2/2), f'{s2:.1f}', ha='center', color='black', fontsize=9)
    plt.text(0, s1 + s2 + (s3/2), f'{s3:.1f}', ha='center', color='black', fontsize=9)
    
    plt.text(1, avg1/2, f'{avg1:.1f}', ha='center', color='black', fontsize=9)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # save
    plt.savefig(f"{filename}.png")
    plt.close()
        
bar_plots_stacked(run1=run1_counts, run2=run2_counts, run3=run3_counts)