import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt


# Assuming input is: [{in: [formality, gender], out: [formality, gender]}, {}...]
# example_out = [
#                 {"input": ["1", "M"], "output": ["2", "M"]},
#                 {"input": ["1", "F"], "output": ["3", "F"]},
#                 {"input": ["2", "M"], "output": ["1", "M"]},
#                 {"input": ["1", "M"], "output": ["2", "M"]},
#                 {"input": ["2", "M"], "output": ["2", "M"]},
#               ]

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
            # [FILTER]Input: if formality 3 and male --> then neutral
            f"{item['input'][0]}-n" if str(item['input'][0]) == "3" and item['input'][1][0].lower() == "m" 
            else f"{item['input'][0]}-{item['input'][1][0].lower()}",
            # Output
            f"{item['output'][0]}-{item['output'][1][0].lower()}"
        ) 
        for item in info_list]

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

def plot_heatmap(matrix, max_val:int=None, filename:str="testplot"):
    """
        Takes in a matrix created earlier, and creates a heatmap.
    """
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


def matrix_plot(info_list:list, max_val:int=None, filename:str="testplot", mode:int=1):
    """
        Depending on the mode, creates respective matrix and plots heatmap.
        Mode 1: Formality adaptation between input and output.
        Mode 2: Formality vs gender in output.
        Mode 3: Formality & gender adaptation between input and output.
    """
    
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
    


# 1. Replace these with your actual lists of numbers
run1_counts = [120, 150, 130, 170, 140]  # Example data
run2_counts = [110, 160, 125, 180, 135]
run3_counts = [130, 145, 140, 165, 150]

def bar_plots_stacked(run1:list, run2:list, run3:list, filename:str="stacked_plot"):
    # 2. Calculate Averages
    avg1 = np.mean(run1)
    avg2 = np.mean(run2)
    avg3 = np.mean(run3)

    # Calculate the "Combined" average (average of all responses across all runs)
    all_data = run1 + run2 + run3
    total_sum = avg1 + avg2 + avg3

    # 3. Data for Plotting
    plt.figure(figsize=(6, 8))
    
    blues = ['#aed6f1', '#3498db', '#21618c']

    # plot each run stacked
    p1 = plt.bar(0, avg1, color=blues[0], label=f'Run 1 (Avg: {avg1:.1f})')
    p2 = plt.bar(0, avg2, bottom=avg1, color=blues[1], label=f'Run 2 (Avg: {avg2:.1f})')
    p3 = plt.bar(0, avg3, bottom=avg1+avg2, color=blues[2], label=f'Run 3 (Avg: {avg3:.1f})')

    # plot 1x
    p4 = plt.bar(1, avg1, color=blues[0], edgecolor='black', alpha=0.9, width=0.6)
    
    plt.xticks([0, 1], ['3xRun', '1xRun'])
    # plt.xlabel('3xRuns', fontsize=12)

    # label for the total height
    plt.text(0, total_sum+5, f'{total_sum:.1f}', ha='center', fontweight='bold')
    plt.text(1, avg1+5, f'{avg1:.1f}', ha='center', fontweight='bold')

    # labels each part
    plt.text(0, avg1/2, f'{avg1:.1f}', ha='center', color='black', fontweight='bold')
    plt.text(0, avg1 + (avg2/2), f'{avg2:.1f}', ha='center', color='black', fontweight='bold')
    plt.text(0, avg1 + avg2 + (avg3/2), f'{avg3:.1f}', ha='center', color='black', fontweight='bold')
    
    plt.text(1, avg1/2, f'{avg1:.1f}', ha='center', color='black', fontweight='bold')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # 5. Save and Show
    plt.savefig(f"{filename}.png")
    plt.close()
        
bar_plots_stacked(run1=run1_counts, run2=run2_counts, run3=run3_counts)