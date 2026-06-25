import os, sys
import pandas as pd
import matplotlib.pyplot as plt
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.helper_functions import OUTPUT_DIR_ADD_STATS


def plot_save_rq3_mean_std_perQ(partition_df, filename_end:str):
    """
        Takes in either all Q, non-control, or control Q and plot distribution
        with mean and std.
        Saves plot in /mount/studenten/projects/takamamo/add_stats_out
    """
    # get means and stds
    means = df[partition_df].mean()
    stds = df[partition_df].std()

    dynamic_width = max(5, len(partition_df) * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(dynamic_width, 7))

    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    # plot; color of mean (red), std (rose)
    ax.errorbar(
        partition_df, means, yerr=stds, fmt='o', 
        color="#b30303", ecolor="#ff9999", 
        capsize=4, elinewidth=1.5, zorder=3
    )

    # add Y-axis, and some space to prevent cut off for std
    ax.set_ylim(0.5, 6.5)
    ax.set_yticks([1, 2, 3, 4, 5, 6])

    # axis labels
    ax.set_xlabel("Questions", fontsize=12)
    ax.set_ylabel("Rating", fontsize=12)

    # layout adjustments
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # save and close
    out_path = os.path.join(OUTPUT_DIR_ADD_STATS, f"rq3_distribution_survey_res_{filename_end}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    


if __name__ == "__main__":
    
    file_path = "/mount/arbeitsdaten/studenten4/takamamo/rq3_results_gender_ratings.csv"
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # ALL Q, CONTROL Q, NON-CONTROL Q
    all_q = [
        col for col in df.columns 
        if col.lower().startswith('q') or col.lower().startswith('cq')
    ]

    only_control = [
        col for col in df.columns
        if col.lower().startswith('cq')
    ]
    
    only_non_c = [
        col for col in df.columns 
        if col.lower().startswith('q') and not col.lower().startswith('cq')
    ]

    
    plot_save_rq3_mean_std_perQ(partition_df=all_q, filename_end="all_q")
    plot_save_rq3_mean_std_perQ(partition_df=only_control, filename_end="control")
    plot_save_rq3_mean_std_perQ(partition_df=only_non_c, filename_end="non_control")