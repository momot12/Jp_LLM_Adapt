import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_bleu_histograms(info_dict, report_dir:str):
    """
    Generates and saves histograms from a nested BLEU score dictionary.
    Structure: {topic: {run: {'all_scores': [...]}}}
    """

    # data prep
    rows = []
    for topic, runs in info_dict.items():
        for run_id, run_data in runs.items():
            # go through 180 scores
            for score in run_data['all_scores']:
                # store as a flat record (easier for Seaborn to process)
                rows.append({'Topic': topic, 'Run': run_id, 'BLEU_Score': score})
    
    # format into pandas df
    df = pd.DataFrame(rows)
    topics = df['Topic'].unique()     # get unique topic names (e.g. life)
    runs = sorted(df['Run'].unique()) # get sorted run IDs (e.g. 1)

    # set a clean visual style for all plots
    sns.set_theme(style="whitegrid")

    # --- INDIVIDUAL RUNS  ---
    # rows = topics, columns = runs
    fig1, axes1 = plt.subplots(len(topics), len(runs), 
                               figsize=(4 * len(runs), 3.5 * len(topics)), 
                               sharex=True) # all plots share the 0.0-1.0 X-axis
    fig1.suptitle("Individual Run Distributions", fontsize=16, y=1.02)
    
    for i, topic in enumerate(topics):
        for j, run in enumerate(runs):
            # select the correct subplot index
            ax = axes1[i, j] if len(topics) > 1 else axes1[j]
            # filter data for this specific topic and specific run
            subset = df[(df['Topic'] == topic) & (df['Run'] == run)]
            # Kernel Density Estimate (KDE) line -> kde=True
            sns.histplot(subset['BLEU_Score'], bins=20, kde=False, ax=ax, color="blue")
            ax.set_title(f"{topic} | {run}") # Label each small plot
            ax.set_xlim(0, 1) # Force X-axis to standard BLEU range

    # save
    plt.tight_layout()
    fig1.savefig(f"{report_dir}/1_individual_runs.png") 
    plt.close() 

    # --- TOPIC TOTALS (combined runs) ---
    # horizontal row of plots (one per topic)
    fig2, axes2 = plt.subplots(1, len(topics), figsize=(5 * len(topics), 5), sharey=True)
    fig2.suptitle("Topic Totals: Run Comparison", fontsize=16, y=1.05)
    
    for i, topic in enumerate(topics):
        # indexing if there is only 1 topic vs multiple
        ax = axes2[i] if len(topics) > 1 else axes2
        subset = df[df['Topic'] == topic]
        # overlay the 3 runs using 'hue' to compare them in the same space
        sns.histplot(data=subset, x='BLEU_Score', hue='Run', bins=25, 
                     kde=True, element="step", ax=ax)
        ax.set_title(f"Total: {topic}")
        ax.set_xlim(0, 1)

    # save
    plt.tight_layout()
    fig2.savefig(f"{report_dir}/2_topic_totals.png")
    plt.close()

    # --- OVERALL RUN TOTALS (Combined Topics per Run) ---
    # horizontal row (one plot for Run 1, one for Run 2, etc.)
    fig3, axes3 = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5), sharey=True)
    fig3.suptitle("Global Run Performance (All Topics Combined)", fontsize=16, y=1.05)
    
    for i, run in enumerate(runs):
        ax = axes3[i]
        # filter all topics belonging to this run ID
        subset = df[df['Run'] == run]
        # plot
        sns.histplot(subset['BLEU_Score'], bins=30, kde=True, ax=ax, color="orange")
        ax.set_title(f"Overall {run}")
        ax.set_xlim(0, 1)

    # save
    plt.tight_layout()
    fig3.savefig(f"{report_dir}/3_global_run_totals.png") 
    plt.close()

    # --- GRAND TOTAL ---
    # single plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['BLEU_Score'], bins=40, kde=True, color="purple")
    plt.title("Overall Total: All Topics & All Runs Combined", fontsize=14)
    plt.xlabel("BLEU Score")
    plt.xlim(0, 1)
    
    # save
    plt.tight_layout()
    plt.savefig(f"{report_dir}/4_grand_total.png") 
    # plt.show() 