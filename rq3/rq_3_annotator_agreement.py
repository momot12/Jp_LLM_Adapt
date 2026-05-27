import pandas as pd
import krippendorff as kd



def calc_krippendorff_alpha(df_to_analyze, columns_partition, group_label="All"):
    """
    Extracts the matrix from the given dataframe partition and group,
    then computes Krippendorff's Alpha.
    """
    matrix = df_to_analyze[columns_partition].values

    print(f"\n--- Group: {group_label} ---")
    print(f"Participants: {matrix.shape[0]} | Questions: {matrix.shape[1]}")

    # calc alpha
    alpha = kd.alpha(reliability_data=matrix, level_of_measurement='ordinal')
    print(f"Krippendorff's Alpha (Ordinal): {alpha:.4f}")


if __name__ == "__main__":
    
    file_path = "/mount/arbeitsdaten/studenten4/takamamo/rq3_results_gender_ratings.csv"
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # ALL Q, ONLY CONTORL, NON-CONTROL Q
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

    # define gender dfs
    df_female = df[df['Gender'] == 'female']
    df_male = df[df['Gender'] == 'male']

    
    print("===========================================")
    print("      PARTITION 1: ALL QUESTIONS           ")
    print("===========================================")
    calc_krippendorff_alpha(df, all_q, group_label="All Participants")
    calc_krippendorff_alpha(df_female, all_q, group_label="Females Only")
    calc_krippendorff_alpha(df_male, all_q, group_label="Males Only")

    print("\n===========================================")
    print("      PARTITION 2: CONTROL QUESTIONS ONLY  ")
    print("===========================================")
    calc_krippendorff_alpha(df, only_control, group_label="All Participants")
    calc_krippendorff_alpha(df_female, only_control, group_label="Females Only")
    calc_krippendorff_alpha(df_male, only_control, group_label="Males Only")

    print("\n===========================================")
    print("      PARTITION 3: NON-CONTROL QUESTIONS   ")
    print("===========================================")
    calc_krippendorff_alpha(df, only_non_c, group_label="All Participants")
    calc_krippendorff_alpha(df_female, only_non_c, group_label="Females Only")
    calc_krippendorff_alpha(df_male, only_non_c, group_label="Males Only")