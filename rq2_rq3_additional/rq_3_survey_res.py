"""
This file analyzes survey results by calculating avg, min, max on different subsets.
    - All questions vs control questions vs non-control questions (of the survey)
    - All participants vs only female vs only male
"""
import pandas as pd


def check_csv_format(df):
    """
        Sanity check if the csv file is formatted correctly.
        It is customized to the specifc csv file:
        ==> /mount/arbeitsdaten/studenten4/takamamo/rq3_results_gender_ratings.csv
    """
    
    # number of headers (columns)
    total_columns = len(df.columns) 
    print(f"#columns: {total_columns}")

    total_rows = len(df)
    print(f"#rows: {total_rows}")

    if (total_columns == 26) and (total_rows == 33):
        print("All correctly formatted!")
    elif (total_columns != 26) and (total_rows != 33):
        print(f"#columns should be 26, not {total_columns}.")
        print(f"#rows should be 33, not {total_rows}.")
    elif (total_columns == 26) and (total_rows != 33):
        print(f"#columns should be 26, and are {total_columns}.")
        print(f"#rows should be 33, not {total_rows}.")
    elif (total_columns != 26) and (total_rows == 33):
        print(f"#columns should be 26, not {total_columns}.")
        print(f"#rows should be 33, and are {total_rows}.")

    print("=" * 50)
    # print(df.count())


def get_avg_min_max_wFilter(df, gender: str = "", topic: str = ""): 
    """
        For survey results analysis:
        Get avg, min, max on different subsets.
            - All questions vs control questions vs non-control questions (of the survey)
            - All participants vs only female vs only male
    """
    
    topic_mapping = {
        "career": career,
        "daily": daily,
        "knowledge": knowledge,
        "life": life,
        "simple": simple
    }
    
    gender_filer = False
    topic_filter = False
    
    print("-" * 51)
    if gender != "":
        df = df[df['Gender'].astype(str).str.strip().str.lower() == gender.strip().lower()]
        print(f"Filtered {gender}: N={len(df)}")
        gender_filer = True
    else: print(f"No gender filter: N={len(df)}")
        
    topic_clean = topic.strip().lower()
    if topic_clean in topic_mapping:
        active_columns = topic_mapping[topic_clean]
        print(f"Filtered by {topic}")
        topic_filter = True
    else:
        active_columns = df.columns
        print("No topic filter.")

    # ALL Q, CONTROL Q, NON-CONTROL Q
    all_q = [
        col for col in active_columns 
        if col.lower().startswith('q') or col.lower().startswith('cq')
    ]

    only_control = [
        col for col in active_columns 
        if col.lower().startswith('cq')
    ]
    
    only_non_c = [
        col for col in active_columns 
        if col.lower().startswith('q') and not col.lower().startswith('cq')
    ]

    # AVG - individual
    avg_all = df[all_q].mean(numeric_only=True)
    avg_control = df[only_control].mean(numeric_only=True)
    avg_non_c = df[only_non_c].mean(numeric_only=True)
    
    if not gender_filer and not topic_filter:
        print(avg_all)
    
    # === OVERALL AVERAGES ===
    overall_avg = avg_all.mean()
    overall_control = avg_control.mean()
    overall_non_c = avg_non_c.mean() 

    # === OVERALL MIN VALUES ===
    min_all = avg_all.min()
    min_control = avg_control.min()
    min_non_c = avg_non_c.min() 
    # === OVERALL MAX VALUES ===
    max_all = avg_all.max()
    max_control = avg_control.max()
    max_non_c = avg_non_c.max()

    # === PRINT THE MATRIX ===
    print(f"{'':<15}{'Avg':>12}{'Max':>12}{'Min':>12}")
    
    print(f"{'Overall:':<15}"
          f"{overall_avg:>12.2f}"
          f"{max_all:>12.2f}"
          f"{min_all:>12.2f}")
          
    print(f"{'Only control:':<15}"
          f"{overall_control:>12.2f}"
          f"{max_control:>12.2f}"
          f"{min_control:>12.2f}")
          
    print(f"{'Non-control:':<15}"
          f"{overall_non_c:>12.2f}"
          f"{max_non_c:>12.2f}"
          f"{min_non_c:>12.2f}")
    print("-" * 51)



if __name__ == "__main__":
    
    file_path = "/mount/arbeitsdaten/studenten4/takamamo/rq3_results_gender_ratings.csv"
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    career = ['q3', 'cq4', 'q11', 'q17', 'q22']
    daily = ['q1', 'cq5', 'q8', 'q14', 'q20']
    knowledge = ['q6', 'q12', 'cq15', 'q18', 'q25']
    life = ['q7', 'q13', 'q19', 'cq23', 'q24']
    simple = ['q2', 'q9', 'cq10', 'q16', 'q21']

    genders = ["female", "male"]
    topics = ["career", "daily", "knowledge", "life", "simple"]
    
    # #(optional) check if file is formatted correctly
    check_csv_format(df=df)


    ## -- COMMENT OUT (#) based on which subset -- ##
    ## (1) no filter 
    get_avg_min_max_wFilter(df=df) 

    ## (2) gender filer
    # for gender in genders:
    #     get_avg_min_max_wFilter(df=df, gender=gender)
        

    ## (3) topic filter 
    # for topic in topics:
    #     get_avg_min_max_wFilter(df=df, topic=topic)
        

    ## (4) gender + topic filter 
    # for gender in genders:
    #     for topic in topics:
    #         get_avg_min_max_wFilter(df=df, gender=gender, topic=topic)
        