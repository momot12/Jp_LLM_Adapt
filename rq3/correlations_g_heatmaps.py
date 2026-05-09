import os, sys
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.helper_functions import OUTPUT_DIR, OUTPUT_DIR_ADD_STATS, _check_dir_exist
from rq3.chi_sqare import get_chi2
from src.stats.stat_functions import get_formlvl_gender
from src.stats.plots_heatmaps import count_gender



def get_chi2_matrix_permodel(sub_dir:str, threshold_bleu:float):
    
    print(f"\n[RUNNING] {sub_dir}")
    
    topics = ["career", "daily", "knowledge", "life", "simple"]
    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be topics + single_run
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))

    counts_all = []
    # topic directories
    for dir in subdirs:
        if dir in topics:
            # ~/output_dir/model_name/topic
            OUTPUT_DIR_FULL =  os.path.join(OUTPUT_DIR_subdirs, dir)
            # should be: 3 files for each run (1, 2, 3) with '.._[run].json'
            files = sorted(os.listdir(path=OUTPUT_DIR_FULL))

            # json files
            for file in files:
                # ~/output_dir/model_name/topic/file_[run].json
                filepath = os.path.join(OUTPUT_DIR_FULL, file)
                print(f"FILE: {file}")
                ##############################
                # per file
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    for item in data:
                        for q_id, info in item.items():
                            # add BLEU threshold to info (dict with Q, A etc.)
                            info["threshold"] = threshold_bleu
                
                            # {"input": [formality, gender], "output": [formality, gender]}
                            lvl_gend = get_formlvl_gender(info=info)
                            counts_all.append(lvl_gend)
    ### -------------------------------------- ###
    matrix = count_gender(info_list=counts_all)
    matrix = matrix.values.flatten().tolist()   
                
    return matrix

# ------------------------------------- #

def get_chi2_correlations(DIRS):
    """
        Calculate flattened matrices for gender match count between input and output.
        Calculate chi-square of homogenity of all flattened matrices of 5 prompts 
        and saves it in a file.
    """
    
    data = {
        "--": [],
        "female": [],
        "male": [],
        "Nanako(f)": [],
        "Rintaro(m)": []
    }
    
    for sub_dir in DIRS:
        matrix_flatten = get_chi2_matrix_permodel(sub_dir=sub_dir, threshold_bleu=0.8)
        
        if sub_dir.endswith("_female"):
            data["female"] = matrix_flatten
        elif sub_dir.endswith("_male"):
            data["male"] = matrix_flatten
        elif sub_dir.endswith("_nanako"):
            data["Nanako(f)"] = matrix_flatten
        elif sub_dir.endswith("_rintaro"):
            data["Rintaro(m)"] = matrix_flatten
        else:
            data["--"] = matrix_flatten
    
    matrix_check = False
    for matrix_list in data.values():
        if matrix_list != []:
            if len(matrix_list) == 9:
                matrix_check = True
            else:
                print(f"[INFO] Matrix list is incomplete: len != 9.")
                print(f"Values: {matrix_list}")
        else: print(f"[INFO] Matrix list is empty.")
    
    report_dir = os.path.join(OUTPUT_DIR_ADD_STATS, datetime.now().strftime("%b%d"))
    _check_dir_exist(dir_path=report_dir)
    if matrix_check:
        get_chi2(data=data, report_dir=os.path.join(report_dir, f"{'_'.join(sub_dir.split('_')[:-1])}_chi2.txt"))




if __name__ == "__main__":
    DIRS = [
        "jp_stable_1_6B",
        "jp_stable_1_6B_female",
        "jp_stable_1_6B_male",
        "jp_stable_1_6B_nanako",
        "jp_stable_1_6B_rintaro",
        "jp_stable_7B",
        "jp_stable_7B_female",
        "jp_stable_7B_male",
        "jp_stable_7B_nanako",
        "jp_stable_7B_rintaro",
        "qwen_1_7B",
        "qwen_1_7B_female",
        "qwen_1_7B_male",
        "qwen_1_7B_nanako",
        "qwen_1_7B_rintaro",
        "qwen_4B",
        "qwen_4B_female",
        "qwen_4B_male",
        "qwen_4B_nanako",
        "qwen_4B_rintaro"
    ]
    
    for i in range(0, len(DIRS), 5):
        # slicing model groups to only calculate chi2 among same models
        current_group = DIRS[i : i + 5]
        print(current_group)
        get_chi2_correlations(DIRS=current_group)