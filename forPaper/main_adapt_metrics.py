import os
from datetime import datetime
from forPaper.adapt_metrics import (store_formality_per_model_forPaper,
                                    calc_allcombo_f1_acc_forPaper_ALL)
from src.utils.helper_functions import (OUTPUT_DIR_STATS, _check_dir_exist)


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
    
    # interate through all models (with each 5 system prompts)
    for i in range(0, len(DIRS), 5):
        # slicing model groups
        current_group = DIRS[i : i + 5]
        print(f"\n{current_group}")

        # create folder 
        print(f"\nCreating directory to save statistics for PAPER ...")
        report_dir = os.path.join(OUTPUT_DIR_STATS, datetime.now().strftime("%b%d"), current_group[0]+"_paper")
        _check_dir_exist(report_dir)
        
        # 1. get and format all formality info
        combined_basic_data = store_formality_per_model_forPaper(dirs=current_group)
        # 2. calculate F1, Acc, Conv(t) for different combinations
        calc_allcombo_f1_acc_forPaper_ALL(per_model_dict=combined_basic_data, output_path=report_dir)
