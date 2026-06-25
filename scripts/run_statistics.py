import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import (single_file_stats, 
                                      save_stats_to_file,
                                      generate_heatmaps,
                                      generate_barplots)
from src.stats.f1_acc import calc_f1_acc
from src.stats.conv_t import calc_conv_t
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, _check_dir_exist)


def run_stats(sub_dir:str, threshold_bleu:float):
    
    print(f"\n[RUNNING] {sub_dir}")
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be: 5 topics + single_run 
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))
    
    print(f"\nCreating directory to save statistics ...")
    report_dir = os.path.join(OUTPUT_DIR_STATS, datetime.now().strftime("%b%d"), sub_dir)
    _check_dir_exist(report_dir)

    # 5x all three runs
    statlists_dict_three = {topic: {} for topic in topics}

    # topic directories
    for dir in subdirs:
        if dir in topics:
            # ~/output_dir/model_name/topic
            OUTPUT_DIR_FULL =  os.path.join(OUTPUT_DIR_subdirs, dir)
            # should be: 3 files for each run (1, 2, 3) with '.._[run].json'
            files = sorted(os.listdir(path=OUTPUT_DIR_FULL))
            
            # 3 files each topic
            statlists_dict_each_run = {1: {}, 2: {}, 3: {}}
    ###-------------------------------------------------------------------###
            # each run json file
            print("="*40)
            print(f"*** TOPIC: {dir.upper()} ***")
            for file in files:
                # ~/output_dir/model_name/topic/file_[run].json
                filepath = os.path.join(OUTPUT_DIR_FULL, file)
                print(f"[FILE] {file}")
                # e.g. qwen_1_7B_career_1
                model_topic_run = str(sub_dir) + "_" + "_".join(file[:-5].split('_')[-2:])
                # get dict with lists of basic & psycholing. stats
                statlists_dict = single_file_stats(filepath=filepath, threshold_bleu=threshold_bleu)
                # save 3 files each topic
                statlists_dict_each_run[int(model_topic_run[-1])] = statlists_dict

    ###-------------------------------------------------------------------###
            # 3xRuns for topic -> {topic: {1: {[],[]}, 2: {[],[]}, 3: {[],[]}}}
            statlists_dict_three[dir] = statlists_dict_each_run
    ###-------------------------------------------------------------------###
    
    print(f"\n\nSaving stats in txt file...")
    save_stats_to_file(data=statlists_dict_three, model_name=sub_dir, output_path=report_dir)
    print(f"\nGenerating heatmaps...")
    generate_heatmaps(data=statlists_dict_three, model_name=sub_dir, output_path=report_dir)
    print(f"\nGenerating barplots...")
    generate_barplots(data=statlists_dict_three, model_name=sub_dir, output_path=report_dir)
    
    print(f"\nCalculating F1 & accuracy...")
    calc_f1_acc(data=statlists_dict_three, model_name=sub_dir, output_path=report_dir)
    print(f"\nCalculating Conv(t)...")
    calc_conv_t(data=statlists_dict_three, model_name=sub_dir, output_path=report_dir)
    
    print(f"\n\nAll statistics completed.")
    print(f"############################# END ####\n")
    
# ------------------------------------- #



# DIRS = [
#     "jp_stable_1_6B",
#     "jp_stable_1_6B_female",
#     "jp_stable_1_6B_male",
#     "jp_stable_1_6B_nanako",
#     "jp_stable_1_6B_rintaro",
#     "jp_stable_7B",
#     "jp_stable_7B_female",
#     "jp_stable_7B_male",
#     "jp_stable_7B_nanako",
#     "jp_stable_7B_rintaro",
#     "qwen_1_7B",
#     "qwen_1_7B_female",
#     "qwen_1_7B_male",
#     "qwen_1_7B_nanako",
#     "qwen_1_7B_rintaro",
#     "qwen_4B",
#     "qwen_4B_female",
#     "qwen_4B_male",
#     "qwen_4B_nanako",
#     "qwen_4B_rintaro"
# ]

# for sub_dir in DIRS:
#     run_stats(sub_dir=sub_dir, threshold_bleu=0.8)