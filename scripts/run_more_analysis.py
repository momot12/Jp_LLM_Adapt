import os, sys
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.helper_functions import OUTPUT_DIR, OUTPUT_DIR_ADD_STATS, _check_dir_exist
from src.more_analysis.analysis_functions import get_bleu_stats, get_overall_scores, get_overview_bleu_filter_counts
from src.more_analysis.plot_bleu import plot_bleu_histograms


def run_add_analysis(sub_dir:str, threshold_bleu:float):
    
    print(f"\n[RUNNING] {sub_dir}")
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be topics + single_run
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))
    
    print(f"\nCreating directory to save BLEU stats ...")
    # ~/add_stats_out/<MonthDay>/<model_name>
    report_dir = os.path.join(OUTPUT_DIR_ADD_STATS, datetime.now().strftime("%b%d"), sub_dir)
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

            # json files
            for file in files:
                # ~/output_dir/model_name/topic/file_[run].json
                filepath = os.path.join(OUTPUT_DIR_FULL, file)
                print(f"FILE: {filepath}")
                # e.g. qwen_1_7B_career_1
                model_topic_run = str(sub_dir) + "_" + "_".join(file[:-5].split('_')[-2:])
                ##############################
                bleu_stats_per_run = get_bleu_stats(filepath=filepath, 
                                                    report_dir=report_dir, 
                                                    model_topic_run=model_topic_run,
                                                    threshold=threshold_bleu
                                                    )
                
                # save 3 files each topic
                statlists_dict_each_run[int(model_topic_run[-1])] = bleu_stats_per_run
            
            ###-------------------------------------------------------------------###
            # 3xRuns for topic -> {topic: {1: {}, 2: {}, 3: {}}}
            statlists_dict_three[dir] = statlists_dict_each_run
            ###-------------------------------------------------------------------###
    
    # calculate + count BLEU 
    print(f"\n\nWriting BLEU scores ...")  
    get_overall_scores(info_dict=statlists_dict_three, sub_dir=sub_dir, report_dir=report_dir)
    print(f"\nCounting BLEU scores for filter ...")  
    get_overview_bleu_filter_counts(info_dict=statlists_dict_three, sub_dir=sub_dir, report_dir=report_dir)

    # plot graphs
    print(f"\nPlotting histograms ...")
    plot_bleu_histograms(info_dict=statlists_dict_three, report_dir=report_dir)
 
                    
    print(f"\n\nAll analysis completed.\nSaved in {report_dir}.")
    print(f"############################# END ####\n")    

# ------------------------------------- #

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

# for sub_dir in DIRS:
#     run_add_analysis(sub_dir=sub_dir, threshold_bleu=0.8)
run_add_analysis(sub_dir="qwen_1_7B_nanako", threshold_bleu=0.8)