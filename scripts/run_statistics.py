import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import (single_file_stats, 
                                      plot_all_boxplots,
                                      save_stats_to_file,
                                      generate_heatmaps)
from src.stats.plots_heatmaps import bar_plots_stacked
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, combine_num_lists, _check_dir_exist)


def run_stats(sub_dir:str):
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be: 5 topics + single_run 
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))
    
    print(f"\n[INFO] Creating directory to save statistics ...")
    report_dir = os.path.join(OUTPUT_DIR_STATS, datetime.now().strftime("%b%d"), sub_dir)
    _check_dir_exist(report_dir)

    # 5x all three runs
    statlists_dict_three = {topic: {} for topic in topics}
    FILENAME = ""
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
                print(f"\n[FILE] {filepath}")
                # e.g. qwen_1_7B_career_1
                model_topic_run = str(sub_dir) + "_" + "_".join(file[:-5].split('_')[-2:])
                # FILENAME = f'{model_topic_run}_stats'
                
                # get dict with lists of basic & psycholing. stats
                statlists_dict = single_file_stats(filepath=filepath)
      
                # save 3 files each topic
                statlists_dict_each_run[int(model_topic_run[-1])] = statlists_dict

    ###-------------------------------------------------------------------###
            # 3xRuns for topic -> {topic: {1: {[],[]}, 2: {[],[]}, 3: {[],[]}}}
            statlists_dict_three[dir] = statlists_dict_each_run
    ###-------------------------------------------------------------------###
    
    save_stats_to_file(data=statlists_dict_three, model_name=sub_dir, filepath=report_dir)
    
    
    print(f"#############################END####\n")
    



# ------------------------------------- #

DIRS = ['jp_stable_1_6B',
        'jp_stable_7B',
        'qwen_1_7B',
        'qwen_4B']

for sub_dir in DIRS[:1]:
    print(f"[RUNNING] {sub_dir}")
    run_stats(sub_dir=sub_dir)