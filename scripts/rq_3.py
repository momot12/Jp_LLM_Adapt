import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import get_formlvl_gender
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, _check_dir_exist)



def filter_samples_rq3(sub_dir:str, threshold_bleu:float):
    
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
    ###-------------------------------------------------------------------###
            # each run json file
            print("="*40)
            print(f"*** TOPIC: {dir.upper()} ***")
            for file in files:
                # ~/output_dir/model_name/topic/file_[run].json
                filepath = os.path.join(OUTPUT_DIR_FULL, file)
                print(f"[FILE] {filepath}")
                # e.g. qwen_1_7B_career_1
                model_topic_run = str(sub_dir) + "_" + "_".join(file[:-5].split('_')[-2:])
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                    for item in data:
                        for q_id, info in item.items():
                            # add BLEU threshold to info (dict with Q, A etc.)
                            info["threshold"] = threshold_bleu

                            print(info.keys())
                            # lvl_gend = get_formlvl_gender(info=info)

    ###-------------------------------------------------------------------###
            # 3xRuns for topic -> {topic: {1: {[],[]}, 2: {[],[]}, 3: {[],[]}}}
            # statlists_dict_three[dir] = statlists_dict_each_run
    ###-------------------------------------------------------------------###
    

filter_samples_rq3(sub_dir="jp_stable_1_6B", threshold_bleu=0.8)