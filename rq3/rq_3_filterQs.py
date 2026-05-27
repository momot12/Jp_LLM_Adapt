import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import get_formlvl_gender, _get_sent_first_bleu_filter
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, _check_dir_exist)



def filter_samples_rq3(sub_dir:str, threshold_bleu:float):
    
    print(f"\n[RUNNING] {sub_dir}")
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be: 5 topics + single_run 
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))
    
    print(f"\nCreating directory to save RQ-3 samples ...")
    report_dir = os.path.join(OUTPUT_DIR_STATS, f"{datetime.now().strftime('%b%d')}_{sub_dir}")
    _check_dir_exist(report_dir)

    # 5x all three runs
    pertopic_samples = {topic: [] for topic in topics}
    pertopic_special = {topic: [] for topic in topics}

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

                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                    for item in data:
                        for q_id, info in item.items():
                            # add BLEU threshold to info (dict with Q, A etc.)
                            info["threshold"] = threshold_bleu

                            # Filter 1: female input (endings, pronouns)
                            if info["gender"] == "female":
                                # determine formality & gender of output
                                # {"input": [formality, gender], "output": [formality, gender]}
                                lvl_gend = get_formlvl_gender(info=info)
                                sentences, first_sent = _get_sent_first_bleu_filter(info=info, threshold=threshold_bleu)
                                
                                # extra filter
                                if info["only_out"].startswith(('ふん', 'え', 'は')):
                                    special = {"question": info["question"],
                                              "only_out": info["only_out"],
                                              "first_sent": first_sent
                                             }
                                    accepted_special = special | lvl_gend
                                    # save
                                    pertopic_special[dir].append(accepted_special)
                                
                                
                                # Filter 2: output formality <= input formality 
                                input_formality = int(lvl_gend["input"][0])
                                output_formality = int(lvl_gend["output"][0])
                                if output_formality == input_formality:
                                    sample = {"question": info["question"],
                                              "only_out": info["only_out"],
                                              "first_sent": first_sent
                                             }
                                    # {question:x, only_out:x, first_sent:x, input: [formality, gender], output: [formality, gender]}
                                    accepted_sample = sample | lvl_gend
                                    
                                    # save
                                    pertopic_samples[dir].append(accepted_sample)
                                    

    ###-------------------------------------------------------------------###
    for topic, samples in pertopic_samples.items():
        # Construct the file path
        outpath = os.path.join(report_dir, f"RQ_3_{topic}_{sub_dir}.txt")
        
        # Open with explicit UTF-8 encoding
        with open(outpath, "w", encoding='utf-8') as f:
            # write count
            f.write(f"#samples for {topic}: {len(samples)}\n\n")
            for sample in samples:
                # Ensure every access uses the singular 'sample' variable
                in_out_line = f"Input: {sample['input'][0]}, {sample['input'][1]}\nOutput: {sample['output'][0]}, {sample['output'][1]}\n"
                q_a_line = f"Q: {sample['question']}\nA: {sample['only_out']}\n"
                optional_first_sent_line = f"A (filtered): {sample['first_sent']}\n"
                
                line = in_out_line + q_a_line + optional_first_sent_line
                f.write(line + "\n\n")

    # specials 
    for topic, samples in pertopic_special.items():
        # Construct the file path
        outpath_special = os.path.join(report_dir, f"RQ_3_{topic}_{sub_dir}_special.txt")
        
        # Open with explicit UTF-8 encoding
        with open(outpath_special, "w", encoding='utf-8') as f:
            # write count
            f.write(f"#samples for {topic}: {len(samples)}\n\n")
            for sample in samples:
                # Ensure every access uses the singular 'sample' variable
                in_out_line = f"Input: {sample['input'][0]}, {sample['input'][1]}\nOutput: {sample['output'][0]}, {sample['output'][1]}\n"
                q_a_line = f"Q: {sample['question']}\nA: {sample['only_out']}\n"
                optional_first_sent_line = f"A (filtered): {sample['first_sent']}\n"
                
                line = in_out_line + q_a_line + optional_first_sent_line
                f.write(line + "\n\n")
                 
            
    ###-------------------------------------------------------------------###
    

filter_samples_rq3(sub_dir="jp_stable_7B", threshold_bleu=0.8)