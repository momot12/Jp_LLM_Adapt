import sys
import os
import copy
import numpy as np
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import get_single_file_in_out_formality
from src.stats.f1_acc import get_macro_class_f1_acc
from src.stats.conv_t import calc_conv_t
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, _check_dir_exist)


def get_in_out_form_forPaper(sub_dir:str, threshold_bleu:float):
    
    print(f"\n[RUNNING] {sub_dir}")
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be: 5 topics + single_run 
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))

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
                # get dict with in-out formality level only
                statlists_dict = get_single_file_in_out_formality(filepath=filepath, threshold_bleu=threshold_bleu)
                # save 3 files each topic
                statlists_dict_each_run[int(model_topic_run[-1])] = statlists_dict

    ###-------------------------------------------------------------------###
            # 3xRuns for topic -> {topic: {1: {[]}, 2: {[]}, 3: {[]}}}
            statlists_dict_three[dir] = statlists_dict_each_run
    ###-------------------------------------------------------------------###
    
    return statlists_dict_three
    
    
    # print(f"\nCalculating F1 & accuracy...")
    # calc_f1_acc_formality(data=statlists_dict_three, model_name=sub_dir, output_path=report_dir)
    # # print(f"\nCalculating Conv(t)...")
    # # calc_conv_t(data=statlists_dict_three, model_name=sub_dir, output_path=report_dir)
    
    # print(f"\n\nAll statistics completed.")
    # print(f"############################# END ####\n")
    
# ------------------------------------- #
def store_formality_per_model_forPaper(dirs):
    """
        Collects all in-out formality for the same model over all 5 system prompts.
        Returns a dictionary with in-out formality info.
    """    
    PER_MODEL_dict = {}
    
    for sub_dir in dirs:
        # get in-out formality per topic per 3 runs for a model with specific system prompt
        # e.g. JP-1.6B with neutral system prompt
        stats_perSP = get_in_out_form_forPaper(sub_dir=sub_dir, threshold_bleu=0.8)
        
        # save {jp_stable_1_6B: {topic: {1: {[]}, 2: {[]}, 3: {[]}}}, jp_stable_1_6B_male: {topic: {1: {[]}, 2: {[]}, 3: {[]}}},...}
        if sub_dir not in PER_MODEL_dict.keys():
            PER_MODEL_dict[sub_dir] = stats_perSP
        else: print("WARNING: Duplicate key.")
        
    return PER_MODEL_dict

# ------------------------------------- #
def filter_genders_forPaper(per_model_dict:dict, system_prompt_gender:str, question_gender:str):
    """
        Filter relevant samples by system prompt type and the question gender, 
        e.g. neutral system prompt - female-style questions
    """
    gender_map = {"female": "female",
                   "nanako": "female",
                   "male": "male", 
                   "rintaro": "male",
                   "neutral": "neutral"
                   }
    
    filtered_output = {}
    
    for model_key, topics_dict in per_model_dict.items():
        # 1. determine system prompt gender from the key suffix
        sp_parts = model_key.split('_')
        # original prompt: keeping female and nanako separate
        sp_gender_raw = sp_parts[-1] if sp_parts[-1] in gender_map.keys() else "neutral"
        
        # map all to female, male, or if no suffix, then neutral
        sp_mapped = gender_map.get(sp_gender_raw, "neutral")
        
        # System Prompt: check if wanted gender and is_gender are matching
        if sp_mapped != system_prompt_gender:
            continue
        # Question Gender:
        filtered_topics = {}
        for topic_id, run_ids in topics_dict.items():
            filtered_runs = {}
            for run_id, inner_dict in run_ids.items():
                # 2. Filter inner dictionary keys by question_gender
                filtered_inner = {}
                for q_key, val in inner_dict.items():
                    # e.g. 'q25_career_2_female' -> ['q25', 'career', '2', 'female']
                    q_parts = q_key.split('_')
                    q_gender_raw = q_parts[-1]
                    q_mapped = gender_map.get(q_gender_raw)
                    
                    # 3. Collect only matching pairs (question gender)
                    if q_mapped == question_gender:
                        filtered_inner[q_key] = copy.deepcopy(val)
                        
                # keep matchin on run_id-level
                filtered_runs[run_id] = filtered_inner
            
            # keep matching on q_pairs-level
            filtered_topics[topic_id] = filtered_runs
                
        # keep matching on topic-level
        filtered_output[model_key] = filtered_topics
        
    return filtered_output, sp_gender_raw

# ------------------------------------- #
# TODO: add a function or sth that combines the three 
        

# ------------------------------------- #
def calc_f1_acc_formality_filtered_forPaper(filtered_output:dict, sp_gender_raw:str, output_path:str):
    """
        Calculates F1 scores and accuracy per run (1, 2, 3) and total (mean ± std) 
        for filtered data, and writes out the formatted text file.
    """
    # ([true - in_form], [pred - out_form])
    per_runs_data = {1: ([], []), 2: ([], []), 3: ([], [])}
    total_samples_count = 0
    
    # collect data groups
    for model_key, topics_dict in filtered_output.items():
        # {topic: {run 1: {q_id: [in, out]}}}
        for topic_id, runs_dict in topics_dict.items():
            #{run 1: {q_id: [in, out]}
            for run_id, q_dict in runs_dict.items():
                # {q_id: [in, out]}
                for q_key, val in q_dict.items():
                    in_form, out_form = val[0], val[1]
                    per_runs_data[run_id][0].append(in_form)
                    per_runs_data[run_id][1].append(out_form)
                    total_samples_count += 1
    
    # num sample per run
    samples_per_run = len(per_runs_data[1][0]) if 1 in per_runs_data else 0
    
    # calculate metrics for each run
    run_metrics = {}
    for run_id in [1, 2, 3]:
        true_list, pred_list = per_runs_data[run_id]
        if len(true_list) > 0:
            run_metrics[run_id] = get_macro_class_f1_acc(true=true_list, pred=pred_list)

    # calculate Total (Mean ± Std) across the 3 runs for each metric key
    total_metrics = {}
    if run_metrics[1]:
        # dict_keys(['1_f1', '2_f1', '3_f1', 'macro_f1', 'accuracy'])
        metric_keys = run_metrics[1].keys()
        for key in metric_keys:
            scores = [run_metrics[run][key] for run in [1, 2, 3] if key in run_metrics[run]]
            if scores:
                mean_val = float(np.mean(scores))
                std_val = float(np.std(scores))
                total_metrics[key] = f"{mean_val:.4f} ± {std_val:.4f}"

    # Format the output text content
    # NOTE: remove later
    question_gender = "neutral"
    # output_path = "/mount/studenten/projects/takamamo/stats_out/Aug14/jp_stable_1_6B/_paper"
    ####
    
    output_lines = []
    output_lines.append(f"#Samples = {samples_per_run} {question_gender} x 3 runs = {total_samples_count}")
    
    for run_id in [1, 2, 3]:
        output_lines.append(f"Run {run_id}:")
        if run_metrics[run_id]:
            for metric_key, val in run_metrics[run_id].items():
                # Format metric names nicely (e.g., macro_f1 -> Macro_f1)
                formatted_name = metric_key.replace('_f1', '_f1').capitalize()
                if 'f1' in metric_key and 'macro' not in metric_key:
                    formatted_name = metric_key # keeps things like 1_f1
                elif 'macro' in metric_key:
                    formatted_name = "Macro_f1"
                elif 'accuracy' in metric_key:
                    formatted_name = "Accuracy"
                
                output_lines.append(f"\t\t{formatted_name}: {val:.4f}")

    output_lines.append("	Total:")
    for metric_key, formatted_val in total_metrics.items():
        formatted_name = metric_key.replace('_f1', '_f1')
        if 'macro' in metric_key:
            formatted_name = "Macro_f1"
        elif 'accuracy' in metric_key:
            formatted_name = "Accuracy"
        
        output_lines.append(f"\t\t{formatted_name}: {formatted_val}")

    # Write out to file
    filename = f"f1_acc_{sp_gender_raw}_{question_gender}.txt"
    out_file = os.path.join(output_path, filename)
    
    os.makedirs(output_path, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines) + "\n")
        
    print(f"Successfully saved metrics report to: {out_file}")


    # ------------------------------------- #
def kfj_forPaper():
    # one model
    # 3 dicts: filtered neutral, female, male
    # 3 out files 
    pass


# ------------------------------------- #
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
        
        combined_basic_data = store_formality_per_model_forPaper(dirs=current_group)
        filtered_data = filter_genders_forPaper(per_model_dict=combined_basic_data,
                                                system_prompt_gender="neutral",
                                                question_gender="neutral")
        calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_data)

    
        
        
        
        
        
        # print(f"\nCreating directory to save statistics for PAPER ...")
        # report_dir = os.path.join(OUTPUT_DIR_STATS, datetime.now().strftime("%b%d"), sub_dir+"_paper")
        # _check_dir_exist(report_dir)

