import sys
import os
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import get_single_file_in_out_formality
from src.utils.helper_functions import OUTPUT_DIR
from forPaper.metric_calcs import (calc_f1_acc_formality_filtered_forPaper,
                          calc_conv_t_formality_filtered_forPaper)


def get_in_out_form_forPaper(sub_dir:str, threshold_bleu:float):
    """
        Go through each file with the model genderated answer, and determine 
        formality based on the first sentence determined by the BLEU threshold.
        Returns a dictionary with the information structured as:
        {topic: {run 1: {q_id: [formality_in, formality_out]},
                 run 2:{},
                 run 3: {}
                }
        }
    """
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
        sp_gender_raw_ok = copy.deepcopy(sp_gender_raw)
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
        
    return filtered_output, sp_gender_raw_ok


# ------------------------------------- #
def combine_filtered_outputs(filtered_dicts_list: list):
    """
        Combines multiple filtered dictionaries (e.g., neutral, female, nanako) 
        into a single unified dictionary structure, and returns a dict.
    """
    combined_output = {}
    
    for f_dict in filtered_dicts_list:
        for model_key, topics_dict in f_dict.items():
            # If the model isn't in our combined dict yet, create its entry
            if model_key not in combined_output:
                combined_output[model_key] = {}
                
            for topic_id, runs_dict in topics_dict.items():
                if topic_id not in combined_output[model_key]:
                    combined_output[model_key][topic_id] = {}
                    
                for run_id, q_dict in runs_dict.items():
                    if run_id not in combined_output[model_key][topic_id]:
                        combined_output[model_key][topic_id][run_id] = {}
                    
                    # Merge question items for this run
                    for q_key, val in q_dict.items():
                        # To avoid key collisions if question keys repeat across different system prompt runs,
                        # we can append a suffix or store them directly if q_keys are unique.
                        # Usually, adding them directly works since q_keys have unique IDs.
                        unique_q_key = f"{q_key}_{model_key}"
                        combined_output[model_key][topic_id][run_id][unique_q_key] = copy.deepcopy(val)
                        
    return combined_output


# ------------------------------------- #
def calc_allcombo_f1_acc_forPaper(per_model_dict: dict, output_path: str):
    """
        Runs the complete evaluation pipeline for all individual and combined 
        system prompt vs. question gender combinations, saving results to text files.
    """
    
    # --- NEUTRAL---
    # N-N
    filtered_nn, sp_raw_nn = filter_genders_forPaper(per_model_dict=per_model_dict, 
                                                     system_prompt_gender="neutral", 
                                                     question_gender="neutral")

    calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_nn, 
                                            sp_gender_raw=sp_raw_nn, 
                                            question_gender="neutral", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=filtered_nn, 
                                            sp_gender_raw=sp_raw_nn, 
                                            question_gender="neutral", 
                                            output_path=output_path)
    
    # --- FEMALE ---
    # N-F
    filtered_nf, sp_raw_nf = filter_genders_forPaper(per_model_dict=per_model_dict, 
                                                     system_prompt_gender="neutral", 
                                                     question_gender="female")

    calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_nf, 
                                            sp_gender_raw=sp_raw_nf, 
                                            question_gender="female", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=filtered_nf, 
                                            sp_gender_raw=sp_raw_nf, 
                                            question_gender="female", 
                                            output_path=output_path)
    
    # F-F
    # together: F+Nanako - F
    filtered_ff, _ = filter_genders_forPaper(per_model_dict=per_model_dict, 
                                             system_prompt_gender="female", 
                                             question_gender="female")
    # F-F 
    filtered_ff_strict = {k: v for k, v in filtered_ff.items() if k.endswith('_female')}
    calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_ff_strict, 
                                            sp_gender_raw="female", 
                                            question_gender="female", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=filtered_ff_strict, 
                                            sp_gender_raw="female", 
                                            question_gender="female", 
                                            output_path=output_path)
    
    # Nanako-F
    filtered_nanako = {k: v for k, v in filtered_ff.items() if k.endswith('_nanako')}
    calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_nanako, 
                                            sp_gender_raw="nanako", 
                                            question_gender="female", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=filtered_nanako, 
                                            sp_gender_raw="nanako", 
                                            question_gender="female", 
                                            output_path=output_path)
    
    # N+F+Nanako - F
    combined_female = combine_filtered_outputs([filtered_nf, filtered_ff])
    calc_f1_acc_formality_filtered_forPaper(filtered_output=combined_female, 
                                            sp_gender_raw="neutral_female_nanako", 
                                            question_gender="female", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=combined_female, 
                                            sp_gender_raw="neutral_female_nanako", 
                                            question_gender="female", 
                                            output_path=output_path)

    
    # --- MALE ---
    # N-M
    filtered_nm, sp_raw_nm = filter_genders_forPaper(per_model_dict=per_model_dict, 
                                                     system_prompt_gender="neutral", 
                                                     question_gender="male")

    calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_nm, 
                                            sp_gender_raw=sp_raw_nm, 
                                            question_gender="male", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=filtered_nm, 
                                            sp_gender_raw=sp_raw_nm, 
                                            question_gender="male", 
                                            output_path=output_path)

    # together: M+Rintaro - M 
    filtered_mm, _ = filter_genders_forPaper(per_model_dict=per_model_dict, 
                                             system_prompt_gender="male", 
                                             question_gender="male")
    
    # M-M
    filtered_mm_strict = {k: v for k, v in filtered_mm.items() if k.endswith('_male')}
    calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_mm_strict, 
                                            sp_gender_raw="male", 
                                            question_gender="male", 
                                            output_path=output_path)

    calc_conv_t_formality_filtered_forPaper(filtered_output=filtered_mm_strict, 
                                            sp_gender_raw="male", 
                                            question_gender="male", 
                                            output_path=output_path)

    # Rintaro-M
    filtered_rintaro = {k: v for k, v in filtered_mm.items() if k.endswith('_rintaro')}
    calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_rintaro, 
                                            sp_gender_raw="rintaro", 
                                            question_gender="male", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=filtered_rintaro, 
                                            sp_gender_raw="rintaro", 
                                            question_gender="male", 
                                            output_path=output_path)

    # N+M+Rintaro - M
    combined_male = combine_filtered_outputs([filtered_nm, filtered_mm])
    calc_f1_acc_formality_filtered_forPaper(filtered_output=combined_male, 
                                            sp_gender_raw="neutral_male_rintaro", 
                                            question_gender="male", 
                                            output_path=output_path)
    
    calc_conv_t_formality_filtered_forPaper(filtered_output=combined_male, 
                                            sp_gender_raw="neutral_male_rintaro", 
                                            question_gender="male", 
                                            output_path=output_path)

    print("\nAll paper relevant combo calculations (F1, Acc, Conv(t)) done and saved.")   