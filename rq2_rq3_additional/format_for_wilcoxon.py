import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import get_single_file_in_out_formality
from src.utils.helper_functions import OUTPUT_DIR


def get_formalities_perfile(sub_dir:str, threshold_bleu:float):
    """
        Goes through each file of generated answers of each models (sub_dir), and 
        stores input and output formality-gender information in a dictionary to further process.
    """
    print(f"\n[RUNNING] {sub_dir}")
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be: 5 topics + single_run 
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))

    # 5x all three runs
    q_form_dict_three = {topic: {} for topic in topics}

    # topic directories
    for dir in subdirs:
        if dir in topics:
            # ~/output_dir/model_name/topic
            OUTPUT_DIR_FULL =  os.path.join(OUTPUT_DIR_subdirs, dir)
            # should be: 3 files for each run (1, 2, 3) with '.._[run].json'
            files = sorted(os.listdir(path=OUTPUT_DIR_FULL))
            
            # 3 files each topic
            q_formality_dict_each_run = {1: {}, 2: {}, 3: {}}
    ###-------------------------------------------------------------------###
            # each run json file
            # print("="*40)
            print(f"*** TOPIC: {dir.upper()} ***")
            for file in files:
                # ~/output_dir/model_name/topic/file_[run].json
                filepath = os.path.join(OUTPUT_DIR_FULL, file)
                # print(f"[FILE] {filepath}")
                # e.g. qwen_1_7B_career_1
                model_topic_run = str(sub_dir) + "_" + "_".join(file[:-5].split('_')[-2:])
                # get dict with lists of basic & psycholing. stats
                q_formality_dict = get_single_file_in_out_formality(filepath=filepath, threshold_bleu=threshold_bleu)
                # save 3 files each topic
                q_formality_dict_each_run[int(model_topic_run[-1])] = q_formality_dict

    ###-------------------------------------------------------------------###
            # 3xRuns for topic -> {topic: {1: {[],[]}, 2: {[],[]}, 3: {[],[]}}}
            q_form_dict_three[dir] = q_formality_dict_each_run
    
    return q_form_dict_three


def get_900_points_per_t(data: dict):
    """
    Extracts a dictionary of 900 averaged data points for each formality level (t=1, 2, 3).
    Each point represents a unique Question-Topic combination averaged across 3 runs.
    
    Args:
        data (dict): {topic1: {run1: {q_id: [formality_in, formality_out]},
                               run2: {},
                               run3: {}},
                      topic2: {},
                     }
    """
    # output dict; t=1, 2, 3 (formality levels)
    # list order with each 180 points: career, daily, knowledge, life, simple
    output_900_dict = {1: [], 2: [], 3: []}

    # for each formality level 
    for t in [1, 2, 3]:
        
        # loop per topic
        for topic, runs in data.items():

            # calc all per run baselines per topic 
            run_baselines = {}
            for run_id in [1, 2, 3]:
                run_dict = runs.get(run_id, {})
                
                # how often t in B across all examples
                # AB pairs if B has t
                t_in_B = [q_data for q_data in run_dict.values() if int(q_data[1]) == t]
                # P(b_t_a = 1)
                run_baselines[run_id] = len(t_in_B)/len(run_dict)
            
            # ----
            # calculate for each q and each run, 
            # and average for the specific q & specific t
            sample_run = list(runs.keys())[0] # 1
            all_q_ids = list(runs[sample_run].keys()) # ['q1', q2, ...]
            
            # Loop through each of the 180 questions for this topic
            for q_id in all_q_ids:
                q_scores_across_runs = []
                
                # specific question across all 3 runs
                for run_id_q in [1, 2, 3]:
                    run_dict = runs.get(run_id_q, {})
                    q_data = run_dict.get(q_id)
                        
                    in_formality = int(q_data[0])
                    out_formality = int(q_data[1])
                    baseline_rate = run_baselines[run_id_q]
                    
                    # look at only input with specific t
                    if in_formality == t:
                        # 1.0 if same t as out_formality, else 0
                        trigger_rate = 1.0 if out_formality == t else 0.0
                    # input with different t is also 0
                    else:
                        trigger_rate = 0.0
                        
                    # score = trigger_rate - baseline
                    run_score = trigger_rate - baseline_rate
                    q_scores_across_runs.append(run_score)
                
                # average of all runs for specific question
                avg_q_score = sum(q_scores_across_runs) / len(q_scores_across_runs)
                    
                # save avg score per q in overall for t
                output_900_dict[t].append(avg_q_score)
    
    return output_900_dict
                


def stack_2700_points(points_dict:dict):
    """
        Takes in a dictionary with {t=1: [900 points], t=2: [900 points], t=3:[900 points]}, where t denotes the formality levels,
        and stacks them into one array (900x3=2700)
    """
    # retrieve individual level points
    lvl1 = points_dict.get(1, [])
    lvl2 = points_dict.get(2, [])
    lvl3 = points_dict.get(3, [])
    
    # stack sequentially 
    stacked_2700_array = lvl1 + lvl2 + lvl3
    
    return stacked_2700_array
