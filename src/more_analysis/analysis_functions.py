import os 
import json
import statistics
from src.utils.helper_functions import get_avg_min_max # OUTPUT_DIR_ADD_STATS, _check_dir_exist
from src.more_analysis.bleu_score import write_bleu_sent
# from src.stats.stat_functions import _get_sent_first_bleu_filter


def get_bleu_stats(filepath:str, report_dir: str, model_topic_run:str, threshold:float):
    """
        Takes in a file with QA pairs, formats them for calculating BLEU score for each pair.
        Writes a file with the QA pair and the score.
        Returns a dictionary with:
            - per_q: {q_id: score}
            - all_scores: [bleu, bleu, ...]
            - avg: float
            - min: float
            - max: float
            - std: float
            - bleu_counts: {}
    """
    
    with open(filepath, 'r') as f:
        data = json.load(f)

        # collect all pairs of one file
        pairs = []
        for item in data:
            qa = list(item.values())[0]
            # e.g. 'q1_career_3_neutral'
            q_id = list(item.keys())[0]
            pairs.append({"q_id": q_id, "question": qa["question"], "only_out": qa["only_out"]})
            
        # BLUE score on first sentence
        # dict {per_q: {q_id: score}, all_scores: [bleu, bleu], avg: x, min: x, max: x, std: x}
        bleu_stats = write_bleu_sent(data=pairs, 
                                     threshold=threshold,
                                     model_topic_run=model_topic_run,
                                     output_path=report_dir,
                                     mode=2) # only first sentence

        return bleu_stats
    
    
def get_overall_scores(info_dict:dict, sub_dir:str, report_dir: str):
    """
        Calculates BLEU scores based on runs, topics, and overall.
        Writes it in a file.
    """
    
    lines = [f"OVERALL (bleu): {sub_dir}\n\n"]
    
    # for overall
    global_all_scores = [] 
    global_per_run = {1: [], 2: [], 3: []}
    
    for topic, run_stats in info_dict.items():
        topic_3x_scores = []
        topic_run_stats = {"avgs": [], "mins": [], "maxs": [], "stds": []}
        
        # get all scores of each run 
        for run_id in [1, 2, 3]:
            current_run = run_stats.get(run_id, {})
            
            # individual BLEU scores
            run_scores = current_run.get("all_scores", [])
            topic_3x_scores.extend(run_scores)
    
            # per run stats 
            topic_run_stats["avgs"].append(current_run.get("avg"))
            topic_run_stats["mins"].append(current_run.get("min"))
            topic_run_stats["maxs"].append(current_run.get("max"))
            topic_run_stats["stds"].append(current_run.get("std"))
            
            global_all_scores.extend(run_scores)
            global_per_run[run_id].extend(run_scores)
            
        # average of all three run stats
        run_avg_mean = statistics.mean(topic_run_stats["avgs"])
        run_min_mean = statistics.mean(topic_run_stats["mins"])
        run_max_mean = statistics.mean(topic_run_stats["maxs"])
        run_std_mean = statistics.mean(topic_run_stats["stds"])   
        # stats over all BLEU scores
        avg_3x_scores, min_3x_scores, max_3x_scores, std_3x_scores, = get_avg_min_max(numbers_list=topic_3x_scores)
    
        line = (
            f"BLEU overall ({topic}):\n"
            f"\t\tall_scores, run_stats\n"
            f"\t Avg: {avg_3x_scores:.4f}, {run_avg_mean:.4f}\n"
            f"\t Min: {min_3x_scores:.4f}, {run_min_mean:.4f}\n"
            f"\t Max: {max_3x_scores:.4f}, {run_max_mean:.4f}\n"
            f"\t Std: {std_3x_scores:.4f}, {run_std_mean:.4f}\n\n"
        )  
        lines.append(line)
    
    ### OVERALL ### 
    # NOTE: details not fully done, but not necessary
    lines.append(f"OVERALL (all topics):\n")
    
    # overall per run
    for run_id in [1, 2, 3]:
        r_avg, r_min, r_max, r_std = get_avg_min_max(global_per_run[run_id])
        line = (
            f"\tRun {run_id}:\n"
            # f"\t\tall_scores, run_stats\n"
            f"\t\tAvg: {r_avg:.4f}\n" #, {run_avg_mean:.4f}\n"
            f"\t\tMin: {r_min:.4f}\n" #, {run_min_mean:.4f}\n"
            f"\t\tMax: {r_max:.4f}\n" #, {run_max_mean:.4f}\n"
            f"\t\tStd: {r_std:.4f}\n" #, {run_std_mean:.4f}\n\n"
        ) 
        lines.append(line)
        
    # over all topics, all runs
    g_avg, g_min, g_max, g_std = get_avg_min_max(global_all_scores)
    lines.append(f"\nOVEALL TOTAL (all topics, all runs):\n")
    lines.append(f"\tAvg: {g_avg:.4f}\n\tMin: {g_min:.4f}\n\tMax: {g_max:.4f}\n\tStd: {g_std:.4f}\n")
     
    # write results
    with open(os.path.join(report_dir, f"{sub_dir}_bleu.txt"), "w") as f:
        for i in lines:
            f.write(i)


def get_overview_bleu_filter_counts(info_dict:dict, sub_dir:str, report_dir: str):
    """
        Writes a file with an overview of counts of #QA pairs that have BLEU score above a threshold, and 
        exact 1.0 (=repetition of Q in A).
        Counts are: 
            - per topic, per run
            - per topic, over all runs
            - overall, per run
            - overall, over all runs
    """  
    
    lines = [f"OVERALL (bleu counts): {sub_dir}\n"]
    lines.append(f"Number order is above_threshold, exact_repetition\n\n")

    # for overall
    global_per_run = {1: {'above_threshold': 0, 'exact_repetition': 0}, 
                      2: {'above_threshold': 0, 'exact_repetition': 0},
                      3: {'above_threshold': 0, 'exact_repetition': 0}
                     }
    
    for topic, run_stats in info_dict.items():
        lines.append(f"{topic.capitalize()}:\n")    
        topic_3x_scores = {'above_threshold': 0, 'exact_repetition': 0}
        
        # get all scores of each run 
        for run_id in [1, 2, 3]:
            current_run_bleu = run_stats.get(run_id, {}).get("bleu_counts", [])
            
            above = current_run_bleu.get("above_threshold", 0)
            exact = current_run_bleu.get("exact_repetition", 0)
            
            # over all 3 runs
            topic_3x_scores['above_threshold'] += above
            topic_3x_scores['exact_repetition'] += exact
            
            # count overall per run 
            global_per_run[run_id]['above_threshold'] += above
            global_per_run[run_id]['exact_repetition'] += exact
        
            lines.append(f"\tRun {run_id}: {above}, {exact}\n")
        
        # total per topic
        above_total = topic_3x_scores.get("above_threshold")
        exact_total = topic_3x_scores.get("exact_repetition")
        lines.append(f"\tTotal: {above_total}, {exact_total}\n\n")
    
    # --- overall --- #
    # overall per run
    lines.append(f"\nOverall:\n")
    
    above_overall, exact_overall = 0, 0
    
    for run_id in [1, 2, 3]:
        above_perRun = global_per_run[run_id].get("above_threshold")
        exact_perRun = global_per_run[run_id].get("exact_repetition")
        lines.append(f"\tRun {run_id}: {above_perRun}, {exact_perRun}\n")
        
        # add for total overall
        above_overall += above_perRun
        exact_overall += exact_perRun
    
    lines.append(f"\tTotal: {above_overall}, {exact_overall}")
    
    # write results
    with open(os.path.join(report_dir, f"{sub_dir}_bleu_counts.txt"), "w") as f:
        for i in lines:
            f.write(i)
    
    