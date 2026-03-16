import sys
import os
import json
import re
import numpy as np 
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from src.data_process.detect_f_g import DetectFG
from src.stats.psycholing_measures import PsychoLingMeasures
from src.stats.plots_heatmaps import matrix_plot, plot_boxplot
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, 
                                        _check_dir_exist,
                                        _get_sent_first_sent,
                                        get_avg_min_max)


# Init instance
NLP_MODEL = PsychoLingMeasures()

## --- HELPER --- ##
# def _get_sent_first_sent(output_clean:str):
#     """ Returns all sentences in a list and first sentence of output. It keeps punctuation if there is any.
#         Removes any \n or \s. 
#     """
#     output_clean = re.sub(r'\s+', '', output_clean).strip()
#     sentences = [s.strip() for s in re.split(r'(?<=[。！!？?])', output_clean) if s.strip()]
#     return sentences, sentences[0]


## --- DETERMINE FORMALITY & GENDER --- ##
def measure_formlvl_gender(out_text:str):
    """
        Takes in the clean output (only_out).
        Runs formality and gender detecting class and returns
            {"formality": detected_formality,
                "gender": detected_gender,}
    """
    engine = DetectFG()
    return engine.detect(text=out_text)
    
def get_formlvl_gender(info:dict):
    """
        Get formality levels and gender from input and output, and format.
        Return a dict {"input": [formality, gender], "output": [formality, gender]}.
    """
    formal_gend_out = measure_formlvl_gender(info["only_out"])
    out_dict = {"input":[info["formality"], info["gender"]],
                "output": [formal_gend_out["formality"], formal_gend_out["gender"]],
                }
    return out_dict


## --- BASIC STATS --- ##
def count_output_len(info:dict):
    """
        Counts #output sentences, and #chars of input vs output, including punctuation.
        Returns a dictionary with len(input), len(output), and #output sentences.
        Punctuation is included since some sentences don't include them.
    """
    len_info = {}
    
    input = info["question"]
    output_clean = info["only_out"]
    sentences, first_sent = _get_sent_first_sent(output_clean=output_clean)
    
    if len(output_clean) == 0:
        print(f"Clean output:\n{output_clean}")
        print(f"Original output:\n{info['output']}\n\n")
    
    # if len(sentences) == 8:
    #     print(f"{output_clean}\n{sentences}")
    
    len_info = {"input_len": len(input),
                "output_len": len(output_clean),
                "num_sent_out": len(sentences),
                "out_len_first": len(first_sent)
                }
    
    return len_info


## --- GENERAL FUNCTIONS (min, max, avg) --- ##
# def get_avg_min_max(numbers_list:list):
#     avg = round(sum(numbers_list)/len(numbers_list), 3)
#     val_min = min(numbers_list)
#     val_max = max(numbers_list)
#     val_std = round(np.std(numbers_list), 3)
#     return avg, val_min, val_max, val_std

def get_avg_out_performality(stat_info:dict):
    """
        Takes in the dictionary with all stats. 
        Calculated the average #char and #output sentences per formality.
        
        Returns avg for both stats, and respective counts per formality.
    """
    # totals: out_len, count: occurrence
    totals = {1: 0, 2: 0, 3: 0}
    counts = {1: 0, 2: 0, 3: 0}
    
    totals_sent = {1: 0, 2: 0, 3: 0}
    counts_sent = {1: 0, 2: 0, 3: 0}

    for item in stat_info:
        # Access the detected output formality level
        # Based on your data structure: item["output"] is [level, gender]
        level = int(item["output"][0])
        length = int(item["output_len"])
        num_sent = int(item["num_sent_out"])

        if level in totals:
            totals[level] += length
            counts[level] += 1
            
            totals_sent[level] += num_sent
            counts_sent[level] += 1

    avg_dict, avg_sent_dict = {}, {}
    # print("Avg #chars output per formality:")
    for level in [1, 2, 3]:
        # if 0 division, then keep 0
        avg = totals[level] / counts[level] if counts[level] > 0 else 0
        avg_dict[level] = round(avg, 2)
        # print(f"\tF{level}: {avg:.2f} (#samples: {counts[level]})")
    
    # print("Avg #sent output per formality:")
    for level in [1, 2, 3]:
        # if 0 division, then keep 0
        avg_sent = totals_sent[level] / counts_sent[level] if counts_sent[level] > 0 else 0
        avg_sent_dict[level] = round(avg_sent, 2)
        # print(f"\tF{level}: {avg_sent:.2f} (#samples: {counts_sent[level]})")
    
    return avg_dict, avg_sent_dict, counts, counts_sent

def get_avg_psych_ling_performality(stat_info:dict):
    """
        Takes in the dictionary with all stats. 
        Calculates the average psycholinguistic stat per formality.
        
        Stats: "AoA", "AoA_w_list", "syntax_depth", "lexical_complexity", "total_count_particle", "count_particle_skip"
        
        Returns
    """
    metrics = ["AoA", "AoA_w_list", "syntax_depth", "lexical_complexity", "total_count_particle", "count_particle_skip"]
    # totals == psycholing. nums
    totals = {metric: {1: 0, 2: 0, 3: 0} for metric in metrics}
    counts = {1: 0, 2: 0, 3: 0} # occurrences
    
    for item in stat_info:
        level = int(item["output"][0])
        if level in counts:
            counts[level]+=1
            # add values to each metric
            for metric in metrics:
                totals[metric][level] += float(item.get(metric, 0)) # if item missing, then add 0
    
    averages = {metric: {} for metric in metrics}
    averages["Samples"] = {level: counts[level] for level in [1, 2, 3]}
    
    for metric in metrics:
        # print(f"Avg {metric} per formality:")
        for level in [1, 2, 3]:
            # calc avg
            num = counts[level]
            avg = totals[metric][level] / num if num > 0 else 0.0
            # store in dict; e.g. {'AoA': {1: avg, 2: avg, 3: avg}, 'stat':{1:...}}    
            averages[metric][level] = round(avg, 3)
            # print
            # print(f"\tF{level}: {avg:.2f} (#samples: {num})")
        
    return averages
    

## --- PSYCHO.LING. STATS --- ##
def get_aoa_complexities(info:dict):
    """
        Get psychcolinguostic measures:
            - #Particles: more particles == more formal/advanced
                            - skippable vs. common
            - Lexical: Kango (complex, >2 Kanjis) vs. Wago (simple)
                = #Kango/#content_token
            - Syntactic: sub-clauses == depth tree
                => deeper == complex --> high number == complex
            - AoA (proxy): more Kanji usage, more advanced 
                            - proxy vs. list
                => more Kanjis, older person
                => min=5; max=15.0
    """
    output_clean = info["only_out"]
    _, first_sent = _get_sent_first_sent(output_clean=output_clean)
    stats = NLP_MODEL.get_psycholinguistic_metrics(sentence=first_sent)
    return stats


## --- SAVING, PLOTTING, PRINTING --- ##
def get_flat_samples(data, mode="single"):
    """
    Returns a flat list of 'full' samples for heatmaps or global stats.
    
    # Mode 'single': data = {'full': [{}, {}], 'len_in': []}
    # Mode 'three_runs': data = {'1': {'full': []}, '2': {...}, '3': {...}}
    # Mode 'global': data = {'career': {'1': {'full': []}, '2': ...}, 'daily': ...}
    """
    combined = []
    if mode == "single":
        return data.get("full", [])
    
    elif mode == "three_runs":
        for run_dict in data.values():
            combined.extend(run_dict.get("full", []))
            
    elif mode == "global":
        for topic_dict in data.values():
            for run_dict in topic_dict.values():
                combined.extend(run_dict.get("full", []))
    return combined

def get_flat_metric(data, key, mode="single"):
    """
    Extracts a flat list of numbers for a specific metric (e.g., 'len_in').
    Same logic as above but for numeric keys.
    """
    combined = []
    if mode == "single":
        return data.get(key, [])
    elif mode == "three_runs":
        for run_dict in data.values():
            combined.extend(run_dict.get(key, []))
    elif mode == "global":
        for topic_dict in data.values():
            for run_dict in topic_dict.values():
                combined.extend(run_dict.get(key, []))
    return combined

def save_stats_to_file(data:dict, filepath:str, mode:str="single", part:int=1):
    """
        Takes a dictionary with a number of lists, to print and save stats in txt.
        Keys: {"len_in"            : len_in,
               "len_out"           : len_out,
               "out_sent"          : out_sent,
               "out_sent1_len"     : out_sent1_len,
               "AoA_proxy".        : aoa_proxy,
               "AoA_list".         : aoa_list,
               "syntax_depth"      : syntax_depth,
               "lexical_complex"   : lexical_complexity,
               "num_particles"     : total_particle,
               "num_particles_skip": particle_skip,
               "full"              : per_topic_stats, 
              }
        
        Args:
                 part (int): 0 = "per topic", 1 = "overall"
            max_val_2 (int): for scaling the heatmap; mode=2
            max_val_3 (int): for scaling the heatmap; mode=3
             to_plot (bool): True==plotting, False==no plotting
   
    """
    which_part = ["per topic", "overall"]
    p_label = which_part[part]
    
    with open(filepath, "w", encoding="utf-8") as f:
        # helper function:
        def process_stat(label, key, is_header=False):
            # write header
            if is_header:
                f.write(f"\n---[{p_label.upper()}] - {label}:---\n")
                return
            
            # label: stat names, e.g. "#chars input per topic:"
            f.write(f"{label} {p_label}:\n")
            
            # --- THREE RUNS ---
            if mode == "three_runs":
                # to get total of 3xRuns
                all_vals = []
                # from run 1 to 3
                for r_id in sorted(data.keys()):
                    # per run key, e.g. data['1']['len_in']
                    vals = data[r_id].get(key, [])
                    avg, v_min, v_max, v_std = get_avg_min_max(vals)
                    f.write(f"\tRun {r_id} Avg: {avg:.3f}\n")
                    # collect all runs for specific topcic
                    all_vals.extend(vals)
                
                # combined 3 runs per topic
                g_avg, g_min, g_max, g_std = get_avg_min_max(all_vals)
                block = (f"TOTAL (three runs):\n"
                         f"\tAvg: {g_avg}\n"
                         f"\tMin: {g_min}\n"
                         f"\tMax: {g_max}\n"
                         f"\tStd: {g_std}\n")
                f.write(block)  
            
            # --- ONE RUN / OVERALL ---
            else:
                # Access the flat list directly from the dictionary
                vals = data.get(key, [])
                avg, v_min, v_max, v_std = get_avg_min_max(vals)
                # Format block
                block = (f"\tAvg: {avg}\n"
                         f"\tMin: {v_min}\n"
                         f"\tMax: {v_max}\n"
                         f"\tStd: {v_std}\n")
                f.write(block)      
                
        # --- BASIC STATS ---
        process_stat("BASIC STATS", None, is_header=True)
        process_stat(f"#chars input:", "len_in")
        process_stat(f"#chars output:", "len_out")
        process_stat(f"#sentence output:", "out_sent")
        process_stat(f"#chars for first sentence:", "out_sent1_len")

        # --- EXECUTE PSYCHOLING STATS ---
        print()
        process_stat("PSYCHOLING STATS", None, is_header=True)
        process_stat("Avg AoA (proxy):", "AoA_proxy")
        process_stat("Avg AoA (list)", "AoA_list")
        process_stat("Avg depth:", "syntax_depth")
        process_stat("Avg Lexical complexity (#kanjis):", "lexical_complex")
        process_stat("Avg #Particles (common):", "num_particles")  
        process_stat("Avg #Particles (skip):", "num_particles_skip")  
        
        # --- EXECUTE FORMALITY ---
        print()
        process_stat("PER FORMALITY", None, is_header=True)
        
        # Formatting the dictionary output for the file
        def write_formality_block(sample_list, run_prefix=""):
            # Get data from your existing analyzer functions
            avg_dict, avg_sent_dict, counts, counts_sent = get_avg_out_performality(stat_info=sample_list)
            averages_psych_ling_dict = get_avg_psych_ling_performality(stat_info=sample_list)
            
            header_prefix = f"Run {run_prefix} " if run_prefix else ""
            
            # --- Chars per Formality ---
            f.write(f"{header_prefix}Avg #chars output per formality:\n")
            for lv in [1, 2, 3]:
                f.write(f"\tF{lv}: {avg_dict.get(lv, 0):.2f} (#samples: {counts.get(lv, 0)})\n")
            
            # --- Sentences per Formality ---
            f.write(f"\n{header_prefix}Avg #sent output per formality:\n")
            for lv in [1, 2, 3]:
                f.write(f"\tF{lv}: {avg_sent_dict.get(lv, 0):.2f} (#samples: {counts_sent.get(lv, 0)})\n")

            # --- Psycholing metrics per Formality (The loop for all metrics) ---
            for metric, levels in averages_psych_ling_dict.items():
                if metric == "Samples": continue
                f.write(f"\n{header_prefix}Avg {metric} per formality:\n")
                for lv in [1, 2, 3]:
                    val = levels.get(lv, 0)
                    num = averages_psych_ling_dict.get("Samples", {}).get(lv, 0)
                    f.write(f"\tF{lv}: {val:.2f} (#samples: {num})\n")
            f.write("-" * 40 + "\n")

        # Execute the formality logic based on mode
        if mode == "three_runs":
            # Breakdown for each run individually
            for r_id in sorted(data.keys()):
                write_formality_block(data[r_id]["full"], run_prefix=r_id)
            
            # Combine all 'full' lists to show the total Topic average
            all_full = []
            for r_id in data: all_full.extend(data[r_id]["full"])
            f.write("TOPIC TOTAL Per Formality:\n")
            write_formality_block(all_full)
        else:
            # Simple single report
            write_formality_block(data["full"])
                
    print(f"***Saved stat file in: {filepath}")
    

def generate_heatmaps(data: dict, filename_base: str, save_path: str, mode: str = "single"):
    """
    Handles the logic of extracting the 'full' list and creating all 3 heatmaps.
    """
    # 1. Extraction: Get the flat list of samples
    if mode == "three_runs":
        # Combine Run 1, 2, and 3 samples into one list for the topic heatmap
        samples = []
        for r_id in data:
            samples.extend(data[r_id].get("full", []))
    else:
        # Just use the single run's samples
        samples = data.get("full", [])

    if not samples:
        print(f"!!! No samples found for heatmaps: {filename_base}")
        return

    # 2. Call the 3 Matrix Modes
    # Mode 1: Formality Input vs Output
    matrix_plot(
        samples, 
        filename=os.path.join(save_path, f"{filename_base}_f"), 
        mode=1
    )
    
    # Mode 2: Output Formality vs Output Gender
    matrix_plot(
        samples, 
        filename=os.path.join(save_path, f"{filename_base}_f_g"), 
        mode=2
    )
    
    # Mode 3: Formality-Gender Input vs Output (The 9x9 matrix)
    matrix_plot(
        samples, 
        filename=os.path.join(save_path, f"{filename_base}_FG"), 
        mode=3
    )

    print(f"*** Heatmaps generated for: {filename_base}")
   
    
def plot_all_boxplots(lists:dict, part:int):
    """
        Takes a dictionary with a number of lists, and creates boxplots - excluding the "full".
        Keys: {"len_in"            : len_in,
               "len_out"           : len_out,
               "out_sent"          : out_sent,
               "out_sent1_len"     : out_sent1_len,
               "AoA_proxy".        : aoa_proxy,
               "AoA_list".         : aoa_list,
               "syntax_depth"      : syntax_depth,
               "lexical_complex"   : lexical_complexity,
               "num_particles"     : total_particle,
               "num_particles_skip": particle_skip,
               "full"              : per_topic_stats, 
              }
    """
    parts = ["perTopic", "overall"]
    # today's date, e.g. Feb12
    new_dir = datetime.now().strftime("%b%d")
    FiNAL_DIR_path = os.path.join(OUTPUT_DIR_STATS, new_dir)
    _check_dir_exist(FiNAL_DIR_path)
    
    plot_dir = "boxplots"
    dir_in_FiNAL_DIR_path = os.path.join(FiNAL_DIR_path, plot_dir)
    _check_dir_exist(dir_in_FiNAL_DIR_path)
    
    for stat_type, num_list in lists.items():
        if stat_type != "full":
            file = f"distribution_plot_{str(stat_type)}_{parts[part]}"
            plot_boxplot(data=num_list, 
                        plot_title=str(stat_type), 
                        filename=os.path.join(dir_in_FiNAL_DIR_path, file))


## --- COMBINE ALL FOR SINGLE FILE --- ##
def single_file_stats(filepath:str):
    """ Calculate all basic and psycholinguistc stats on one file. """
    # basic stats
    len_in, len_out, out_sent, out_sent1_len = [], [], [], []
    # psycholing. stats
    aoa_proxy, aoa_list = [], []
    syntax_depth, lexical_complexity = [], []
    total_particle, particle_skip = [], []
    # all combined
    per_topic_stats = []
    
    with open(filepath, 'r') as f:
        data = json.load(f)

        # sometimes all {} in one list, sometimes not
        # if len(data)==1:
        #     data = data[0]
        
        # should be 180 for an individual file 
        print(f"\nData size: {len(data)}")
        
        # per file
        for item in data:
            for q_id, info in item.items():
                # FILENAME = f"{info['topic']}_20_"
                lvl_gend = get_formlvl_gender(info=info)
                counts = count_output_len(info=info)
                # "AoA", "AoA_w_list", "syntax_depth", "lexical_complexity", "total_count_particle", "count_particle_skip"
                aoa_complex = get_aoa_complexities(info=info)
                # {'input': ['3', 'neutral'], 'output': [2, 'neutral'], 'input_len': 25, 'output_len': 67, 'num_sent_out': 1}
                combined_nums = lvl_gend | counts | aoa_complex
                
                per_topic_stats.append(combined_nums)
                
                # basic stats per topic
                len_in.append(combined_nums['input_len'])
                len_out.append(combined_nums['output_len'])
                out_sent.append(combined_nums['num_sent_out'])
                out_sent1_len.append(combined_nums['out_len_first'])
                
                # psycholing. stats per topic
                aoa_proxy.append(combined_nums["AoA"])
                aoa_list.append(combined_nums["AoA_w_list"])
                syntax_depth.append(combined_nums["syntax_depth"])
                lexical_complexity.append(combined_nums["lexical_complexity"])
                total_particle.append(combined_nums["total_count_particle"])
                particle_skip.append(combined_nums["count_particle_skip"])
        
    # save all stat lists into one dict
    statlists_dict = {"len_in": len_in,
                      "len_out": len_out,
                      "out_sent": out_sent,
                      "out_sent1_len": out_sent1_len,
                      "AoA_proxy": aoa_proxy,
                      "AoA_list": aoa_list,
                      "syntax_depth": syntax_depth,
                      "lexical_complex": lexical_complexity,
                      "num_particles": total_particle,
                      "num_particles_skip": particle_skip,
                      "full": per_topic_stats, 
                    }
    
    return statlists_dict


## --- INDIVIDUAL RUNS--- ##
def individual_runs(file_paths_list:list):
    if len(file_paths_list) == 3:
        
        count = 1
        individual_stats = {}
        for file in file_paths_list:
            filepath = os.path.join(OUTPUT_DIR, file)
            statlist_dict, FILENAME = single_file_stats(filepath=filepath)
            
            individual_stats[count] = statlist_dict
            count += 1
    
    stats_run1 = individual_stats[1]
    stats_run2 = individual_stats[2]
    stats_run3 = individual_stats[3]
    
    return stats_run1, stats_run2, stats_run3


# def check_FG_now(filepath:str):
    
#     with open(filepath, 'r') as f:
#         data = json.load(f)[0]  # [{}, {}, ...]
        
#         # per file
#         for q_id, info in data.items():
#             FILENAME = f"{info['topic']}_20_Check"
#             lvl_gend = get_formlvl_gender(info=info)
            
            
            
# sent = "効果的なプレゼンテーションの仕方には以下のようなポイントがあります：\n\n1. **目的を明確に**：プレゼンの目的を明確にし、何を伝えたいのかを明確にします\n\n2. **構成を整え**：タイトル、概要、主なポイント、結論の順序で構成します。\n\n3. **わかりやすくする**：データや図表を使って、情報をわかりやすく伝える。\n\n4. **視覚的な要素を活用**：スライドのデザイン、図表、動画などを使って、視覚的に"
# s, s1 = _get_sent_first_sent(output_clean=sent)

# for i in s:
#     print(i)

# se = "効：\n\n1. **目的を明確に*。"
# sss, ss1 = _get_sent_first_sent(se)
# for i, k in enumerate(ss1):
#     print(i, k)

# print(len(se))
# print(s1)
