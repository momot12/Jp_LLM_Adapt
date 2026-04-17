import sys
import os
import json
import numpy as np 
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from src.data_process.detect_f_g import DetectFG
from src.stats.psycholing_measures import PsychoLingMeasures
from src.stats.plots_heatmaps import matrix_plot, plot_boxplot, plot_barplot
from src.more_analysis.bleu_score import calc_bleu_score
from src.utils.helper_functions import (OUTPUT_DIR_STATS, 
                                        _check_dir_exist,
                                        _get_sent_first_sent,
                                        get_avg_min_max)


# Init instance
NLP_MODEL = PsychoLingMeasures()

## --- BLEU FILTER --- #
def _get_sent_first_bleu_filter(info:dict, threshold:float=0.8):
    """
        Takes input and output QA pair, calculated BLEU score.
        If BLEU score is >theshold (e.g. 0.8), then take second sentence as first sentence.
        If output only has one sentence, keep that sentence despite threshold.
    """
    input = str(info["question"])
    output_clean = str(info["only_out"])
    sentences, first_sent = _get_sent_first_sent(output_clean=output_clean)
    
    # calculate BLEU score per QA pair
    score = calc_bleu_score(reference_str=input, candidate_str=first_sent)
    
    if score > threshold:
        if len(sentences) > 1:
            # take second sentence
            optional_first_sent = sentences[1]
            return sentences, optional_first_sent
        else:
            # only one sent output -> keep original first sent
            return sentences, first_sent
    else:
        # score < threshold -> keep original first sent
        return sentences, first_sent


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
    # filter first sentence based on BLEU score 
    _, first_sent = _get_sent_first_bleu_filter(info=info)
    # take first sent to measure formality and gender 
    formal_gend_out = measure_formlvl_gender(first_sent)
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
    sentences, first_sent = _get_sent_first_bleu_filter(info=info)
    
    # if len(output_clean) == 0:
    #     print(f"Clean output:\n{output_clean}")
    #     print(f"Original output:\n{info['output']}\n\n")
    
    len_info = {"input_len": len(input),
                "output_len": len(output_clean),
                "num_sent_out": len(sentences),
                "out_len_first": len(first_sent)
                }
    
    return len_info

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
    _, first_sent = _get_sent_first_bleu_filter(info=info)
    stats = NLP_MODEL.get_psycholinguistic_metrics(sentence=first_sent)
    return stats

## --- PER FORMALITY FUNCTION --- ##
def get_avg_out_performality(stat_info:dict):
    """
        Takes in the dictionary with all stats. 
        Calculates average of each formality per metric
        
        Return a dictionary with per formality avg per metric.
    """
    # get all metric keys, e.g. "AoA_proxy"
    excluded = ['input', 'output', 'target']
    metric_keys = [k for k in stat_info[0].keys() if k not in excluded]
    
    # init
    counts = {1: 0, 2: 0, 3: 0}
    totals = {title: {1: 0.0, 2: 0.0, 3: 0.0} for title in metric_keys}

    for item in stat_info:
        level = int(item['output'][0])
        counts[level] += 1
        
        for metric in metric_keys:
            totals[metric][level] += float(item.get(metric, 0)) # if item missing, then add 0
    
    averages = {metric: {} for metric in metric_keys}
    averages["Samples"] = {level: counts[level] for level in [1, 2, 3]}
    
    for metric in metric_keys:
        # print(f"Avg {metric} per formality:")
        for level in [1, 2, 3]:
            # calc avg
            num = counts[level]
            avg = totals[metric][level] / num if num > 0 else 0.0
            # store in dict; e.g. {'AoA': {1: avg, 2: avg, 3: avg}, 'stat':{1:...}}    
            averages[metric][level] = round(avg, 3)
        
    return averages

def _format_formality_lines(f_stats: dict, title_label: str, performality_titles: dict):
    """
    Helper to turn a formality stats dictionary into formatted text lines.
    Args:
        f_stats (dict)    : per formality stats returned by get_avg_out_performality()
        title_label (str) : e.g. "Run 1", "TOPIC TOTAL", or "OVERALL TOTAL"
        performality_titles: The mapping of keys to the stats names 
    """
    lines = []
    lines.append(f"\t--- {title_label} ---\n")
    
    f_samples = f_stats["Samples"]
    for m_key, f_levels in f_stats.items():
        if m_key == "Samples": continue
            
        display_name = performality_titles.get(m_key, m_key)
        lines.append(f"\t{display_name}\n")    
        lines.append(f"\t\tF1: {f_levels[1]} (N={f_samples[1]})\n")
        lines.append(f"\t\tF2: {f_levels[2]} (N={f_samples[2]})\n")
        lines.append(f"\t\tF3: {f_levels[3]} (N={f_samples[3]})\n")
        
    lines.append("\n")
    return lines


## --- SAVING, PLOTTING--- ##
def save_stats_to_file(data:dict, model_name:str, output_path:str):
    """
        Takes a dictionary with topics, and statistics of 3 runs per topic.
        Saves six .txt files with statistics per topic (each run, total), and over al topics.
        
        Args:
            data (dict)     : {topic1: {run1: {stat: [numbers]}, run2: {}}}
            model_name (str): model name, e.g. qwen_4B
            filepath (str)  : a filepath to a created directory to save statistics
              
    """
    basic_metrics = {
        "#chars input:": "len_in",
        "#chars output:": "len_out",
        "#sentence output:": "out_sent",
        "#chars for first sentence:": "out_sent1_len"
    }

    psych_metrics = {
        "AoA (proxy):": "AoA_proxy",
        "AoA (list):": "AoA_list",
        "Depth:": "syntax_depth",
        "Lexical complexity (#kanjis):": "lexical_complex",
        "Particles (common):": "num_particles",
        "Particles (skip):": "num_particles_skip"
    }
    
    performality_titles = basic_metrics | psych_metrics
    
    # used for titles
    categories = [
        ("BASIC STATS", basic_metrics),
        ("PSYCHOLING. STATS", psych_metrics)
    ]
    
    # for 3xRuns
    overall_data = {1: {}, 2: {}, 3: {}}
    # for per formality
    global_full_data = []
    
    for topic, runs in data.items():
        topic_lines = [f"=== PER TOPIC FOR: {model_name} ===\n\n"]
        # for per formality
        topic_full_data = []
        
        for section_name, metrics_dict in categories:
            # e.g. --- [PER TOPIC] BASIC STATS: career --- 
            topic_lines.append(f"--- [PER TOPIC] {section_name}: {topic.upper()} ---\n")
                
            for stat_title, key in metrics_dict.items():
                # e.g. #chars input:
                topic_lines.append(f"{stat_title}\n")
                all_3_runs_combined = []
            
                for run_id in [1, 2, 3]:
                    # list of nums for specific run
                    run_data = runs.get(run_id, {}).get(key, [])
                    # collect all runs here (each a list of num)
                    all_3_runs_combined.extend(run_data)
                    
                    # add for total
                    if key not in overall_data[run_id]:
                        overall_data[run_id][key] = []
                    overall_data[run_id][key].extend(run_data)
                    
                    # RUN: get avg, min, max, std
                    if run_data:
                        avg, v_min, v_max, std = get_avg_min_max(run_data)
                        topic_lines.append(f"\tRun {run_id}:\n"
                                           f"\t\tAvg: {avg}\n"
                                           f"\t\tMin: {v_min}\n"
                                           f"\t\tMax: {v_max}\n"
                                           f"\t\tStd: {std}\n")
                
                # TOTAL: all three runs
                if all_3_runs_combined:
                    t_avg, t_min, t_max, t_std = get_avg_min_max(all_3_runs_combined)
                    topic_lines.append(f"\tTOTAL (three runs):\n"
                                       f"\t\tAvg: {t_avg}\n"
                                       f"\t\tMin: {t_min}\n"
                                       f"\t\tMax: {t_max}\n"
                                       f"\t\tStd: {t_std}\n\n")

        # [PER TOPIC] PER FORMALITY
        topic_lines.append(f"--- [PER TOPIC] PER FORMALITY: {topic.upper()} ---\n")
        for run_id in [1, 2, 3]:
            run_samples = runs.get(run_id, {}).get('full', [])
            if run_samples:
                topic_full_data.extend(run_samples)
                global_full_data.extend(run_samples)
                run_f_stats = get_avg_out_performality(run_samples)
                topic_lines.extend(_format_formality_lines(run_f_stats, f"RUN {run_id}", performality_titles))
        
        if topic_full_data:
            topic_total_f = get_avg_out_performality(topic_full_data)
            topic_lines.extend(_format_formality_lines(topic_total_f, "TOPIC TOTAL", performality_titles))

        # save individual topic file
        outfile = os.path.join(output_path, f"{model_name}_{topic}_pertopic.txt")
        with open(outfile, "w", encoding="utf-8") as f:
            f.writelines(topic_lines)
         
    # ================ # 
    overall_lines = [f"=== OVERALL FOR: {model_name} ===\n\n"]

    for section_name, metrics_stat in categories:
        overall_lines.append(f"--- OVERALL {section_name} ---\n")
        
        for stat_title, key in metrics_stat.items():
            overall_lines.append(f"{stat_title}\n")
            # every datapoint (5x3)
            overall_total_list = []
            
            for run_id in [1, 2, 3]:
                run_vals = overall_data.get(run_id, {}).get(key, [])
                if run_vals:
                    overall_total_list.extend(run_vals)
                    # Calculate stats for Run X across all topics
                    o_avg, o_min, o_max, o_std = get_avg_min_max(run_vals)
                    overall_lines.append(f"\tRun {run_id}: \n"
                                         f"\t\tAvg: {o_avg}\n"
                                         f"\t\tMin: {o_min}\n"
                                         f"\t\tMax: {o_max}\n"
                                         f"\t\tStd: {o_std}\n")

            # overall 
            if overall_total_list:
                g_avg, g_min, g_max, g_std = get_avg_min_max(overall_total_list)
                overall_lines.append(f"\tOVERALL TOTAL: \n"
                                    f"\t\tAvg: {g_avg}\n"
                                    f"\t\tMin: {g_min}\n"
                                    f"\t\tMax: {g_max}\n"
                                    f"\t\tStd: {g_std}\n\n")

    # [OVERALL] PER FORMALITY
    if global_full_data:
        overall_lines.append(f"--- OVERALL PER FORMALITY (ALL TOPICS) ---\n")
        g_f_stats = get_avg_out_performality(global_full_data)
        overall_lines.extend(_format_formality_lines(g_f_stats, "OVERALL TOTAL", performality_titles))
       
    # write overall file
    overall_outfile = os.path.join(output_path, f"{model_name}_over5topics_overall.txt")
    with open(overall_outfile, "w", encoding="utf-8") as f:
        f.writelines(overall_lines)

def generate_heatmaps(data: dict, model_name: str, output_path: str):
    """
    Generates heatmaps for individual runs, topic totals, and overall totals.
    Scaling is handled automatically per plot.
    """
    overall_runs_data = {1: [], 2: [], 3: []}
    overall_total_data = [] 

    topic_folder = os.path.join(output_path, "PerTOPIC_heatmaps")
    _check_dir_exist(topic_folder)

    for topic, runs in data.items():
        topic_total_data = []
        for run_id in [1, 2, 3]:
            run_samples = runs.get(run_id, {}).get("full", [])
            
            if run_samples:
                topic_total_data.extend(run_samples)
                overall_runs_data[run_id].extend(run_samples)
                overall_total_data.extend(run_samples)

                # [RUNs] Per Topic
                base_fname = os.path.join(topic_folder, f"{model_name}_{topic}_run_{run_id}")
                matrix_plot(run_samples, filename=base_fname)

        # [PER TOPIC] TOTAL - 3 runs
        if topic_total_data:
            base_fname = os.path.join(topic_folder, f"{model_name}_{topic}_TOTAL")
            matrix_plot(topic_total_data, filename=base_fname)

    # --- OVERALL STATS ---
    overall_folder = os.path.join(output_path, "OVERALL_heatmaps")
    _check_dir_exist(overall_folder)

    # [RUNs] OVERALL 
    for run_id, run_samples in overall_runs_data.items():
        if run_samples:
            base_fname = os.path.join(overall_folder, f"{model_name}_overall_run_{run_id}")
            matrix_plot(run_samples, filename=base_fname)

    # [OVERALL] TOTAL (3 runs, 5 topics)
    if overall_total_data:
        base_fname = os.path.join(overall_folder, f"{model_name}_OVERALL_TOTAL")
        matrix_plot(overall_total_data, filename=base_fname)

def generate_barplots(data: dict, model_name: str, output_path: str):
    """
        Takes in a dictionary with all topics, all runs stats and 
        generates bar plots.
    """
    basic_metrics = {
        "Avg IN": "input_len",
        "Avg OUT": "output_len",
        "Avg OUT (1st)": "out_len_first",
        "Avg OUT_sent": "num_sent_out"
    }

    psych_metrics = {
        "AoA (proxy)": "AoA",
        "AoA (list)": "AoA_w_list",
        "Depth": "syntax_depth",
        "Lexical complexity": "lexical_complexity",
        "Particles (common)": "total_count_particle",
        "Particles (skip)": "count_particle_skip"
    }
    
    all_metrics = basic_metrics | psych_metrics
    
    # [PER TOPIC] create folder
    topic_folder = os.path.join(output_path, "PerTopic_barplots")
    _check_dir_exist(topic_folder)
    
    overall_runs_full = {1: [], 2: [], 3: []}
    for topic, runs in data.items():
        # get full list of each run
        r1_full = runs.get(1, {}).get('full', [])
        r2_full = runs.get(2, {}).get('full', [])
        r3_full = runs.get(3, {}).get('full', [])
        # accumulate over all topics, each run
        overall_runs_full[1].extend(r1_full)
        overall_runs_full[2].extend(r2_full)
        overall_runs_full[3].extend(r3_full)

        # [PER TOPIC]
        # --- BASIC STATS ---
        b_keys = list(basic_metrics.values())
        b_labels = list(basic_metrics.keys())
        
        # 3x 
        br1 = [np.mean([float(i.get(k, 0)) for i in r1_full]) if r1_full else 0 for k in b_keys]
        br2 = [np.mean([float(i.get(k, 0)) for i in r2_full]) if r2_full else 0 for k in b_keys]
        br3 = [np.mean([float(i.get(k, 0)) for i in r3_full]) if r3_full else 0 for k in b_keys]
        # 1x
        br1_f = [np.mean([float(i.get(k, 0)) for i in r1_full]) if r1_full else 0 for k in b_keys]
        
        basic_fname = os.path.join(topic_folder, f"{model_name}_{topic}_basic_stats.png")
        plot_barplot(labels=b_labels,
                     r1_means=br1, r2_means=br2, r3_means=br3, 
                     run1_f=br1_f,
                     title=f"{topic.upper()} - Basic Stats",
                     save_path=basic_fname)
        
        # --- PSYCH STATS ---
        p_keys = list(psych_metrics.values())
        p_labels = list(psych_metrics.keys())
        # 3x
        pr1 = [np.mean([float(item.get(k, 0)) for item in r1_full if k in item]) if r1_full else 0 for k in p_keys]
        pr2 = [np.mean([float(item.get(k, 0)) for item in r2_full if k in item]) if r2_full else 0 for k in p_keys]
        pr3 = [np.mean([float(item.get(k, 0)) for item in r3_full if k in item]) if r3_full else 0 for k in p_keys]
        # 1x
        pr1_f = [np.mean([float(item.get(k, 0)) for item in r1_full if k in item]) if r1_full else 0 for k in p_keys]
        
        psych_fname = os.path.join(topic_folder, f"{model_name}_{topic}_psych_stats.png")
        plot_barplot(labels=p_labels, 
                     r1_means=pr1, r2_means=pr2, r3_means=pr3, 
                     run1_f=pr1_f,
                     title=f"{topic.upper()} - Psycholinguistic Stats", 
                     save_path=psych_fname)
        
        
        # [PER TOPIC] - PER FORMALITY
        stats_r1 = get_avg_out_performality(r1_full)
        stats_r2 = get_avg_out_performality(r2_full)
        stats_r3 = get_avg_out_performality(r3_full)

        for pretty_name, key in all_metrics.items():
            f_labels = ["F1", "F2", "F3"]
            
            # Extract values for each formality level (1, 2, 3)
            fr1 = [stats_r1.get(key, {}).get(f, 0) for f in [1, 2, 3]]
            fr2 = [stats_r2.get(key, {}).get(f, 0) for f in [1, 2, 3]]
            fr3 = [stats_r3.get(key, {}).get(f, 0) for f in [1, 2, 3]]
            
            # We use fr1 as the "1x Run" comparison baseline
            formality_fname = os.path.join(topic_folder, f"{model_name}_{topic}_{key}_per_formality.png")
            
            plot_barplot(labels=f_labels, 
                         r1_means=fr1, r2_means=fr2, r3_means=fr3, 
                         run1_f=fr1, 
                         title=f"{topic.upper()} - {pretty_name} (Per Formality)", 
                         save_path=formality_fname)
    
    ### --- [OVERALL] --- ###
    overall_folder = os.path.join(output_path, "OVERALL_barplots")
    _check_dir_exist(overall_folder)

    # collect stats
    or1_full = overall_runs_full[1]
    or2_full = overall_runs_full[2]
    or3_full = overall_runs_full[3]

    # --- OVERALL BASIC STATS ---
    o_b_keys = list(basic_metrics.values())
    o_b_labels = list(basic_metrics.keys())
    # 3x vs 1x 
    o_br1 = [np.mean([float(i.get(k, 0)) for i in or1_full]) if or1_full else 0 for k in o_b_keys]
    o_br2 = [np.mean([float(i.get(k, 0)) for i in or2_full]) if or2_full else 0 for k in o_b_keys]
    o_br3 = [np.mean([float(i.get(k, 0)) for i in or3_full]) if or3_full else 0 for k in o_b_keys]
    o_br1_f = [np.mean([float(i.get(k, 0)) for i in or1_full]) if or1_full else 0 for k in o_b_keys]

    overall_basic_fname = os.path.join(overall_folder, f"{model_name}_OVERALL_basic_stats.png")
    plot_barplot(labels=o_b_labels,
                 r1_means=o_br1, r2_means=o_br2, r3_means=o_br3, 
                 run1_f=o_br1_f,
                 title=f"OVERALL - Basic Stats",
                 save_path=overall_basic_fname)

    # --- OVERALL PSYCH STATS ---
    o_p_keys = list(psych_metrics.values())
    o_p_labels = list(psych_metrics.keys())

    o_pr1 = [np.mean([float(i.get(k, 0)) for i in or1_full]) if or1_full else 0 for k in o_p_keys]
    o_pr2 = [np.mean([float(i.get(k, 0)) for i in or2_full]) if or2_full else 0 for k in o_p_keys]
    o_pr3 = [np.mean([float(i.get(k, 0)) for i in or3_full]) if or3_full else 0 for k in o_p_keys]
    o_pr1_f = [np.mean([float(i.get(k, 0)) for i in or1_full]) if or1_full else 0 for k in o_p_keys]

    overall_psych_fname = os.path.join(overall_folder, f"{model_name}_OVERALL_psych_stats.png")
    plot_barplot(labels=o_p_labels, 
                 r1_means=o_pr1, r2_means=o_pr2, r3_means=o_pr3, 
                 run1_f=o_pr1_f,
                 title=f"OVERALL - Psycholinguistic Stats", 
                 save_path=overall_psych_fname)

    # --- OVERALL PER FORMALITY ---
    o_stats_r1 = get_avg_out_performality(or1_full)
    o_stats_r2 = get_avg_out_performality(or2_full)
    o_stats_r3 = get_avg_out_performality(or3_full)

    for pretty_name, key in all_metrics.items():
        f_labels = ["F1", "F2", "F3"]
        
        # Extract the 3 formality values for each run
        ofr1 = [o_stats_r1.get(key, {}).get(f, 0) for f in [1, 2, 3]]
        ofr2 = [o_stats_r2.get(key, {}).get(f, 0) for f in [1, 2, 3]]
        ofr3 = [o_stats_r3.get(key, {}).get(f, 0) for f in [1, 2, 3]]
        
        overall_f_fname = os.path.join(overall_folder, f"{model_name}_OVERALL_{key}_per_formality.png")
        
        plot_barplot(labels=f_labels, 
                     r1_means=ofr1, r2_means=ofr2, r3_means=ofr3, 
                     run1_f=ofr1, 
                     title=f"OVERALL - {pretty_name} (By Formality)", 
                     save_path=overall_f_fname)


# TODO
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

        # should be 180 for an individual file 
        # print(f"Data size: {len(data)}")
        
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