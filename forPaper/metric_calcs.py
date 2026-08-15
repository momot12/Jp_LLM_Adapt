import os
import numpy as np
from src.stats.f1_acc import get_macro_class_f1_acc
from src.stats.conv_t import conv_a_b_t


def calc_f1_acc_formality_filtered_forPaper(filtered_output:dict, sp_gender_raw:str, question_gender:str, output_path:str):
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

    # format it for output file
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

    # write out to file
    filename = f"{list(filtered_output.keys())[0]}_f1_acc_{sp_gender_raw}_{question_gender}.txt"
    out_file = os.path.join(output_path, filename)
    
    os.makedirs(output_path, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines) + "\n")
        
    # print(f"\nSuccessfully saved metrics report to: {out_file}")
    


def calc_conv_t_formality_filtered_forPaper(filtered_output: dict, sp_gender_raw: str, question_gender: str, output_path: str):
    """
        Calculates convergence scores (Danescu-Niculescu-Mizil & Lee, 2011) 
        per run (1, 2, 3) and total (mean ± std) for filtered data, 
    """
    # [(in_form, out_form)] mapped as pairs per run
    per_runs_data = {1: [], 2: [], 3: []}
    total_samples_count = 0

    # collect data groups
    for model_key, topics_dict in filtered_output.items():
        # {topic: {run 1: {q_id: [in, out]}}}
        for topic_id, runs_dict in topics_dict.items():
            #{run 1: {q_id: [in, out]}
            for run_id, q_dict in runs_dict.items():
                # {q_id: [in, out]}
                for q_key, val in q_dict.items():
                    in_form, out_form = int(val[0]), int(val[1])
                    per_runs_data[run_id].append((in_form, out_form))
                    total_samples_count += 1
    
    # Num samples per run
    samples_per_run = len(per_runs_data[1]) if 1 in per_runs_data else 0
    
    # calculate convergence scores for each run
    run_metrics = {}
    for run_id in [1, 2, 3]:
        run_pairs = per_runs_data[run_id]
        if run_pairs:
            run_metrics[run_id] = {
                1: conv_a_b_t(pairs=run_pairs, t=1),
                2: conv_a_b_t(pairs=run_pairs, t=2),
                3: conv_a_b_t(pairs=run_pairs, t=3)
            }
        else:
            run_metrics[run_id] = {1: 0.0, 2: 0.0, 3: 0.0}

    # Calculate Total (Mean ± Std) across the 3 runs for each formality level
    total_metrics = {}
    final_scores_per_run_avg = []
    
    for lvl in [1, 2, 3]:
        scores = [run_metrics[run][lvl] for run in [1, 2, 3] if lvl in run_metrics[run]]
        if scores:
            mean_val = float(np.mean(scores))
            std_val = float(np.std(scores))
            total_metrics[lvl] = f"{mean_val:.4f} ± {std_val:.4f}"
            final_scores_per_run_avg.append(mean_val)
        else:
            total_metrics[lvl] = "0.0000 ± 0.0000"
            
    # calculate Final convergence score (mean and std across the three formality levels)
    final_score_mean = float(np.mean(final_scores_per_run_avg)) if final_scores_per_run_avg else 0.0
    final_score_std = float(np.std(final_scores_per_run_avg)) if final_scores_per_run_avg else 0.0

    # format it for output file
    output_lines = []
    output_lines.append(f"#Samples = {samples_per_run} {question_gender} x 3 runs = {total_samples_count}")
    output_lines.append("OVERALL:")
    
    for run_id in [1, 2, 3]:
        output_lines.append(f"\tRun {run_id}:")
        if run_metrics[run_id]:
            for lvl in [1, 2, 3]:
                score = run_metrics[run_id][lvl]
                output_lines.append(f"\t\tConv(formality {lvl}): {score:.4f}")

    output_lines.append("\tTotal:")
    for lvl in [1, 2, 3]:
        val_str = total_metrics[lvl]
        output_lines.append(f"\t\tConv(formality {lvl}): {val_str}")

    output_lines.append("")
    output_lines.append(f"Final convergence score: {final_score_mean:.4f} ± {final_score_std:.4f}")

    # Write output file
    filename = f"{list(filtered_output.keys())[0]}_conv_formality_{sp_gender_raw}_{question_gender}.txt"
    out_file = os.path.join(output_path, filename)
    
    os.makedirs(output_path, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines) + "\n")
        
    # print(f"\nSuccessfully saved convergence report to: {out_file}")