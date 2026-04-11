import os
import numpy as np


def conv_a_b_t(pairs:list, t:int):
    """
        Takes in a list of pairs with formality and gender info of input and output.
        Args: 
            pairs (list): list of tuples with formality and gender info
                         e.g. [([in_formality, in_gender], [out_formality, out_gender])]
            t: specifies which feature we look at, e.g. formality level 1
        
        Reference: Danescu-Niculescu-Mizil, Lee (2011).
        Formula:   Conv_A_B(t) = Trigger rate - Baseline
                               = P(b_t_a = 1 | a_t = 1) - P(b_t_a = 1)
    """
    # AB pairs if A has t
    t_in_A = [i for i in pairs if int(i[0][0]) == t]
    
    # how many times B uses t, if A uses it before
    # b_t_a = 1
    t_in_B_given_A = [i for i in t_in_A if int(i[1][0]) == t]
    # P(b_t_a = 1 | a_t = 1)
    trigger_rate = len(t_in_B_given_A)/len(t_in_A)
    
    # how often t in B across all examples
    # AB pairs if B has t
    t_in_B = [i for i in pairs if int(i[1][0]) == t]
    # P(b_t_a = 1)
    baseline = len(t_in_B)/len(pairs)
    
    # final score
    conv_a_b = trigger_rate - baseline
    
    return conv_a_b
    

def calc_conv_t(data:dict, model_name:str, output_path:str):
    """
        Takes a dictionary with topics, and statistics of 3 runs per topic.
        Calculates and saves a txt file with various convergence scores based on Danescu-Niculescu-Mizil, Lee (2011).
        
        Args:
            data (dict)     : {topic1: {run1: {stat: [numbers]}, run2: {}}}
            model_name (str): model name, e.g. qwen_4B
            filepath (str)  : a filepath to a created directory to saves output txt
              
    """
    final_scores = []
    total_per_t = {1:[], 2:[], 3:[]}
    overall_per_run = { # run_id: {lvl: [scores]}
                        1: {1: [], 2: [], 3: []},
                        2: {1: [], 2: [], 3: []},
                        3: {1: [], 2: [], 3: []}
                      }
    lines = []
    lines.append(f"=== CONVERGENCE SCORES ===\n")
    
    for topic, runs in data.items():
        lines.append(f"Topic: {topic}\n")
        pertopic_3x = []
        
        for run_id in [1, 2, 3]:
            # all infos including input/output formality & gender, and stat numbers
            run_data = runs.get(run_id, {}).get('full', [])
            
            # retrieve input: [formality, gender], output: [formality, gender]
            # [([formality, gender], [formality, gender])]
            run_pairs = [(item.get('input'), item.get('output')) 
                        for item in run_data]
    
            # collect for Total of 3 runs
            pertopic_3x.extend(run_pairs)
            
            # Each run per topic
            lines.append(f"\tRun {run_id}:\n")
            for lvl in [1, 2, 3]:
                score = conv_a_b_t(pairs=run_pairs, t=lvl)
                lines.append(f"\t\tConvA,B(formality {lvl}): {score:.4f}\n")
                # for overall per run 
                overall_per_run[run_id][lvl].append(score)

        # Over 3 runs per topic
        lines.append(f"\tTotal:\n")
        for lvl in [1, 2, 3]:
            score = conv_a_b_t(pairs=pertopic_3x, t=lvl)
            lines.append(f"\t\tConvA,B(formality {lvl}): {score:.4f}\n")
            total_per_t[lvl].append(score)
        lines.append(f"\n")
            
    # --- OVERALL -- #
    lines.append(f"OVERALL:\n")
    
    # global per run per t
    for run_id in [1, 2, 3]:
        lines.append(f"\tRun {run_id}:\n")
        for lvl in [1, 2, 3]:
            scores = overall_per_run[run_id][lvl]
            avg_run_score = sum(scores) / len(scores) if scores else 0
            lines.append(f"\t\tConv(formality {lvl}): {avg_run_score:.4f}\n")
    
    # global scores per t
    lines.append(f"\tTotal:\n")
    for lvl in [1, 2, 3]:
        scores = total_per_t[lvl]
        conv_t_per_t = sum(scores)/len(scores)
        std_per_t = float(np.std(scores)) 
        lines.append(f"\t\tConv(formality {lvl}): {conv_t_per_t:.4f} ± {std_per_t:.4f}\n")
        final_scores.append(conv_t_per_t)
    
    # Final score 
    final_score = sum(final_scores)/len(final_scores)
    final_std = float(np.std(final_scores))
    lines.append(f"\nFinal convergence score: {final_score:.4f} ± {final_std:.4f}")
    
    # write txt
    outfile = os.path.join(output_path, f"{model_name}_conv_t_scores.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.writelines(lines)
            