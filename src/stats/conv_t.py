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
    t_in_A = [i for i in pairs if i[0] == t]
    
    # how many times B uses t, if A uses it before
    # b_t_a = 1
    t_in_B_given_A = [i for i in t_in_A if i[1] == t]
    # P(b_t_a = 1 | a_t = 1)
    trigger_rate = len(t_in_B_given_A)/len(t_in_A)
    
    # how often t in B across all examples
    # AB pairs if B has t
    t_in_B = [i for i in pairs if i[1] == t]
    # P(b_t_a = 1)
    baseline = len(t_in_B)/len(pairs)
    
    # final score
    conv_a_b = trigger_rate - baseline
    
    return conv_a_b
    

def calc_conv_t_formality(data:dict, model_name:str, output_path:str):
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
            # run_pairs = [(item.get('input'), item.get('output')) 
            #             for item in run_data] 
            # retrieve only formality
            run_pairs = [(int(item.get('input')[0]), int(item.get('output')[0])) 
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
    outfile = os.path.join(output_path, f"{model_name}_conv_t_formality.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.writelines(lines)


def calc_conv_t_gender(data:dict, model_name:str, output_path:str):
    """
        Takes a dictionary with topics, and statistics of 3 runs per topic.
        Calculates and saves a txt file with various convergence scores based on Danescu-Niculescu-Mizil, Lee (2011).
        
        Args:
            data (dict)     : {topic1: {run1: {stat: [numbers]}, run2: {}}}
            model_name (str): model name, e.g. qwen_4B
            filepath (str)  : a filepath to a created directory to saves output txt
              
    """
    to_int_map = {'neutral': 1, 'female': 2, 'male': 3}
    to_gender_map = {1: 'neutral', 2: 'female' , 3: 'male'}
    
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
            # run_pairs = [(item.get('input'), item.get('output')) 
            #             for item in run_data]
            # retieve only gender
            run_pairs = [(to_int_map[str(item.get('input')[1])], 
                        to_int_map[str(item.get('output')[1])]) 
                        for item in run_data]
    
            # collect for Total of 3 runs
            pertopic_3x.extend(run_pairs)
            
            # Each run per topic
            lines.append(f"\tRun {run_id}:\n")
            for lvl in [1, 2, 3]:
                score = conv_a_b_t(pairs=run_pairs, t=lvl)
                lines.append(f"\t\tConvA,B({to_gender_map[lvl]}): {score:.4f}\n")
                # for overall per run 
                overall_per_run[run_id][lvl].append(score)

        # Over 3 runs per topic
        lines.append(f"\tTotal:\n")
        for lvl in [1, 2, 3]:
            score = conv_a_b_t(pairs=pertopic_3x, t=lvl)
            lines.append(f"\t\tConvA,B({to_gender_map[lvl]}): {score:.4f}\n")
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
            lines.append(f"\t\tConv({to_gender_map[lvl]}): {avg_run_score:.4f}\n")
    
    # global scores per t
    lines.append(f"\tTotal:\n")
    for lvl in [1, 2, 3]:
        scores = total_per_t[lvl]
        conv_t_per_t = sum(scores)/len(scores)
        std_per_t = float(np.std(scores)) 
        lines.append(f"\t\tConv({to_gender_map[lvl]}): {conv_t_per_t:.4f} ± {std_per_t:.4f}\n")
        final_scores.append(conv_t_per_t)
    
    # Final score 
    final_score = sum(final_scores)/len(final_scores)
    final_std = float(np.std(final_scores))
    lines.append(f"\nFinal convergence score: {final_score:.4f} ± {final_std:.4f}")
    
    # write txt
    outfile = os.path.join(output_path, f"{model_name}_conv_t_gender.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.writelines(lines)       
        

def calc_conv_t_fg(data:dict, model_name:str, output_path:str):
    """
        Takes a dictionary with topics, and statistics of 3 runs per topic.
        Calculates and saves a txt file with various convergence scores based on Danescu-Niculescu-Mizil, Lee (2011).
        
        Args:
            data (dict)     : {topic1: {run1: {stat: [numbers]}, run2: {}}}
            model_name (str): model name, e.g. qwen_4B
            filepath (str)  : a filepath to a created directory to saves output txt
              
    """
    # define combos
    formalities = [1, 2, 3]
    genders = ['neutral', 'female', 'male']
    all_combos = [f"{f}_{g}" for f in formalities for g in genders]
    
    final_scores = []
    # {1-neutral: [], ...}
    total_per_t = {combo: [] for combo in all_combos}
    # {run_id: {1-neutral: [scores]}
    overall_per_run = {run_id: {combo: [] for combo in all_combos}
                        for run_id in [1, 2, 3]
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
            # [(formality_gender, formality_gender), ()]
            run_pairs = [
                (f"{item.get('input')[0]}_{item.get('input')[1]}", 
                f"{item.get('output')[0]}_{item.get('output')[1]}") 
                for item in run_data
            ]
    
            # collect for Total of 3 runs
            pertopic_3x.extend(run_pairs)
            
            # Each run per topic
            lines.append(f"\tRun {run_id}:\n")
            for combo in all_combos:
                score = conv_a_b_t(pairs=run_pairs, t=combo)
                lines.append(f"\t\tConvA,B({combo}): {score:.4f}\n")
                # for overall per run 
                overall_per_run[run_id][combo].append(score)

        # Over 3 runs per topic
        lines.append(f"\tTotal:\n")
        for combo in all_combos:
            score = conv_a_b_t(pairs=pertopic_3x, t=combo)
            lines.append(f"\t\tConvA,B({combo}): {score:.4f}\n")
            total_per_t[combo].append(score)
        lines.append(f"\n")
            
    # --- OVERALL -- #
    lines.append(f"OVERALL:\n")
    
    # global per run per t
    for run_id in [1, 2, 3]:
        lines.append(f"\tRun {run_id}:\n")
        for combo in all_combos:
            scores = overall_per_run[run_id][combo]
            avg_run_score = sum(scores) / len(scores) if scores else 0
            lines.append(f"\t\tConv({combo}): {avg_run_score:.4f}\n")
    
    # global scores per t
    lines.append(f"\tTotal:\n")
    for combo in all_combos:
        scores = total_per_t[combo]
        conv_t_per_t = sum(scores)/len(scores)
        std_per_t = float(np.std(scores)) 
        lines.append(f"\t\tConv({combo}): {conv_t_per_t:.4f} ± {std_per_t:.4f}\n")
        final_scores.append(conv_t_per_t)
    
    # Final score 
    final_score = sum(final_scores)/len(final_scores)
    final_std = float(np.std(final_scores))
    lines.append(f"\nFinal convergence score: {final_score:.4f} ± {final_std:.4f}")
    
    # write txt
    outfile = os.path.join(output_path, f"{model_name}_conv_t_FG.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.writelines(lines) 
        
        
def calc_conv_t(data:dict, model_name:str, output_path:str):
    """
        Takes in a dictionary with input and output stat, formality, and gender information.
        Calculated Conv(t) (convergence scores based on Danescu-Niculescu-Mizil, Lee (2011))
        for formality, gender, and combined.
    
    """
    calc_conv_t_formality(data=data, model_name=model_name, output_path=output_path)
    calc_conv_t_gender(data=data, model_name=model_name, output_path=output_path)
    calc_conv_t_fg(data=data, model_name=model_name, output_path=output_path)