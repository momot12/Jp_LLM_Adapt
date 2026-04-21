import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def write_f1(data:dict, output_path:str):
    """
        Takes in a dictionary with topics (5) and per run (1-3) metrics.
        Metrics are per_class_F1_score, macro_F1, and accuracy
        Writes a file with all the metrics per topic and overall.
    """
    with open (output_path, 'w') as f:
        f.write(f"=== F1 SCORES, ACCURACIES ===\n")
        for topic, runs in data.items():
            # write topics
            if topic != "overall_runs":
                f.write(f"Topic: {topic}\n")
        
                # Runs 1–3
                for run_id in [1, 2, 3]:
                    # dict with per class F1, macro F1, acc
                    value = runs.get(run_id, None)
                    if value is not None:
                        f.write(f"\tRun {run_id}:\n")
                        for k, v in value.items():
                            f.write(f"\t\t{k.capitalize()}: {v:.4f}\n")
                              
                # Total
                total = runs.get("total", None)
                if total is not None:
                    f.write(f"\tTotal:\n")
                    for k, v in total.items():
                        if not k.startswith("std_"):
                            mean = total.get(k, 0.0)
                            std = total.get(f"std_{k}", 0.0)
                            f.write(f"\t\t{k.capitalize()}: {mean:.4f} ± {std:.4f}\n")
                    f.write(f"\n")
            
        # --- Overall --- #
        f.write(f"Overall:\n")
        overall = data.get("overall_runs", {})
        # Runs 1–3
        for run_id in [1, 2, 3]:
            value = overall.get(run_id, None)
            if value is not None:
                f.write(f"\tRun {run_id}:\n")
                for k, v in value.items():
                    f.write(f"\t\t{k.capitalize()}: {v:.4f}\n")
                
        # Total
        total = overall.get("total", None)
        if total is not None:
            f.write(f"\tTotal:\n")
            for k, v in total.items():
                if not k.startswith("std_"):
                    mean = total.get(k, 0.0)
                    std = total.get(f"std_{k}", 0.0)
                    f.write(f"\t\t{k.capitalize()}: {mean:.4f} ± {std:.4f}\n")


def get_macro_class_f1_acc(true:list, pred: list):
    """
        Takes in two lists - true and prediction, and calculates 
        F1 macro score and for individual classes.
        Returns a dict with F1 scores (classes and macro) and accuracy.
    """
    # get unique lables and sort -> dynamic labels (not just numbers)
    labels = sorted(list(set(true) | set(pred)))
    
    f1_macro = f1_score(y_true=true, y_pred=pred, average='macro')
    f1_perclass = f1_score(y_true=true, y_pred=pred, average=None, labels=labels)
    acc = accuracy_score(y_true=true, y_pred=pred)
    
    # add all individual labels F1, Macro F1, and accuracy
    metrics = {f"{label}_f1": float(f1) for label, f1 in zip(labels, f1_perclass)}
    metrics["macro_f1"] = float(f1_macro)
    metrics["accuracy"] = float(acc)
    return metrics
    

def calc_f1_acc_formality(data, model_name, output_path):
    """
        Takes in a dictionay with topics (5) and each run (1-3) stats.
        Calculates metrics (F1 score, accuracy) for each topic, each run, and
        over all topics, each run, and overall.
        Results are written in a .txt file.
    """
    # for all F1 scores, accuracies
    # {topic: {1: {metrics}, 2: {metrics}, 3: {metrics}, total: {metrics}}}
    f1_dict = {}
    overall_each_run = {1: [[], []], 2: [[], []], 3: [[], []]}
    
    for topic, runs in data.items():
        
        each_run = {1: [], 2: [], 3: []}
        each_run_f1 = {1: 0.0, 2: 0.0, 3: 0.0}
        
        for run_id, stat_info in runs.items():
            # get input & output formality-gender info
            samples = stat_info.get('full', [])
            input_formality = [int(s.get('input')[0]) for s in samples]
            output_formality = [int(s.get('output')[0]) for s in samples]
            
            # per topic per run
            each_run[run_id] = (input_formality, output_formality)

            # {class1_f1: xx, class2_f1: xx, ..., macro: xx, accuracy: xx}
            f1_topic_run = get_macro_class_f1_acc(true=input_formality,
                                                  pred=output_formality)
            # add F1 score for each run per topic
            each_run_f1[run_id] = f1_topic_run

        # per topic, each run 
        f1_dict[topic] = each_run_f1
        for run, stats in each_run.items():
            # add to calc each run over all topics
            overall_each_run[run][0].extend(stats[0])
            overall_each_run[run][1].extend(stats[1])
           
        # calculate F1 over all 3 runs per 
        metric_keys = each_run_f1[1].keys() # class_1_f1, etc.
        f1_topic_total = {}
        for key in metric_keys:
            # scores for each metric per run
            scores = [each_run_f1[run][key] for run in [1, 2, 3]]
            f1_topic_total[key] = float(np.mean(scores))
            f1_topic_total[f"std_{key}"] = float(np.std(scores))
        
        f1_dict[topic]['total'] = f1_topic_total
    
    ### --- OVERALL --- ###
    for run, stats in overall_each_run.items():
        in_run = stats[0]
        out_run = stats[1]
        f1_run_overall = get_macro_class_f1_acc(true=in_run,
                                                pred=out_run)
        
        # save F1 scores
        f1_dict.setdefault('overall_runs', {})[run] = f1_run_overall
    
    # Overall each run + total
    metric_keys = f1_dict['overall_runs'][1].keys()
    f1_overall = {}

    for key in metric_keys:
        # scores for each metric per run
        scores = [f1_dict['overall_runs'][run][key] for run in [1, 2, 3]]    
        f1_overall[key] = float(np.mean(scores))
        f1_overall[f"std_{key}"] = float(np.std(scores))
    
    f1_dict['overall_runs']['total'] = f1_overall
    
    # save results
    out_file = os.path.join(output_path, f"{model_name}_f1_acc_formality.txt")
    write_f1(data=f1_dict, output_path=out_file)
    

def calc_f1_acc_gender(data, model_name, output_path):
    """
        Takes in a dictionay with topics (5) and each run (1-3) stats.
        Calculates metrics (F1 score, accuracy) for each topic, each run, and
        over all topics, each run, and overall.
        Results are written in a .txt file.
    """
    # for all F1 scores, accuracies
    # {topic: {1: {metrics}, 2: {metrics}, 3: {metrics}, total: {metrics}}}
    f1_dict = {}
    overall_each_run = {1: [[], []], 2: [[], []], 3: [[], []]}
    
    for topic, runs in data.items():
        
        each_run = {1: [], 2: [], 3: []}
        each_run_f1 = {1: 0.0, 2: 0.0, 3: 0.0}
        
        for run_id, stat_info in runs.items():
            # get input & output formality-gender info
            samples = stat_info.get('full', [])
            input_gender = [str(s.get('input')[1]) for s in samples]
            output_gender = [str(s.get('output')[1]) for s in samples]
            
            # per topic per run
            each_run[run_id] = (input_gender, output_gender)

            # {class1_f1: xx, class2_f1: xx, ..., macro: xx, accuracy: xx}
            f1_topic_run = get_macro_class_f1_acc(true=input_gender,
                                                  pred=output_gender)
            # add F1 score for each run per topic
            each_run_f1[run_id] = f1_topic_run

        # per topic, each run 
        f1_dict[topic] = each_run_f1
        for run, stats in each_run.items():
            # add to calc each run over all topics
            overall_each_run[run][0].extend(stats[0])
            overall_each_run[run][1].extend(stats[1])
           
        # calculate F1 over all 3 runs per 
        metric_keys = each_run_f1[1].keys() # class_1_f1, etc.
        f1_topic_total = {}
        for key in metric_keys:
            # scores for each metric per run
            scores = [each_run_f1[run][key] for run in [1, 2, 3]]
            f1_topic_total[key] = float(np.mean(scores))
            f1_topic_total[f"std_{key}"] = float(np.std(scores))
        
        f1_dict[topic]['total'] = f1_topic_total
    
    ### --- OVERALL --- ###
    for run, stats in overall_each_run.items():
        in_run = stats[0]
        out_run = stats[1]
        f1_run_overall = get_macro_class_f1_acc(true=in_run,
                                                pred=out_run)
        
        # save F1 scores
        f1_dict.setdefault('overall_runs', {})[run] = f1_run_overall
    
    # Overall each run + total
    metric_keys = f1_dict['overall_runs'][1].keys()
    f1_overall = {}

    for key in metric_keys:
        # scores for each metric per run
        scores = [f1_dict['overall_runs'][run][key] for run in [1, 2, 3]]    
        f1_overall[key] = float(np.mean(scores))
        f1_overall[f"std_{key}"] = float(np.std(scores))
    
    f1_dict['overall_runs']['total'] = f1_overall
    
    # save results
    out_file = os.path.join(output_path, f"{model_name}_f1_acc_gender.txt")
    write_f1(data=f1_dict, output_path=out_file)


def calc_f1_acc_fg(data, model_name, output_path):
    """
        Takes in a dictionay with topics (5) and each run (1-3) stats.
        Calculates metrics (F1 score, accuracy) for each topic, each run, and
        over all topics, each run, and overall.
        Results are written in a .txt file.
    """
    # for all F1 scores, accuracies
    # {topic: {1: {metrics}, 2: {metrics}, 3: {metrics}, total: {metrics}}}
    f1_dict = {}
    overall_each_run = {1: [[], []], 2: [[], []], 3: [[], []]}
    
    for topic, runs in data.items():
        
        each_run = {1: [], 2: [], 3: []}
        each_run_f1 = {1: 0.0, 2: 0.0, 3: 0.0}
        
        for run_id, stat_info in runs.items():
            # get input & output formality-gender info
            samples = stat_info.get('full', [])

            in_formality = [str(s.get('input')[0]) for s in samples]
            in_gender = [str(s.get('input')[1]) for s in samples]

            out_formality = [str(s.get('output')[0]) for s in samples]
            out_gender = [str(s.get('output')[1]) for s in samples]

            # Create the 9 possible combinations
            input_combined = [f"{f}_{g}" for f, g in zip(in_formality, in_gender)]
            output_combined = [f"{f}_{g}" for f, g in zip(out_formality, out_gender)]

            
            # per topic per run
            each_run[run_id] = (input_combined, output_combined)

            # {class1_f1: xx, class2_f1: xx, ..., macro: xx, accuracy: xx}
            f1_topic_run = get_macro_class_f1_acc(true=input_combined,
                                                  pred=output_combined)
            # add F1 score for each run per topic
            each_run_f1[run_id] = f1_topic_run

        # per topic, each run 
        f1_dict[topic] = each_run_f1
        for run, stats in each_run.items():
            # add to calc each run over all topics
            overall_each_run[run][0].extend(stats[0])
            overall_each_run[run][1].extend(stats[1])
           
        # calculate F1 over all 3 runs per 
        metric_keys = each_run_f1[1].keys() # class_1_f1, etc.
        f1_topic_total = {}
        for key in metric_keys:
            # scores for each metric per run
            scores = [each_run_f1[run][key] for run in [1, 2, 3]]
            f1_topic_total[key] = float(np.mean(scores))
            f1_topic_total[f"std_{key}"] = float(np.std(scores))
        
        f1_dict[topic]['total'] = f1_topic_total
    
    ### --- OVERALL --- ###
    for run, stats in overall_each_run.items():
        in_run = stats[0]
        out_run = stats[1]
        f1_run_overall = get_macro_class_f1_acc(true=in_run,
                                                pred=out_run)
        
        # save F1 scores
        f1_dict.setdefault('overall_runs', {})[run] = f1_run_overall
    
    # Overall each run + total
    metric_keys = f1_dict['overall_runs'][1].keys()
    f1_overall = {}

    for key in metric_keys:
        # scores for each metric per run
        scores = [f1_dict['overall_runs'][run][key] for run in [1, 2, 3]]    
        f1_overall[key] = float(np.mean(scores))
        f1_overall[f"std_{key}"] = float(np.std(scores))
    
    f1_dict['overall_runs']['total'] = f1_overall
    
    # save results
    out_file = os.path.join(output_path, f"{model_name}_f1_acc_FG.txt")
    write_f1(data=f1_dict, output_path=out_file)


def calc_f1_acc(data, model_name, output_path):
    """
        Calculates F1 scores and accuracies, and writes file with results.
        Calculation is for formality, gender, and combined.
    """
    calc_f1_acc_formality(data=data, model_name=model_name, output_path=output_path)  
    calc_f1_acc_gender(data=data, model_name=model_name, output_path=output_path) 
    calc_f1_acc_fg(data=data, model_name=model_name, output_path=output_path)