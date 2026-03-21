import os
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
                        f.write(f"\t\t{k.capitalize()}: {v:.4f}\n")
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
                f.write(f"\t\t{k.capitalize()}: {v:.4f}\n")


def get_macro_class_f1_acc(true:list, pred: list):
    """
        Takes in two lists with numbers - true and prediction, and calculates 
        F1 macro score and for individual classes.
        Returns a dict with F1 scores (classes and macro) and accuracy.
    """
    f1_macro = f1_score(y_true=true, y_pred=pred, average='macro')
    f1_perclass = f1_score(y_true=true, y_pred=pred, average=None)
    acc = accuracy_score(y_true=true, y_pred=pred)
    
    metrics = {
        "class_1_f1": float(round(f1_perclass[0], 4)),
        "class_2_f1": float(round(f1_perclass[1], 4)),
        "class_3_f1": float(round(f1_perclass[2], 4)),
        "macro_f1":   float(round(f1_macro, 4)),
        "accuracy":   float(round(acc, 4))
        }
    
    return metrics
    

def calc_f1_acc(data, model_name, output_path):
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
        topic_3runs_input, topic_3runs_output = [],[]
        for run, stats in each_run.items():
            topic_3runs_input.extend(stats[0])
            topic_3runs_output.extend(stats[1])
            
            # add to calc each run over all topics
            overall_each_run[run][0].extend(stats[0])
            overall_each_run[run][1].extend(stats[1])
            
            
        # calculate F1 over all 3 runs per topic
        f1_topic_total = get_macro_class_f1_acc(true=topic_3runs_input,
                                                pred=topic_3runs_output)
        
        f1_dict[topic]['total'] = f1_topic_total
    
    ### --- OVERALL --- ###
    all_input, all_output = [], []
    for run, stats in overall_each_run.items():
        in_run = stats[0]
        out_run = stats[1]
        f1_run_overall = get_macro_class_f1_acc(true=in_run,
                                                pred=out_run)
        # collect all for overall 
        all_input.extend(in_run)
        all_output.extend(out_run)
        
        # save F1 scores
        f1_dict.setdefault('overall_runs', {})[run] = f1_run_overall
    
    f1_overall = get_macro_class_f1_acc(true=all_input,
                                        pred=all_output)

    f1_dict['overall_runs']['total'] = f1_overall
    
    # save results
    out_file = os.path.join(output_path, f"{model_name}_f1_acc.txt")
    write_f1(data=f1_dict, output_path=out_file)       