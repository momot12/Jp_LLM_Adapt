import json
import matplotlib.pyplot as plt
from src.stats.stat_functions import get_formlvl_gender


def get_all_f_g(filepath:str):
    """
        Takes in a filepath to model output and gets formality and gender category for each QA pair.
        Returns a dictionaries with q_id as keys and output formality-gender categories as values.
        E.g. {q_id: [2, 'neutral'], q_id: [formality, gender], ...}
    """
    all_f_g_dict = {}
    with open(filepath, 'r') as f:
        data = json.load(f)[0]  # [{}, {}, ...]
        
        for q_id, info in data.items():
            # e.g. {'input': ['3', 'neutral'], 'output': [2, 'neutral']}
            lvl_gend = get_formlvl_gender(info=info)
            # {q_id: [formality, gender]}
            all_f_g_dict[q_id] = lvl_gend["output"]
    
    return all_f_g_dict


def count_run_agreement(data:dict):
    """
        Counts how consistent the formality and gender is in each run of the output with the same q_id.
        Input:
            data (dict): A dictionary with keys 1, 2, 3 for each run and a list of formality-gender information as values.
                         E.g. {1: {q_id: [2, 'neutral'], q_id: []}, 2: ...}
        
            - Full match: 3/3
            - Majority: 2/3
            - No match: 0/3
        
        Returns a count dict, e.g. {"full_match": 0, "majority_match": 0, "no_match": 0}
    """
    
    results = {"full_match": 0, "majority_match": 0, "no_match": 0}
    
    # get q_ids
    question_ids = data[1].keys()
    
    for q_id in question_ids:
        # get [formality, gender] list for each run
        run1 = data[1][q_id]['output']
        run2 = data[2][q_id]['output']
        run3 = data[3][q_id]['output']
        
        # Convert to tuples so they are hashable/comparable
        combined = [tuple(run1), tuple(run2), tuple(run3)]
        
        # Use a set to find unique responses
        unique_count = len(set(combined))
        
        if unique_count == 1:
            results["full_match"] += 1      
        elif unique_count == 2:
            results["majority_match"] += 1 
        else:
            results["no_match"] += 1      
            
    return results


def plot_pie_topic(data, filename:str):
    """
        Creates a pie chart with percentage.
        Input: {"full_match": 0, "majority_match": 0, "no_match": 0}
    """
    
    # Extract labels and values from the dictionary
    labels = list(data.keys())
    values = list(data.values())
    
    colors = ['#4CAF50', '#FFC107', '#F44336'] 
    
    # Create the pie chart
    # autopct='%1.1f%%' calculates and formats the percentage automatically
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    
    plt.title('Run Consistency Overview')
    plt.axis('equal') # circle 
    
    # Save output
    plt.savefig(f'{filename}.png')
    plt.close()