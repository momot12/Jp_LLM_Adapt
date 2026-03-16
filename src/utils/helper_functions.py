# global helperfunctions
import yaml
import csv
import os
import re
import numpy as np
import logging
from pathlib import Path


def _get_logger(name=__name__):
    """
        Defines a logger, so info can be logged.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        # filename=os.path.join(HF_CACHE_DIR, "run.log"),  # log file
        # filemode="a"  # append
    )
    return logging.getLogger(name)

def _wrap_prompt(text: str, system_prompt="一文で答え。"): #あなたは役立つアシスタントです。
    """
        Convert a string (questions) into the messages format expected by the model.
        NOTE: system_prompt might change depending on model
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

def _check_dir_exist(dir_path):
    """
        Checks if directory exists, and creates if needed.
    """
    logger = _get_logger()
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        logger.info(f"Created directory: {dir_path}")
    else:
        logger.info(f"Directory already exists: {dir_path}")
        
def _load_yaml(config_path="/mount/studenten/projects/takamamo/Jp_LLM_adapt/configs/config.yaml"):
    """
        Load YAML config file and return as a dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def _load_csv(data_dir: str, data_file: str):
    """
        Read CSV file and formats it into a dictionary.
        Output: {q_id: {question: xx, topic: xx, formality: x, gender: xx}}
    """
    csv_path = os.path.join(data_dir, data_file)
    formatted_data = {}
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            q_id = row["q_id"]
            formatted_data[q_id] = {key: row[key] for key in row if key != "q_id"}
        
        return formatted_data


## --- HELPER for STATS + ANALYSIS --- ##
def _get_sent_first_sent(output_clean:str):
    """ Returns all sentences in a list and first sentence of output. It keeps punctuation if there is any.
        Removes any \n or \s. 
    """
    output_clean = re.sub(r'\s+', '', output_clean).strip()
    sentences = [s.strip() for s in re.split(r'(?<=[。！!？?])', output_clean) if s.strip()]
    return sentences, sentences[0]

def get_avg_min_max(numbers_list:list):
    """Get average, min, max, and std of a list with numbers."""
    # just to make sure it is a list
    # numbers_list = [float(x) for x in numbers_list]
    avg = round(sum(numbers_list)/len(numbers_list), 3)
    val_min = round(min(numbers_list), 3)
    val_max = round(max(numbers_list), 3)
    val_std = round(np.std(numbers_list), 3)
    return avg, val_min, val_max, val_std

def combine_num_checkkkk(data_dict:dict):
    """
        Combines lists of numbers of the same key.
        Args:
            dict_lists (dict): A dictionary with dictionaries as values.
                               E.g. {key: {subkey: [num, num]}}
        
        Returns a dictionary with the combined lists.
            E.g. {subkey: [num, num]}
                , where the numbers are combined from all and the same subkeys.
    """
    combined_runs = {}

    for topic, stats in data_dict.items():
        for stat, values in stats.items():
            try:
                combined_runs[stat].extend(values)
            except KeyError:
                print(f"[Error] Stat not available: '{stat}' (found in topic: {topic}).")
            except Exception as e:
                print(f"[Error] Unexpected issue with {topic} -> {stat}: {e}")
            
    return combined_runs


def combine_num_lists(list_of_dicts):
    """
    Combines multiple stat dictionaries into one.
    Input: [{'a': [1, 2], 'b': [5]}, {'a': [3], 'b': [6, 7]}]
    Output: {'a': [1, 2, 3], 'b': [5, 6, 7]}
    """
    if not list_of_dicts:
        return {}
    
    combined = {}
    # Get all unique keys present in any of the dictionaries
    keys = set().union(*(d.keys() for d in list_of_dicts))
    
    for key in keys:
        combined[key] = []
        for d in list_of_dicts:
            if key in d and isinstance(d[key], list):
                combined[key].extend(d[key])
                
    return combined


#### CONFIGS ####

### MODEL PATH ###
# global path to loaded models
HF_CACHE_DIR = "/mount/studenten/projects/takamamo/hf_cache"
# Path("/mount/arbeitsdaten/studenten4/takamamo/hf_models_cache")
# checks if directory exists, and creates if needed
# _check_dir_exist(dir_path=HF_CACHE_DIR)


### OUTPUT PATH ###
OUTPUT_DIR = Path("/mount/studenten/projects/takamamo/outputs/") #NOTE: change later
_check_dir_exist(dir_path=OUTPUT_DIR)

### OUTPUT PATH - STATS (heatmaps, txt, plots) ###
OUTPUT_DIR_STATS = Path("/mount/studenten/projects/takamamo/stats_out")
# checks if directory exists, and creates if needed
_check_dir_exist(dir_path=OUTPUT_DIR_STATS)

### OUTPUT PATH - additional stats ###
OUTPUT_DIR_ADD_STATS = Path("/mount/studenten/projects/takamamo/add_stats_out")
# checks if directory exists, and creates if needed
_check_dir_exist(dir_path=OUTPUT_DIR_ADD_STATS)