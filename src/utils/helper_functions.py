# global helperfunctions
import yaml
import csv
import os
import re
import numpy as np
import subprocess
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

def _get_sent_first_sent(output_clean:str):
    """ Returns all sentences in a list and first sentence of output. It keeps punctuation if there is any.
        Removes any \n or \s. 
    """
    output_clean = re.sub(r'\s+', '', output_clean).strip()
    sentences = [s.strip() for s in re.split(r'(?<=[。！!？?])', output_clean) if s.strip()]
    return sentences, sentences[0]


def get_avg_min_max(numbers_list:list):
    # just to make sure it is a list
    # numbers_list = [float(x) for x in numbers_list]
    avg = round(sum(numbers_list)/len(numbers_list), 3)
    val_min = round(min(numbers_list), 3)
    val_max = round(max(numbers_list), 3)
    val_std = round(np.std(numbers_list), 3)
    return avg, val_min, val_max, val_std


# TODO: revise and check if ok !
def combine_run_files(path_runs_dir:str):
    
    # umbrella dirctory, e.g. /mount/studenten/projects/takamamo/outputs/qwen_1_7B_Mar13
    root = Path(path_runs_dir)
    # sub dir of roots => topics
    target_subdirs = ['career', 'daily', 'knowledge', 'life', 'simple']
    
    output_dir = root / "three_runs"
    output_dir.mkdir(exist_ok=True)

    for subdir_name in target_subdirs:
        subdir = root / subdir_name
        
        if not subdir.exists():
            print(f"Warning: {subdir_name} not found. Skipping.")
            continue

        # 2. Identify the three files based on their suffix
        # Adjust glob patterns if your naming convention differs slightly
        file1 = list(subdir.glob("*_1.json"))[0]
        file2 = list(subdir.glob("*_2.json"))[0]
        file3 = list(subdir.glob("*_3.json"))[0]

        # 3. Construct the jq command
        # We use map_values or walk to find q_ids and append "a" or "b" 
        # based on your specific requirements.
        jq_filter = (
            '([.[0][]] | map(.q_id |= .)) + ' # Run 1: keep as is
            '([.[1][]] | map(.q_id |= sub("q(?<num>\\\\d+)"; "q\\(.num)a"))) + ' # Run 2: q3 -> q3a
            '([.[2][]] | map(.q_id |= sub("q(?<num>\\\\d+)"; "q\\(.num)b"))) | ' # Run 3: q3 -> q3b
            '[add]' # Wrap in final array if needed by your schema
        )

        output_filename = f"run3x_{subdir_name}.json"
        output_path = output_dir / output_filename

        # 4. Execute the bash command
        # We pass the files into jq as a positional array using --slurp (-s)
        cmd = f"jq -s '{jq_filter}' {file1} {file2} {file3} > {output_path}"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully created: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {subdir_name}: {e}")

# Usage
# combine_run_files('./my_data_folder')


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
OUTPUT_DIR_STATS = Path("/mount/studenten/projects/takamamo/maps_out")
# checks if directory exists, and creates if needed
_check_dir_exist(dir_path=OUTPUT_DIR_STATS)

### OUTPUT PATH - additional stats ###
OUTPUT_DIR_ADD_STATS = Path("/mount/studenten/projects/takamamo/add_stats_out")
# checks if directory exists, and creates if needed
_check_dir_exist(dir_path=OUTPUT_DIR_ADD_STATS)