# global helperfunctions
import yaml
import csv
import os
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
    

# print(_load_csv("/mount/arbeitsdaten/studenten4/takamamo/data/", "sample_data.csv"))


#### CONFIGS ####

### MODEL PATH ###
# global path to loaded models
HF_CACHE_DIR = "/mount/studenten/projects/takamamo/hf_cache"
# Path("/mount/arbeitsdaten/studenten4/takamamo/hf_models_cache")
# checks if directory exists, and creates if needed
# _check_dir_exist(dir_path=HF_CACHE_DIR)

# NOTE (TBD) :add to yaml if I want to create separate directories for each model
### OUTPUT PATH ###
OUTPUT_DIR = Path("/mount/studenten/projects/takamamo/outputs/") #NOTE: change later
_check_dir_exist(dir_path=OUTPUT_DIR)

### OUTPUT PATH - STATS###
OUTPUT_DIR_STATS = Path("/mount/studenten/projects/takamamo/maps_out")
# checks if directory exists, and creates if needed
_check_dir_exist(dir_path=OUTPUT_DIR_STATS)