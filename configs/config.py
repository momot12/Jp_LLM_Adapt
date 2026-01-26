from pathlib import Path
from utils.helper_functions import _check_dir_exist
import os


### MODEL PATH ###
# global path to loaded models
HF_CACHE_DIR = "/mount/studenten/projects/takamamo/hf_cache"
# Path("/mount/arbeitsdaten/studenten4/takamamo/hf_models_cache")
# checks if directory exists, and creates if needed
# _check_dir_exist(dir_path=HF_CACHE_DIR)



# NOTE (TBD) :add to yaml if I want to create separate directories for each model
### OUTPUT PATH ###
OUTPUT_DIR = Path("/mount/studenten/projects/takamamo/outputs")
# checks if directory exists, and creates if needed
_check_dir_exist(dir_path=OUTPUT_DIR)