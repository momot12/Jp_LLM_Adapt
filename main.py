# Step 1: Data Preprocessing
## - read files (raw data)
## - lingusitic variatios
## - save

# Step 2: Run models
## - run each model with specific params
## - save output

# Step 3: Output Processing
## - determine F, G
## - get basic stats
## - get psycholing stats
## - PLOT stats
## - save stats

# Step 4: Additional Analysis
## - Calc Convergence (conv(t)) or sth similar
## - Correlations between e.g. skippable vs all particles

import os
import yaml
import argparse
from datetime import datetime
from scripts.run_data_preprocess import run_data_preprocessing
from scripts.run_models import run_model
from scripts.run_statistics import run_stats


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None) 
    args = parser.parse_args()
    
    # Load YAML config once
    yaml_path = f"./configs/{args.config}.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # set up which processes to run
    pipe = config.get('pipeline', {})
    # get base paths
    paths = config.get('paths', {})
    
    # ---------------------------------- 
    ## Step 1: Data Preprocessing
    if pipe.get('run_data_prep'):
        data_in_path = paths.get('raw_data_dir')
        data_out_path = paths.get('data_dir')
        run_data_preprocessing(data_in=data_in_path, data_out=data_out_path)

    
    ## Step 2: Run Models
    dir_saved = None
    if pipe.get('run_model'):
        # override if the flags are used
        if args.data_file: config['files']['data_file'] = args.data_file
        if args.out_file: config['files']['out_file_name'] = args.out_file
        
        # write in a temporary yaml file -> to run all combinations    
        temp_name = "temp_config"
        temp_path = f"./configs/{temp_name}.yaml"
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)
            
        dir_saved = run_model(config_file=temp_name)
    
    
    ## Step 3: Run Stats
    if pipe.get('run_stats'):
        # fallback on existing runs 
        if not dir_saved:
            model_name = config['model']['name']
            dir_saved = f"{model_name}"
            
            # sub_dir = "qwen_1_7B_Mar12"
        print(f"[INFO] Running stats on: {dir_saved}")
        run_stats(sub_dir=dir_saved)
    
    
    ## Step 4: Run Additional Analysis
    # TODO: add this + add threshold in YAML
    if pipe.get('run_analysis'):
        # run_add_analysis()
        pass
        
    
    
    # TODO: maybe delete temp yaml at the end?
    # try:
    #         dir_saved = run_model(config_file="temp_config")
    #     finally:
    #         # This runs NO MATTER WHAT (even if run_model crashes)
    #         if os.path.exists(temp_path):
    #             os.remove(temp_path)
    #             # print("Cleaned up temporary config.")