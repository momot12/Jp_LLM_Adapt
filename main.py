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

import yaml
import argparse
from scripts.run_data_preprocess import run_data_preprocessing
from scripts.run_models import run_model
from scripts.run_statistics import run_stats


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",           # argument name
        type=str,             # expected type
        required=True,        # mandatory
        help="Config file"
    )
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None) 
    args = parser.parse_args()
    
    # Load the YAML
    yaml_path = f"/mount/studenten/projects/takamamo/Jp_LLM_adapt/configs/{args.config}.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # OVERRIDE the YAML values if the flags are used
    if args.data_file:
        config['files']['data_file'] = args.data_file
    if args.out_file:
        config['files']['out_file_name'] = args.out_file
        
    temp_name = "temp_config"
    temp_path = f"/mount/studenten/projects/takamamo/Jp_LLM_adapt/configs/{temp_name}.yaml"
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)
    #TODO: collect all params from yaml, so only need to set params in yaml
    
    # Step 1:
    # data_in_path = "/mount/arbeitsdaten/studenten4/takamamo/data/raw_qs_samples/"
    # data_out_path = "/mount/arbeitsdaten/studenten4/takamamo/data"
    # run_data_preprocessing(data_in=data_in_path, data_out=data_out_path)
    
    # Step 2:
    config_yaml = ["config", "config_stable", "config_qwen"]
    dir_saved = run_model(config_file=temp_name)
    
    print(dir_saved)
    
    # Step 3:
    # subdir = dir_saved
    # sub_dir = "qwen_1_7B_Mar12"
    # run_stats(sub_dir=subdir)
    
    # Step 4:
    # run_add_analysis()
    # pass