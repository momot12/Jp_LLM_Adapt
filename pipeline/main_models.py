import yaml
import argparse
from scripts.run_models import run_model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_file")
    parser.add_argument("--out_file")
    args = parser.parse_args()

    # Load YAML config once
    with open(f"./configs/{args.config}.yaml") as f:
        config = yaml.safe_load(f)
        
    # get model name
    model_name = config['model']['name']

    # e.g. conv_qs_20_career.csv
    if args.data_file: config["files"]["data_file"] = args.data_file
    # number of run, e.g. 3
    if args.out_file: config["files"]["out_file_name"] = args.out_file

    # write in a temporary yaml file -> to run all combinations   
    temp_name = f"temp_config_{model_name}"

    # write temp yaml file with dynamic arguments (in bash script)
    with open(f"./configs/{temp_name}.yaml", "w") as f:
        yaml.dump(config, f)

    # generate model answers
    run_model(config_file=temp_name)