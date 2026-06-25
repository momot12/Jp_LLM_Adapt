import yaml
import argparse
from scripts.run_data_preprocess import run_data_preprocessing


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # Load YAML config once
    with open(f"./configs/{args.config}.yaml") as f:
        config = yaml.safe_load(f)

    # get base paths
    paths = config.get('paths', {})

    # run data preprocessing: 9 variations of the original questions
    run_data_preprocessing(
        data_in=paths["raw_data_dir"],
        data_out=paths["data_dir"]
    )