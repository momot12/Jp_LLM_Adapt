import yaml
import argparse
from scripts.run_statistics import run_stats


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subdir")
    args = parser.parse_args()

    # Load YAML config once
    with open(f"./configs/{args.config}.yaml") as f:
        config = yaml.safe_load(f)

    # get model name
    # model_name = config["model"]["name"]
    # get threshold for BLEU analysis; default is BLEU=0.8
    bleu_threshold = config.get('parameters', {}).get('threshold_bleu', 0.8)

    subdir = args.subdir #or model_name

    print(f"[INFO] Running stats with BLEU threshold of {bleu_threshold}\n") 
    run_stats(
        sub_dir=subdir,
        threshold_bleu=float(bleu_threshold)
    )