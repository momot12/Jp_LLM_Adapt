from configs.config import OUTPUT_DIR, HF_CACHE_DIR
import os

print(f"HF_HOME Environment: {os.environ.get('HF_HOME')}")
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_CACHE_DIR, "hub")

from utils.helper_functions import _load_csv, _load_yaml
from models.jp_stable_instruct import Jp_stable_instruct_6b
from huggingface_hub import login
import os
import json


# login HF
login(token=os.environ.get("HF_TOKEN"))

# read yaml for config
yaml_file = "/mount/studenten/projects/takamamo/Jp_LLM_adapt/configs/config.yaml"
config = _load_yaml(config_path=yaml_file)

# load data (csv); datapath from yaml
data = _load_csv(config["files"]["data_dir"], config["files"]["data_file"])
print(f"\nData loaded.")

# load model
model = Jp_stable_instruct_6b(device=config["model"]["device"])
print(f"\nModel loaded.")

# get params from yaml
params = config.get("parameters", {})

# loop over data for generation
print(f"\nGenerating output...")
results = {}
for entry in data:
    # info from data
    q_id = entry["q_id"]
    question = entry["q_id"]["question"]
    topic = entry["q_id"]["topic"]
    formality = entry["q_id"]["formality"]
    gender = entry["q_id"]["gender"]
    
    # generate output
    output = model.generate(
                prompt=question,
                max_new_tokens=params.get("max_new_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                do_sample=params.get("do_sample")
            )
    
    results[q_id] = {
            "question": question,
            "topic": topic,
            "formality": formality,
            "gender": gender,
            "output": output
        }
        
    # save output 
    # TODO: use model name (yaml)
    print(f"\nSaving results...")
    param_info = "_".join(str(v) for v in params.values())
    out_file = "jp_stable_instruct_6b_"+param_info+".json"
    with open(os.path.join(OUTPUT_DIR, out_file), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved. Process completed.")