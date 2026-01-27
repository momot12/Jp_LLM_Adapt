import os
import json
from utils.helper_functions import (_load_csv, 
                                    _load_yaml, 
                                    _get_logger, 
                                    OUTPUT_DIR, 
                                    HF_CACHE_DIR)
from models.jp_stable_instruct import Jp_stable_instruct_6b



# UNUSED: login HF
# from huggingface_hub import login
# login(token=os.environ.get("HF_TOKEN"))

# create logger
logger = _get_logger()

# checking for right HF_HOME
logger.info("Checking right -> HF_HOME Environment: {os.environ.get('HF_HOME')}\n")

# read yaml for config
yaml_file = "/mount/studenten/projects/takamamo/Jp_LLM_adapt/configs/config.yaml"
config = _load_yaml(config_path=yaml_file)

# load data (csv); datapath from yaml
data = _load_csv(config["files"]["data_dir"], config["files"]["data_file"])
logger.info("Data loaded.\n")

# load model
model = Jp_stable_instruct_6b(device=config["model"]["device"])
print()
logger.info("Model loaded.")

# get params from yaml
params = config.get("parameters", {})

# loop over data for generation
logger.info("Generating output...")
results = {}
for entry in data:
    # info from data
    q_id = entry
    question = data[q_id]["question"]
    topic = data[q_id]["topic"]
    formality = data[q_id]["formality"]
    gender = data[q_id]["gender"]
    
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
logger.info("Saving results...")
model_name_save = model.model_name.replace("/", "_")
param_info = "_".join(str(v) for k,v in sorted(params.items()))
out_file = f"{model_name_save}_{param_info}.json"
with open(os.path.join(OUTPUT_DIR, out_file), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logger.info("Saved. Process completed.")