import os
import json
import logging
from datetime import datetime
from utils.helper_functions import (_load_csv, 
                                    _load_yaml, 
                                    _get_logger, 
                                    OUTPUT_DIR,
                                    )
from models.jp_stable_instruct import JP_Stable_Instruct
from models.ml_qwen_instruct import ML_Qwen_Instruct_1_7_b
from models.jp_instruct_1_6 import Jp_stable_instruct_6b
from models.ml_qwen_instruct_4b import ML_Qwen_Instruct_4_b

# suppress HTTP Request logs (the GET/HEAD lines)/transformer libs
logging.getLogger("httpx").setLevel(logging.WARNING)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # surpress warning about HF_HOME


def run_model(config_file:str="config"):

    # create logger
    logger = _get_logger()

    # checking for right HF_HOME
    logger.info(f"Checking right -> HF_HOME Environment: {os.environ.get('HF_HOME')}\n")

    # read yaml for config
    yaml_file = f"/mount/studenten/projects/takamamo/Jp_LLM_adapt/configs/{config_file}.yaml"
    config = _load_yaml(config_path=yaml_file)

    # load data (csv); datapath from yaml
    data = _load_csv(config["files"]["data_dir"], config["files"]["data_file"])
    logger.info(f"File: {os.path.join(config['files']['data_dir'], config['files']['data_file'])}.")
    logger.info("Data loaded.\n")

    # device 
    device = config["model"]["device"]

    ## -- MODEL SELECTION -- ##
    selected_model = config["model"]["name"]
    logger.info(f"Loading model: {selected_model}.")
    # load model
    if selected_model == 'qwen_1_7B':
        model = ML_Qwen_Instruct_1_7_b(device=device)
    elif selected_model == 'qwen_4B':
        model = ML_Qwen_Instruct_4_b(device=device)
    elif selected_model == 'jp_stable_7B': #in ['jp_stable_1_6B', 'jp_stable_7B']:
        model = JP_Stable_Instruct(model_size=config["model"]["model_size"], device=device)
    elif selected_model == 'jp_stable_1_6B':
        model = Jp_stable_instruct_6b(device=device)
    else:
        raise ValueError(f"Unknown model: {selected_model}.")
    print() # formatting purposes
    logger.info("Model loaded.")

    # create new output directory
    new_folder = f"{selected_model}_{datetime.now().strftime('%b%d')}"
    OUTPUT_DIR_NEW = os.path.join(OUTPUT_DIR, new_folder)
    # add version tag if already exists to prevent overwriting
    counter = 1
    temp_path = OUTPUT_DIR_NEW
    while os.path.exists(temp_path):
        temp_path = f"{OUTPUT_DIR_NEW}_v{counter}"
        counter += 1
    out_path = temp_path
    os.makedirs(out_path, exist_ok=True)
    logger.info(f"New output directory: {OUTPUT_DIR_NEW}.")

    # get params from yaml
    params = config.get("parameters", {})
    sys_prompt = params.pop("system_prompt", "")

    # loop over data for generation
    logger.info("Generating output...")
    result = {}
    results = []
    for entry in data:
        # info from data
        q_id = entry
        question = data[q_id]["question"]
        topic = data[q_id]["topic"]
        formality = data[q_id]["formality"]
        gender = data[q_id]["gender"]
        
        # generate output
        output, only_out = model.generate(
                            prompt=question,
                            sys_prompt=sys_prompt,   #params.get("system_prompt")
                            **params
                        )
        
        result[q_id] = {
                "question": question,
                "topic": topic,
                "formality": formality,
                "gender": gender,
                "output": output,
                "only_out": only_out,
            }
            
        results.append(result)

    # save output 
    logger.info("Saving results...")
    model_name_save = model.model_name.replace("/", "_")
    param_info = "_".join(str(v) for k,v in sorted(params.items()))
    filename_end = config["files"]["out_file_name"]
    out_file = f"{model_name_save}_{param_info}_{filename_end}.json"
    with open(os.path.join(OUTPUT_DIR_NEW, out_file), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved as {out_file}.")
    logger.info("Process completed.")