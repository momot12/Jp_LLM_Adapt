import os
import json
import logging
import warnings
from datetime import datetime
from src.utils.helper_functions import (_load_csv, 
                                    _load_yaml, 
                                    _get_logger, 
                                    OUTPUT_DIR,
                                    )
from src.utils.validity_output import check_language
from src.utils.helper_functions import _check_dir_exist
from src.models.jp_stable_instruct import JP_Stable_Instruct
from src.models.ml_qwen_instruct import ML_Qwen_Instruct_1_7_b
from src.models.jp_instruct_1_6 import Jp_stable_instruct_6b
from src.models.ml_qwen_instruct_4b import ML_Qwen_Instruct_4_b

# --- SILENCE WARNINGS --- #
# suppress FutureWarnings 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Specified kernel cache directory could not be created")
# suppress matplotlib font warnings
logging.getLogger('matplotlib.font_manager').disabled = True
# suppress HTTP Request logs (the GET/HEAD lines)/transformer libs
logging.getLogger("httpx").setLevel(logging.WARNING)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # surpress warning about HF_HOME
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# ------------------------- #


def run_model(config_file:str="config"):

    # create logger
    logger = _get_logger()

    # checking for right HF_HOME
    logger.info(f"Checking right -> HF_HOME Environment: {os.environ.get('HF_HOME')}\n")

    # model sizes
    SIZE_MAP = {
        "jp_stable_1_6B": "1.6b",
        "jp_stable_7B": "7b",
        "qwen_1_7B": "1.7b",
        "qwen_4B": "4b",
    }

    # read yaml for config
    yaml_file = f"/mount/studenten/projects/takamamo/Jp_LLM_adapt/configs/{config_file}.yaml"
    config = _load_yaml(config_path=yaml_file)

    # load data (csv); datapath from yaml
    data = _load_csv(config["paths"]["data_dir"], config["files"]["data_file"])
    logger.info(f"File: {os.path.join(config['paths']['data_dir'], config['files']['data_file'])}.")
    logger.info("Data loaded.\n")

    # device 
    device = config["model"]["device"]

    ## -- MODEL SELECTION -- ##
    selected_model = config["model"]["name"]
    model_size = SIZE_MAP.get(selected_model, "7b")
    logger.info(f"Loading model: {selected_model}.")
    # load model
    if selected_model == 'qwen_1_7B':
        model = ML_Qwen_Instruct_1_7_b(device=device)
    elif selected_model == 'qwen_4B':
        model = ML_Qwen_Instruct_4_b(device=device)
    elif selected_model == 'jp_stable_7B':
        model = JP_Stable_Instruct(model_size=model_size, device=device)
    elif selected_model == 'jp_stable_1_6B':
        model = Jp_stable_instruct_6b(device=device)
    else:
        raise ValueError(f"Unknown model: {selected_model}.")
    print() # formatting purposes
    logger.info("Model loaded.")

    # create new output directory
    new_folder = f"{selected_model}_{datetime.now().strftime('%b%d')}"
    OUTPUT_DIR_NEW = os.path.join(OUTPUT_DIR, new_folder)
    logger.info(f"New output directory: {OUTPUT_DIR_NEW}.")

    # get params from yaml
    params = config.get("parameters", {})
    sys_prompt = params.pop("system_prompt", "")

    # loop over data for generation
    print() # formatting purposes
    logger.info("Generating output...")
    results = []
    for entry in data:
        result = {}
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
        
        # Check language first generate
        lang_jp, lang = check_language(out_sent=only_out)
        
        result[q_id] = {
                "question": question,
                "topic": topic,
                "formality": formality,
                "gender": gender,
                "output": output,
                "only_out": only_out,
            }
        
        # repeat generating until it's Japanese
        while lang_jp == False:
            logger.info(f"NOT JAPANESE -> {lang}. Retrying with langauge restriction prompt.")
            print(result)
            # generate answer for Q again with language restriction
            output, only_out = model.generate(
                            prompt=question,
                            sys_prompt="日本語で一文のみでの回答。",   #params.get("system_prompt")
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
            lang_jp, lang = check_language(out_sent=only_out)
            
        # only append Japanese output    
        results.append(result)

    # save output 
    print() # formatting purposes
    logger.info("Saving results...")
    model_name_save = model.model_name.replace("/", "_")
    param_info = "_".join(str(v) for k,v in sorted(params.items()))
    # create directory for each topic
    _check_dir_exist(dir_path=os.path.join(OUTPUT_DIR_NEW, topic))
    filename_end = config["files"]["out_file_name"]
    out_name = f"{model_name_save}_{param_info}_{topic}_{filename_end}.json"
    out_file = os.path.join(topic, out_name)
    with open(os.path.join(OUTPUT_DIR_NEW, out_file), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # add first run to a separate directory
    if "1" in filename_end[-1]:
        single_run_dir = os.path.join(OUTPUT_DIR_NEW, "single_run")
        _check_dir_exist(dir_path=single_run_dir)
        with open(os.path.join(single_run_dir, out_name), 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print() # formatting purposes
    logger.info(f"Saved as {out_file}.")
    logger.info("Process completed.")
    
    # for stats --> Arg: sub_dir
    return OUTPUT_DIR_NEW