# from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from models.jp_stable_instruct import Jp_stable_instruct_6b
from models.ml_qwen_instruct_4b import ML_Qwen_Instruct_4_b
import logging


# 1. Suppress HTTP Request logs (the GET/HEAD lines)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 2. Suppress Transformers library info (the tied weights warning)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# 3. Suppress the FutureWarning about HF_HOME (if it persists)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"



HF_CACHE_DIR = "/mount/studenten/projects/takamamo/hf_cache"
# model_name = "stabilityai/japanese-stablelm-2-instruct-1_6b"


# models = [Jp_stable_instruct_6b(device="cuda:1"), 
#           ML_Qwen_Instruct_1_7_b(device="cuda:1")]

models =  ML_Qwen_Instruct_4_b(device="cuda:0")
prompt = "こんにちは、元気ですか？"


output = models.generate(
            prompt=prompt,
            sys_prompt='一文で答え'
        )

print(f"Model {models.model_name} says:\n{output}")