# from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Add that root to sys.path so 'src' becomes importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ml_qwen_instruct import ML_Qwen_Instruct_1_7_b
import logging


# 1. Suppress HTTP Request logs (the GET/HEAD lines)
logging.getLogger("httpx").setLevel(logging.WARNING)
# 2. Suppress Transformers library info (the tied weights warning)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
# 3. Suppress the FutureWarning about HF_HOME (if it persists)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"



HF_CACHE_DIR = "/mount/studenten/projects/takamamo/hf_cache"

models =  ML_Qwen_Instruct_1_7_b(device="cuda:1")
prompt = "あなた、最近、夜はよく眠れるかしら？"


output, only_out = models.generate(
            prompt=prompt,
            sys_prompt="一文のみでの回答。"
        )

print(f"Model {models.model_name} says:\n{only_out}")