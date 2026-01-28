from transformers import AutoTokenizer, AutoModelForCausalLM
from jp_stable_instruct import Jp_stable_instruct_6b

HF_CACHE_DIR = "/mount/studenten/projects/takamamo/hf_cache"
model_name = "stabilityai/japanese-stablelm-2-instruct-1_6b"

# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     cache_dir=HF_CACHE_DIR,
#     # use_fast=True,
#     # force_download=True
# )

model = Jp_stable_instruct_6b()

prompt = "こんにちは、元気ですか？"


output = model.generate(
            prompt=prompt,
        )