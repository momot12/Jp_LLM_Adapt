import os

HF_CACHE_DIR = "/mount/studenten/projects/takamamo/hf_cache"

print(f"HF_HOME Environment: {os.environ.get('HF_HOME')}")
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_CACHE_DIR, "hub")


import tiktoken
from transformers import AutoTokenizer
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN"))

# print(f"Tiktoken version: {tiktoken.__version__}")
tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/japanese-stablelm-2-instruct-1_6b",
    trust_remote_code=True,
    cache_dir=HF_CACHE_DIR,
    # use_fast=True,
    # force_download=True
)

test_text = "こんにちは、元気ですか？"
tokens = tokenizer.tokenize(test_text)
print(f"Text: {test_text}")
print(f"Tokens: {tokens}")
