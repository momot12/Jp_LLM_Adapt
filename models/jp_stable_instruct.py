import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import HF_CACHE_DIR
from utils.helper_functions import _wrap_prompt


class Jp_stable_instruct_6b:
    
    def __init__(self, 
                 model_name = "stabilityai/japanese-stablelm-2-instruct-1_6b", 
                 device=None,
                 max_new_tokens = 128,   # set as default, override later
                 temperature = 0.99,     # set as default, override later
                 top_p = 0.95,           # set as default, override later
                 do_sample = True,       # set as default, override later
                ):
        
        self.model_name = model_name
        self.device = device
        
        # default generation settings -> def generate()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
        )
    
        # set manually
        if device is not None:
            self.model.to(device) 
        
        
    def generate(self, 
                 prompt,  # either str or list[dict]
                #  messages: list[dict],  # [{"role": "...", "content": "..."}]
                 max_new_tokens: int | None = None,
                 temperature: float | None = None,
                 top_p: float | None = None,
                 do_sample: bool | None = None,
                ):
        
        # set params; take init params if not specifically set here
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        # format string prompt into expected message style
        if isinstance(prompt, str):
            # [{"role": "...", "content": "..."}]
            messages = _wrap_prompt(prompt) 
        else:
            # already in list[dict] format
            messages = prompt   
        
        # convert into model input token (given by hf)
        inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.device) # token on same device as model
        
        # generate tokens
        tokens = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
        
        # convert output to readable
        output = self.tokenizer.decode(tokens[0], skip_special_tokens=False) # set True to remove <s>, </s> tokens
        
        return output
    