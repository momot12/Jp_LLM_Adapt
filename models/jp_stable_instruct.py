import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.helper_functions import _wrap_prompt, HF_CACHE_DIR


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
        
        # default generation settings -> def generate()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
        )

        # ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            # device_map="auto",        # only set if device is not set manually
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
        )

        # set device and put model to device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        
    def generate(self, 
                 prompt,  # either str or list[dict]
                #  messages: list[dict],  # [{"role": "...", "content": "..."}]
                 max_new_tokens: int | None = None,
                 temperature: float | None = None,
                 top_p: float | None = None,
                 do_sample: bool | None = None,
                 sys_prompt:str = ""
                ):
        
        # set params; take init params if not specifically set here
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        # format string prompt into expected message style
        if isinstance(prompt, str):
            # [{"role": "...", "content": "..."}]
            messages = _wrap_prompt(prompt, system_prompt=sys_prompt) 
        else:
            # already in list[dict] format
            messages = prompt   
        
        # convert into model input token (given by hf) -> torch.Tensor
        inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.device)  # token on same device as model
        
        # generate tokens
        tokens = self.model.generate(
                    inputs,     # NOTE: **inputs for attention mask
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id, 
                )
        
        # convert output to readable
        output = self.tokenizer.decode(tokens[0], skip_special_tokens=False) # set True to remove <s>, </s> tokens
        
        return output
    