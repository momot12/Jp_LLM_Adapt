import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from src.utils.helper_functions import _wrap_prompt, HF_CACHE_DIR


class ML_Qwen_Instruct_4_b:
    
    def __init__(self,
                 model_name = "Qwen/Qwen3-4B",
                 device = None,
                 max_new_tokens=128,
                 temperature=0.7,  
                 top_p=0.8,
                 do_sample=True,
                 **kwargs, 	# all other params from yaml
                ):

        self.model_name = model_name
        
        # default generation settings -> def generate()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        self.extra_params = kwargs

        # load tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
        )
        
        # for quantization cuz of GPU
        config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
                
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,            
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            device_map=None, 	# None if manually
            quantization_config=config,
        )
  
        # ensure pad token is set --> usually Qwen handles
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
  
        # set device and put model to device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def generate(self, 
                prompt,  # str or list[dict]
                sys_prompt: str = "",
                **kwargs,
                ):
    
        # set params; take init params if not specifically set here
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        do_sample = kwargs.get("do_sample", self.do_sample)
     
        # format string prompt into expected message style
        if isinstance(prompt, str):
              # [{"role": "...", "content": "..."}]
            messages = _wrap_prompt(prompt, system_prompt=sys_prompt)
            # print(f"\n\n{messages}\n\n")
        else:
               # already in list[dict] format
            messages = prompt   
        
        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k == "enable_thinking"}
          # convert into model input token (given by hf) -> torch.Tensor
        inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict = True, 
                    # **tokenizer_kwargs,
                    enable_thinking = False,
                ).to(self.device)
        
        # generate tokens
        tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id, 
                )
        
        # convert output to readable
        output = self.tokenizer.decode(tokens[0], skip_special_tokens=False) # set True to remove <s>, </s> tokens
  
        only_out = self.tokenizer.decode(tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        # print(f"\n\nOutput only:\n{only}\n\n")
        
        return output, only_out