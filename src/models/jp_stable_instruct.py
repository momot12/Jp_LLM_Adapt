import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from utils.helper_functions import _wrap_prompt, HF_CACHE_DIR


class JP_Stable_Instruct:
    
    def __init__(self, 
                 model_size = '7b', #'1.6b'
                #  model_name = "stabilityai/japanese-stablelm-2-instruct-1_6b", 
                 device=None,
                 max_new_tokens = 128,   # set as default, override later
                 temperature = 0.99,     # set as default, override later
                 top_p = 0.95,           # set as default, override later
                 do_sample = True,       # set as default, override later
                ):
        
        model_options = {
                "1.6b": "stabilityai/japanese-stablelm-2-instruct-1_6b",
                "7b": "stabilityai/japanese-stablelm-instruct-beta-7b"
            }
        
        # select model
        self.model_size = model_size
        self.model_name = model_options[model_size]
        
        # default generation settings -> def generate()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
        )

        # ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # config - needs pad_token for 1.6B
        config = AutoConfig.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            cache_dir=HF_CACHE_DIR
        )
        # Manually force the pad_token_id into the config object
        config.pad_token_id = self.tokenizer.pad_token_id
        
        # params for self.model
        load_params = {
            "config": config,
            "trust_remote_code": True,
            "cache_dir": HF_CACHE_DIR,
            # "pad_token_id": self.tokenizer.pad_token_id
        }
        
        if self.model_size == '7b':
            # remove pad_token_id for 7B -> only needed for 1.6B
            load_params.pop("config")
            # formal bnb because pip install bitsandbytes not recognized despite installed
            # AVOID: TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            load_params.update({
                "quantization_config": bnb_config,
                "device_map": "auto",        # Automatically picks best GPU
                # low_cpu_mem_usage=True,
            })
        else:
            load_params.update({
                "torch_dtype": torch.float32,
                "attn_implementation": "eager",     # forces standard math (avoids version bugs)
            })
            
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_params
        )
        # set padding
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # set device and put model to device
        if self.model_size == '7b':
            self.device = self.model.device     # saves which device chosen by "auto" -> avoid crash
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        
        
    def generate(self, 
                 prompt,  # either str or list[dict]
                #  messages: list[dict],  # [{"role": "...", "content": "..."}]
                 sys_prompt:str = "",
                 **kwargs
                ):
        
        # set params; take init params if not specifically set here
        gen_config = {
                "max_new_tokens": kwargs.get('max_new_tokens', self.max_new_tokens),
                "temperature":    kwargs.get('temperature', self.temperature),
                "top_p":          kwargs.get('top_p', self.top_p),
                "do_sample":      kwargs.get('do_sample', self.do_sample),
                "pad_token_id":   self.tokenizer.pad_token_id,
                # "use_cache":      True,
                # "return_legacy_cache": True if self.model_size == "1.6b" else None,
            }
        # to avoid version issues of DynamicCache
        if self.model_size == "1.6b":
            gen_config["use_cache"] = False
        
        # format string prompt into expected message style
        if isinstance(prompt, str):
            if self.model_size == "7b":
                messages = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<<SYS>>\n\n{prompt} [/INST] "
                inputs = self.tokenizer(
                            messages,
                            add_special_tokens=False,
                            return_tensors="pt"
                        ).to(self.device)
            else:
            # [{"role": "...", "content": "..."}]
                messages = _wrap_prompt(prompt, system_prompt=sys_prompt) 
                # convert into model input token (given by hf) -> torch.Tensor
                inputs = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(self.device)
        # already in correct format
        else:
            messages = prompt
            inputs = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(self.device)
            
        # generate tokens
        tokens = self.model.generate(
                    **inputs,     # NOTE: **inputs for attention mask?
                    **gen_config
                )
        
        # convert output to readable
        output = self.tokenizer.decode(tokens[0], skip_special_tokens=False) # set True to remove <s>, </s> tokens
        only_out = self.tokenizer.decode(tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        return output, only_out
    