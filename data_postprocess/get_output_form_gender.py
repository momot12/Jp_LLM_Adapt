# read output files (done)
## Func1:
    # get input formality, gender (done)
    # measure output formaility TODO
## Func2: (done)
    # measure len(in) vs len(out) --> num chars
    # num sent out
## in main or func3:
    # for each formality: avg output length 
    # for matching formality: avg output length 

import sys
import os
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_functions import OUTPUT_DIR


def measure_formlvl_gender():
    # NOTE: put in class somewehere else?
    pass

def get_formlvl_gender(info:dict):
    """
        Get formality levels and gender from input and output, and format.
        Return a dict {"input": [formality, gender], "output": [formality, gender]}.
    """
    out_dict = {}
    for item in info:
        out_dict["input"] = [info["formality"], info["gender"]]
        # TODO: measure_formlvl_gender()
        out_dict["output"] = []
    
    return out_dict

def count_output_len(info:dict):
    """
        Counts #output sentences, and #chars of input vs output, including punctuation.
        Returns a dictionary with len(input), len(output), and #output sentences.
        
        NOTE: maybe exclude punctuation in calculation?
    """
    len_info = {}
    for item in info:
        match = re.search(r"<\|assistant\|>\s*(.*?)<\|endoftext\|>",
                            info["output"], re.DOTALL)
        output_clean = match.group(1).strip() if match else ""
        input = info["question"]
        sentences = [s for s in re.split(r'[。！!？?]', output_clean) if s.strip()]
        
        len_info = {"input_len": len(input),
                    "output_len": len(output_clean),
                    "num_sent_out": len(sentences),
                    }
        
    return len_info

def get_avg_min_max(numbers_list:list):
    print(f"Avg: {sum(numbers_list)/len(numbers_list)}")
    print(f"Min: {min(numbers_list)}")
    print(f"Max: {max(numbers_list)}")

# ------- MAIN ------
# specific for sample data
files = sorted(os.listdir(path=OUTPUT_DIR))[2:-1]

# iterate through all output files and read
overall_len_in, overall_len_out, overall_out_sent = [], [], []
for file in files:
    filepath = os.path.join(OUTPUT_DIR, file)
    
    len_in, len_out, out_sent = [], [], []
    with open(filepath, 'r') as f:
        data = json.load(f)[0]  # [{}, {}, ...]
        
        for q_id, info in data.items():
            lvl_gend = get_formlvl_gender(info=info)
            counts = count_output_len(info=info)
            combined_nums = lvl_gend | counts
            # print(combined_nums)
            
            len_in.append(combined_nums['input_len'])
            len_out.append(combined_nums['output_len'])
            out_sent.append(combined_nums['num_sent_out'])
            # break
    
    # TODO: check output format cuz min is saying 0        
    print(len_out)
    get_avg_min_max(len_out)        
    # overall counts
    overall_len_in.extend(len_in)
    overall_len_out.extend(len_out)
    overall_out_sent.extend(out_sent)
    break


