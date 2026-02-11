# read output files (done)
## Func1: (done)
    # get input formality, gender (done)
    # measure output formaility (done)
## Func2: (done)
    # measure len(in) vs len(out) --> num chars (done)
    # num sent out (done)
## in main or func3:
    # for each formality: avg output length (done)


import sys
import os
import json
import re
from detect_f_g import DetectFG
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.calc_stats import matrix_plot
from utils.helper_functions import OUTPUT_DIR

# OUTPUT_DIR = "/mount/studenten/projects/takamamo/Jp_LLM_adapt/data_SAMPLE"

def measure_formlvl_gender(out_text:str):
    """
        Cleans output string, only taking the raw output string, no special tags like <|assistant|>.
        Runs formality and gender detecting class and returns
            {"formality": detected_formality,
                "gender": detected_gender,
                }
    """
    match = re.search(r"<\|assistant\|>\n?(.*?)(?=<\|endoftext\|>|$)", 
                  out_text, re.DOTALL)
    output_clean = match.group(1).strip() if match else ""
    
    engine = DetectFG()
    return engine.detect(text=output_clean)
    
def get_formlvl_gender(info:dict):
    """
        Get formality levels and gender from input and output, and format.
        Return a dict {"input": [formality, gender], "output": [formality, gender]}.
    """
    formal_gend_out = measure_formlvl_gender(info["output"])
    
    out_dict = {"input":[info["formality"], info["gender"]],
                "output": [formal_gend_out["formality"], formal_gend_out["gender"]],
                }
    
    return out_dict

def count_output_len(info:dict):
    """
        Counts #output sentences, and #chars of input vs output, including punctuation.
        Returns a dictionary with len(input), len(output), and #output sentences.
        
        NOTE: maybe exclude punctuation in calculation?
    """
    len_info = {}
    
    match = re.search(r"<\|assistant\|>\n?(.*?)(?=<\|endoftext\|>|$)", 
                  info["output"], re.DOTALL)
    output_clean = match.group(1).strip() if match else ""
    input = info["question"]
    sentences = [s for s in re.split(r'[。！!？?]', output_clean) if s.strip()]
    
    if len(output_clean) == 0:
        print(f"Clean output:\n{output_clean}")
        print(f"Original output:\n{info['output']}\n\n")
    
    # if len(sentences) == 8:
    #     print(f"{output_clean}\n{sentences}")
        
    
    len_info = {"input_len": len(input),
                "output_len": len(output_clean),
                "num_sent_out": len(sentences),
                "out_len_first": len(sentences[0])
                }
    
    return len_info

def get_avg_min_max(numbers_list:list):
    print(f"\tAvg: {sum(numbers_list)/len(numbers_list)}")
    print(f"\tMin: {min(numbers_list)}")
    print(f"\tMax: {max(numbers_list)}")

def get_avg_out_performality(stat_info:dict):
    # totals: out_len, count: occurrence
    totals = {1: 0, 2: 0, 3: 0}
    counts = {1: 0, 2: 0, 3: 0}
    
    totals_sent = {1: 0, 2: 0, 3: 0}
    counts_sent = {1: 0, 2: 0, 3: 0}

    for item in stat_info:
        # Access the detected output formality level
        # Based on your data structure: item["output"] is [level, gender]
        level = int(item["output"][0])
        length = int(item["output_len"])
        num_sent = int(item["num_sent_out"])

        if level in totals:
            totals[level] += length
            counts[level] += 1
            
            totals_sent[level] += num_sent
            counts_sent[level] += 1

    print("Avg #chars output per formality:")
    for level in [1, 2, 3]:
        # if 0 division, then keep 0
        avg = totals[level] / counts[level] if counts[level] > 0 else 0
        print(f"\tF{level}: {avg:.2f} (#samples: {counts[level]})")
    
    print("Avg #sent output per formality:")
    for level in [1, 2, 3]:
        # if 0 division, then keep 0
        avg_sent = totals_sent[level] / counts_sent[level] if counts[level] > 0 else 0
        print(f"\tF{level}: {avg_sent:.2f} (#samples: {counts_sent[level]})")


# ------- MAIN ------
def run_stats():
    # specific for sample data
    files = sorted(os.listdir(path=OUTPUT_DIR))[2:-1]

    # iterate through all output files and read
    overall_len_in, overall_len_out, overall_out_sent, overall_out_sent1_len = [], [], [], []
    overall_stats = []
    for file in files:
        filepath = os.path.join(OUTPUT_DIR, file)
        
        len_in, len_out, out_sent, out_sent1_len = [], [], [], []
        per_topic_stats = []
        with open(filepath, 'r') as f:
            data = json.load(f)[0]  # [{}, {}, ...]
            
            # per topic file 
            # q_id_count = 1
            for q_id, info in data.items():
                FILENAME = f"{info['topic']}_20"
                # if q_id_count <= 10:
                lvl_gend = get_formlvl_gender(info=info)
                counts = count_output_len(info=info)
                # {'input': ['3', 'neutral'], 'output': [2, 'neutral'], 'input_len': 25, 'output_len': 67, 'num_sent_out': 1}
                combined_nums = lvl_gend | counts
                # print(combined_nums)
                
                per_topic_stats.append(combined_nums)
                # print(f"{q_id}: {combined_nums['output']}")
                len_in.append(combined_nums['input_len'])
                len_out.append(combined_nums['output_len'])
                out_sent.append(combined_nums['num_sent_out'])
                out_sent1_len.append(combined_nums['out_len_first'])
                    # q_id_count+=1
                # else:
                # break
                
        # per topic done
        ## PER TOPIC:
        print(f"\nFilename: {file}")
        # IN/OUT
        print(f"#chars input per topic:")
        get_avg_min_max(len_in)
        print(f"#chars output per topic:")
        get_avg_min_max(len_out)
        print(f"#sentence output per topic:")
        get_avg_min_max(out_sent)
        print(f"#chars for first sentence per topic:")
        get_avg_min_max(out_sent1_len)
        
        # get avg #char of output - per topics
        get_avg_out_performality(stat_info=per_topic_stats)
        
        ## PLOTS PER TOPIC:
        # matrix plot per topic - only formality
        # matrix_plot(per_topic_stats, filename=FILENAME+"_f", mode=1)
        # matrix plot per topic - formality & gender 
        # matrix_plot(per_topic_stats, max_val=160, filename=FILENAME+"_f_g", mode=2)
        # matrix plot per topic - formality-gender (input vs output)
        # matrix_plot(per_topic_stats, max_val=50, filename=FILENAME+"_FG", mode=3) 
    
        # overall - over all topics
        overall_stats.extend(per_topic_stats)
        overall_len_in.extend(len_in)
        overall_len_out.extend(len_out)
        overall_out_sent.extend(out_sent)
        overall_out_sent1_len.extend(out_sent1_len)
        
        # break

    ## OVERALL - all topics
    print(f"--------------------------------\nOverall:")
    # IN/OUT
    print(f"#chars input overall:")
    get_avg_min_max(overall_len_in)
    print(f"#chars output overall:")
    get_avg_min_max(overall_len_out)
    print(f"#sentence output overall:")
    get_avg_min_max(overall_out_sent)
    print(f"#chars for first sentence overall:")
    get_avg_min_max(overall_out_sent1_len)
    # get avg #char of output - overall
    get_avg_out_performality(stat_info=overall_stats)

    ## PLOTS - OVERALL:
    # matrix plot - only formality (input vs output)
    # matrix_plot(overall_stats, filename="over_5_topics_F", mode=1)
    # matrix plot per topic - formality & gender 
    # matrix_plot(info_list=overall_stats, max_val=800, filename="over_5_topics_F_G", mode=2)
    # matrix plot per topic - formality-gender  (input vs output)
    # matrix_plot(info_list=overall_stats, max_val=150, filename="over_5_topics_FG", mode=3)
    print(f"#############################END####\n")
    
    
if __name__ == "__main__":
    run_stats()