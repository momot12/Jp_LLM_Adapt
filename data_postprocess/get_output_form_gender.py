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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from detect_f_g import DetectFG
from utils.calc_stats import matrix_plot
from utils.helper_functions import OUTPUT_DIR, OUTPUT_DIR_STATS, _check_dir_exist
from utils.psycholing_measures import get_psycholinguistic_metrics, load_model_psycholing

# OUTPUT_DIR = "/mount/studenten/projects/takamamo/Jp_LLM_adapt/data_SAMPLE"

def _output_clean_first_sent(raw_sent:str):
    """
        Filters the clean output.
    """
    match = re.search(r"<\|assistant\|>\n?(.*?)(?=<\|endoftext\|>|$)", 
                  raw_sent, re.DOTALL)
    output_clean = match.group(1).strip() if match else ""
    
    sentences = [s for s in re.split(r'[。！!？?]', output_clean) if s.strip()]
    return output_clean, sentences[0]

def measure_formlvl_gender(out_text:str):
    """
        Cleans output string, only taking the raw output string, no special tags like <|assistant|>.
        Runs formality and gender detecting class and returns
            {"formality": detected_formality,
                "gender": detected_gender,
                }
    """
    output_clean, _ = _output_clean_first_sent(raw_sent=out_text)
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
    
    output_clean, first_sent = _output_clean_first_sent(raw_sent=info["output"])
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
                "out_len_first": len(first_sent)
                }
    
    return len_info

def get_avg_min_max(numbers_list:list):
    avg = round(sum(numbers_list)/len(numbers_list), 3)
    val_min = min(numbers_list)
    val_max = max(numbers_list)
    # print(f"\tAvg: {avg}")
    # print(f"\tMin: {val_min}")
    # print(f"\tMax: {val_max}")
    return avg, val_min, val_max

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

    avg_dict, avg_sent_dict = {}, {}
    print("Avg #chars output per formality:")
    for level in [1, 2, 3]:
        # if 0 division, then keep 0
        avg = totals[level] / counts[level] if counts[level] > 0 else 0
        avg_dict[level] = round(avg, 2)
        print(f"\tF{level}: {avg:.2f} (#samples: {counts[level]})")
    
    print("Avg #sent output per formality:")
    for level in [1, 2, 3]:
        # if 0 division, then keep 0
        avg_sent = totals_sent[level] / counts_sent[level] if counts_sent[level] > 0 else 0
        avg_sent_dict[level] = round(avg_sent, 2)
        print(f"\tF{level}: {avg_sent:.2f} (#samples: {counts_sent[level]})")
    
    return avg_dict, avg_sent_dict, counts, counts_sent

def get_aoa_complexities(info:dict, loaded_model):
    """
        Get psychcolinguostic measures:
            - #Particles: more particles == more formal/advanced
            - Lexical: Kango (complex, >2 Kanjis) vs. Wago (simple)
                = #Kango/#content_token
            - Syntactic: sub-clauses == depth tree
                => deeper == complex --> high number == complex
            - AoA (proxy): more Kanji usage, more advanced 
                => more Kanjis, older person
                => min=5; max=15.0
    """
    _, first_sent = _output_clean_first_sent(raw_sent=info["output"])
    stats = get_psycholinguistic_metrics(sentence=first_sent, loaded_model=loaded_model)
    
    # print(f"\nSentence: {first_sent}")
    # for i, v in stats.items():
    #     print(f"\t{i}: {v}")
    return stats

def _print_save_stats(lists:dict, filename:str, max_val_2:int=None, max_val_3:int=None, part:int=1, to_plot:bool=False):
    """
        Takes a dictionary with a number of lists, to print and save stats in txt.
        Keys: {"len_in"         : len_in,
               "len_out"        : len_out,
               "out_sent"       : out_sent,
               "out_sent1_len"  : out_sent1_len,
               "AoA"            : aoa,
               "syntax_depth"   : syntax_depth,
               "lexical_complex": lexical_complexity,
               "num_particles"  : total_particle,
               "full"           : per_topic_stats, 
              }
        
        Args:
                 part (int): 0 = "per topic", 1 = "overall"
            max_val_2 (int): for scaling the heatmap; mode=2
            max_val_3 (int): for scaling the heatmap; mode=3
             to_plot (bool): True==plotting, False==no plotting
   
    """
    which_part = ["per topic", "overall"]
    
    # today's date, e.g. Feb12
    new_dir = datetime.now().strftime("%b%d")
    FiNAL_DIR_path = os.path.join(OUTPUT_DIR_STATS, new_dir)
    _check_dir_exist(FiNAL_DIR_path)
    
    # TXT file
    prefix = "".join(t.capitalize() for t in which_part[part].split())
    txt_file = os.path.join(FiNAL_DIR_path, f"{prefix}_{filename}.txt")
    
    
    with open(txt_file, "w", encoding="utf-8") as f:
        # helper function:
        def process_stat(label, key, is_header=False):
                if is_header:
                    header = f"\n---[{which_part[part].upper()}] - {label}:---\n"
                    print(header.strip())
                    f.write(header)
                    return
                # Get values
                avg, v_min, v_max = get_avg_min_max(lists[key])
                # Format block
                block = (f"{label}\n"
                        f"\tAvg: {avg}\n"
                        f"\tMin: {v_min}\n"
                        f"\tMax: {v_max}\n")
                
                print(block.strip()) # print in terminal
                f.write(block)       # write to file
                
        # --- EXECUTE BASIC STATS ---
        process_stat("BASIC STATS", None, is_header=True)
        process_stat(f"#chars input {which_part[part]}:", "len_in")
        process_stat(f"#chars output {which_part[part]}:", "len_out")
        process_stat(f"#sentence output {which_part[part]}:", "out_sent")
        process_stat(f"#chars for first sentence {which_part[part]}:", "out_sent1_len")

        # --- EXECUTE PSYCHOLING STATS ---
        print()
        process_stat("PSYCHOLING STATS", None, is_header=True)
        process_stat("Avg AoA:", "AoA")
        process_stat("Avg depth:", "syntax_depth")
        process_stat("Avg Lexical complexity (#kanjis):", "lexical_complex")
        process_stat("Avg #Particles:", "num_particles")  
        
        # --- EXECUTE FORMALITY ---
        print()
        process_stat("PER FORMALITY", None, is_header=True)
        avg_dict, avg_sent_dict, counts, counts_sent = get_avg_out_performality(stat_info=lists["full"])
        
        # Formatting the dictionary output for the file
        f.write("Avg #chars output per formality:\n")
        for level in [1, 2, 3]:
            f.write(f"\tF{level}: {avg_dict[level]} (#samples: {counts[level]})\n")
        
        f.write("\nAvg #sent output per formality:\n")
        for level in [1, 2, 3]:
            f.write(f"\tF{level}: {avg_sent_dict[level]} (#samples: {counts_sent[level]})\n")
    
    ## PLOTS PER TOPIC:
    if to_plot:
        # matrix plot per topic - only formality
        matrix_plot(lists["full"], filename=os.path.join(FiNAL_DIR_path, filename+"_f"), mode=1)
        # matrix plot per topic - formality & gender 
        matrix_plot(lists["full"], max_val=max_val_2, filename=os.path.join(FiNAL_DIR_path, filename+"_f_g"), mode=2)
        # matrix plot per topic - formality-gender (input vs output)
        matrix_plot(lists["full"], max_val=max_val_3, filename=os.path.join(FiNAL_DIR_path, filename+"_FG"), mode=3) 

    print(f"\n#----SAVED----#")
    if to_plot:
        print(f"***Saved heatmap (only formality in-out): {FiNAL_DIR_path+filename+'_f'}\
              \n***Saved heatmap (formality vs. gender out): {FiNAL_DIR_path+filename+'_f_g'} \
              \n***Saved heatmap (formality-gender out): {FiNAL_DIR_path+filename+'_FG'}\n")
    print(f"***Saved stat file in: {txt_file}")
    

# ------- MAIN ------
def run_stats():
    # specific for sample data
    files = sorted(os.listdir(path=OUTPUT_DIR))[2:-1]
    
    # load model for psycholing. norms
    nlp_model = load_model_psycholing()

    # iterate through all output files and read
    overall_len_in, overall_len_out, overall_out_sent, overall_out_sent1_len = [], [], [], []
    overall_aoa, overall_syntax, overall_lexical, overall_particles = [], [], [], []
    overall_stats = []
    for file in files:
        filepath = os.path.join(OUTPUT_DIR, file)
        
        len_in, len_out, out_sent, out_sent1_len = [], [], [], []
        aoa, syntax_depth, lexical_complexity, total_particle = [], [], [], []
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
                # "AoA","syntax_depth", "lexical_complexity", "total_count_particle"
                aoa_complex = get_aoa_complexities(info=info, loaded_model=nlp_model)
                # {'input': ['3', 'neutral'], 'output': [2, 'neutral'], 'input_len': 25, 'output_len': 67, 'num_sent_out': 1}
                combined_nums = lvl_gend | counts | aoa_complex
                # print(combined_nums)
                
                per_topic_stats.append(combined_nums)
                # print(f"{q_id}: {combined_nums['output']}")
                
                # psycholing. stats per topic
                aoa.append(combined_nums["AoA"])
                syntax_depth.append(combined_nums["syntax_depth"])
                lexical_complexity.append(combined_nums["lexical_complexity"])
                total_particle.append(combined_nums["total_count_particle"])
                
                # basic stats per topic
                len_in.append(combined_nums['input_len'])
                len_out.append(combined_nums['output_len'])
                out_sent.append(combined_nums['num_sent_out'])
                out_sent1_len.append(combined_nums['out_len_first'])
                #     q_id_count+=1
                # else:
                #     break
                
        # per topic done
        ###-------------------------------------------------------------------###
        ## PER TOPIC:
        print(f"\nFilename: {file}")

        all_per_lists = {"len_in": len_in,
                         "len_out": len_out,
                         "out_sent": out_sent,
                         "out_sent1_len": out_sent1_len,
                         "AoA": aoa,
                         "syntax_depth": syntax_depth,
                         "lexical_complex": lexical_complexity,
                         "num_particles": total_particle,
                         "full": per_topic_stats, 
                         }
        # get all stats visually
        _print_save_stats(lists=all_per_lists, filename=FILENAME, part=0, to_plot=False)
    
        ###-----------###
        # Collect overll- over all topics
        overall_stats.extend(per_topic_stats)
        
        overall_aoa.extend(aoa)
        overall_syntax.extend(syntax_depth)
        overall_lexical.extend(lexical_complexity)
        overall_particles.extend(total_particle)
        
        overall_len_in.extend(len_in)
        overall_len_out.extend(len_out)
        overall_out_sent.extend(out_sent)
        overall_out_sent1_len.extend(out_sent1_len)
        
        # break
    ###-------------------------------------------------------------------###
    ## OVERALL - all topics
    print(f"--------------------------------\nOverall:")
    
    all_overall_lists = {"len_in": overall_len_in,
                         "len_out": overall_len_out,
                         "out_sent": overall_out_sent,
                         "out_sent1_len": overall_out_sent1_len,
                         "AoA": overall_aoa,
                         "syntax_depth": overall_syntax,
                         "lexical_complex": overall_lexical,
                         "num_particles": overall_particles,
                         "full": overall_stats, 
                        }

    FILENAME = f"over_5_topics"
    _print_save_stats(lists=all_overall_lists, part=1, filename=FILENAME, to_plot=False)
    print(f"#############################END####\n")
    
    
if __name__ == "__main__":
    run_stats()