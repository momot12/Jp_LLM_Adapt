import os, sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.helper_functions import OUTPUT_DIR, OUTPUT_DIR_ADD_STATS, get_avg_min_max
from src.more_analysis.bleu_score import write_bleu_sent



def run_add_analysis(sub_dir:str):
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be topics + single_run
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))
    
    # topic directories
    for dir in subdirs:
        OUTPUT_DIR_FULL =  os.path.join(OUTPUT_DIR_subdirs, dir)
        files = sorted(os.listdir(path=OUTPUT_DIR_FULL))
        
        print(f"\n*** TOPIC: {dir} ***")
        
        overall_bleu_pertopic = []
        # json files
        for file in files:
            filepath = os.path.join(OUTPUT_DIR_FULL, file)
            print(f"\nFILE: {filepath}")
            # e.g. qwen_1_7B_career_1
            model_topic_run = str(sub_dir) + "_" + "_".join(file[:-5].split('_')[-2:])
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                if dir == "simple": #topics:
                    pairs = []
                    for item in data:
                        
                        # different json formatting TODO: check again (14.3.26)
                        if len(item) > 1:
                            for qa_data in item.values():
                                pairs.append({"question": qa_data["question"], "only_out": qa_data["only_out"]})
                        else:
                            qa = list(item.values())[0]
                            pairs.append({"question": qa["question"], "only_out": qa["only_out"]})
                        
                    # BLUE score on first sentence
                    # returns [avg, v_min, v_max, v_std]
                    bleu_stats = write_bleu_sent(data=pairs, 
                                                 model_topic_run=model_topic_run,
                                                 output_path=OUTPUT_DIR_ADD_STATS,
                                                 mode=2)
                    # TODO: add BLEU for full -> mode=1
                
                    overall_bleu_pertopic.append(bleu_stats)
        
        if dir == "simple": #in topics:
            # get average per topic (3 runs)
            columns = list(zip(*overall_bleu_pertopic))
            avg_topic, _, _, _ = get_avg_min_max(columns[0])
            _, min_topic, _, _ = get_avg_min_max(columns[1])
            _, _, max_topic, _ = get_avg_min_max(columns[2]) 
            print(f"\nBLEU overall ({dir}):\n\tAvg: {avg_topic}\n\tMin: {min_topic}\n\tMax: {max_topic}")
  
                
                # per file
                # for q_id, info in data.items():
                    
                    


sub_dir = "qwen_1_7B"
run_add_analysis(sub_dir=sub_dir)