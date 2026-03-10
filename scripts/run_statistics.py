import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import single_file_stats, print_save_stats, plot_all_boxplots
from src.utils.helper_functions import OUTPUT_DIR, OUTPUT_DIR_STATS


def run_stats(sub_dir:str):
    # OUTPUT_DIR_FULL = os.path.join(OUTPUT_DIR, sub_dir)
    # specific for sample data
    # files = sorted(os.listdir(path=OUTPUT_DIR_FULL))
    files = [
        "qwen_1_7B_Mar10/Qwen_Qwen3-1.7B_True_False_128_0.7_0.8_trial_qwen_big_canbedeleted.json",
        # "trial1x_week5/stabilityai_japanese-stablelm-2-instruct-1_6b_True_128_一文で答え。_0.99_0.95_TEST_LIFE.json",
        # "sample_runs_week2-5/stabilityai_japanese-stablelm-2-instruct-1_6b_True_128_一文で答え。_0.99_0.95_TRIAL3_knowledge.json",
        # "sample_runs_week2-5/stabilityai_japanese-stablelm-2-instruct-1_6b_True_128_一文で答え。_0.99_0.95_TRIAL3_life.json",
        # "sample_runs_week2-5/stabilityai_japanese-stablelm-2-instruct-1_6b_True_128_一文で答え。_0.99_0.95_TRIAL3_simple.json"
        ]
    
    print(f"Available files: {files}")

    # basic stats
    overall_len_in, overall_len_out, overall_out_sent, overall_out_sent1_len = [], [], [], []
    # psycholing. stats
    overall_aoa_proxy, overall_aoa_list = [], []
    overall_syntax, overall_lexical =[], []
    overall_particles, overall_particles_skip = [], []
    # all combined
    overall_stats = []
    
    for file in files:
        # filepath = os.path.join(OUTPUT_DIR_FULL, file)
        filepath = os.path.join(OUTPUT_DIR, file)
        #  statlists_dict, combined_nums, FILENAME
        statlists_dict, _, FILENAME = single_file_stats(filepath=filepath)
        ###-------------------------------------------------------------------###
        ## PER TOPIC:
        print(f"\nFilename: {file}")
        # get all stats visually
        print_save_stats(lists=statlists_dict, filename=FILENAME, part=0, to_plot=False)
        # get boxplot for basic stats
        plot_all_boxplots(lists=statlists_dict, part=0)
    
        ###-------------------------------------------------------------------###
        ## OVERALL:
        # Collect overll- over all topics
        overall_stats.extend(statlists_dict["full"])
        
        overall_len_in.extend(statlists_dict["len_in"])
        overall_len_out.extend(statlists_dict["len_out"])
        overall_out_sent.extend(statlists_dict["out_sent"])
        overall_out_sent1_len.extend(statlists_dict["out_sent1_len"])
        
        overall_aoa_proxy.extend(statlists_dict["AoA_proxy"])
        overall_aoa_list.extend(statlists_dict["AoA_list"])
        overall_syntax.extend(statlists_dict["syntax_depth"])
        overall_lexical.extend(statlists_dict["lexical_complex"])
        overall_particles.extend(statlists_dict["num_particles"])
        overall_particles_skip.extend(statlists_dict["num_particles_skip"])


    print(f"--------------------------------\nOverall:")
    
    all_overall_lists = {"len_in": overall_len_in,
                         "len_out": overall_len_out,
                         "out_sent": overall_out_sent,
                         "out_sent1_len": overall_out_sent1_len,
                         "AoA_proxy": overall_aoa_proxy,
                         "AoA_list": overall_aoa_list,
                         "syntax_depth": overall_syntax,
                         "lexical_complex": overall_lexical,
                         "num_particles": overall_particles,
                         "num_particles_skip": overall_particles_skip,
                         "full": overall_stats, 
                        }

    FILENAME_over = f"over_5_topics"
    print_save_stats(lists=all_overall_lists, part=1, filename=FILENAME_over, to_plot=False)
    # get boxplot for basic stats
    plot_all_boxplots(all_overall_lists, part=1)
    print(f"#############################END####\n")


if __name__ == "__main__":
    sub_dir = "trial3x_week5" # trial1x_week5
    run_stats(sub_dir=sub_dir)
    
    # file_path = "/mount/studenten/projects/takamamo/outputs/sample_runs_week2-5"
    # files = os.listdir(path=file_path)
    
    # TODO: function that compares all 3 runs 
    # single_file_stats()