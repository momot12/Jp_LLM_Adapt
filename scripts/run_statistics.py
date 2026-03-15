import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import single_file_stats, print_save_stats, plot_all_boxplots, individual_runs, check_FG_now
from src.stats.plots_heatmaps import bar_plots_stacked
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, combine_num_lists)


def run_stats(sub_dir:str):
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be: 5 topics + single_run 
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))
    
    
    # 5x first runs files 
    statlists_dict_single = {topic: {} for topic in topics}
    # 5x all three runs
    statlists_dict_three = {topic: {} for topic in topics}
    FILENAME = ""
    # topic directories
    for dir in subdirs:
        
        # ~/output_dir/model_name/topic
        OUTPUT_DIR_FULL =  os.path.join(OUTPUT_DIR_subdirs, dir)
        # should be: 3 files for each run (1, 2, 3) with '.._[run].json'
        files = sorted(os.listdir(path=OUTPUT_DIR_FULL))
        
        print(f"\n*** TOPIC: {dir} ***")
        # filepaths_list = []
        # 3 files each topic
        statlists_dict_each_run = {1: {}, 2: {}, 3: {}}
    ###-------------------------------------------------------------------###
        # each run json file
        for file in files:
            # ~/output_dir/model_name/topic/file_[run].json
            filepath = os.path.join(OUTPUT_DIR_FULL, file)
            print(f"\nFILE: {filepath}")
            # e.g. qwen_1_7B_career_1
            model_topic_run = str(sub_dir) + "_" + "_".join(file[:-5].split('_')[-2:])
            FILENAME = f'{model_topic_run}_stats'
            
            # get dict with lists of basic & psycholing. stats
            statlists_dict = single_file_stats(filepath=filepath)
            # save individual run in separate dictionary
            # 3 files each topic
            statlists_dict_each_run[model_topic_run[-1]] = statlists_dict
            # 5xfirst runs files 
            if model_topic_run[-1] == 1:
                statlists_dict_single[dir] = statlists_dict
    ###-------------------------------------------------------------------###
        ## PER FILE:
            # TODO: check plots logic
            # only for each run of each topic
            if dir in topics:
                # get heatmaps, boxplots, overall txt
                print_save_stats(lists=statlists_dict, filename=FILENAME, part=0, to_plot=False)
                # plot_all_boxplots(lists=statlists_dict, part=0)
                
                print(f"[PER FILE] Created plots & txt files as {FILENAME}")
            
        # {topic: {1: {[],[]}, 2: {[],[]}, 3: {[],[]}}}
        statlists_dict_three[dir] = statlists_dict_each_run
    ###-------------------------------------------------------------------###
    ## PER TOPIC:
    # 1x vs 3x per topic
    
    # e.g. qwen_1_7B_career_1xrun
    FILENAME_1x = f"{FILENAME[:-2]}_1xrun"
    FILENAME_3x = f"{FILENAME[:-2]}_3xrun"
    
    # 1x -> statlists_dict_single
    # 3x  -> statlists_dict_three
    
    # TODO: check if topics need to be stacked too; CHECK IF COMBINE 3x is needed (plot)
    # get heatmaps, boxplots, overall txt
    # 1xRun
    # print_save_stats(lists=combined_1x, filename=FILENAME_1x, part=0, to_plot=False)
    # plot_all_boxplots(lists=statlists_dict, part=0)
    
    # 3xRun
    # TODO: check plots if they need per run
    # print_save_stats(lists=combined_3x, filename=FILENAME_3x, part=0, to_plot=False)
    # plot_all_boxplots(lists=statlists_dict, part=0)

    ###-------------------------------------------------------------------###
    ## OVERALL:
    print(f"--------------------------------\nOverall:")
    # 1x -> statlists_dict_single (combo)
    overall_1x = combine_num_lists(statlists_dict_single)
    # 3x  -> statlists_dict_three (combo)
    # overall_3x =
    FILENAME_over_1x = f"over_5_topics_1xrun"
    FILENAME_over_3x = f"over_5_topics_3xrun"
    
    # print_save_stats(lists=all_overall_lists, part=1, filename=FILENAME_over, to_plot=False)
    # get boxplot for basic stats
    # plot_all_boxplots(all_overall_lists, part=1)
    print(f"#############################END####\n")
    




# def check_FG(sub_dir:str):
#     OUTPUT_DIR_FULL = os.path.join(OUTPUT_DIR, sub_dir)
#     # specific for sample data
#     files = sorted(os.listdir(path=OUTPUT_DIR_FULL))
#     # NOTE: remove later; just for trying out
#     files = [f for f in files if "SIMPLE" in f]
    
#     print(f"Available files: {files}")
    
#     filepaths_list = []
#     for file in files:
#         # filepath = os.path.join(OUTPUT_DIR_FULL, file)
#         filepath = os.path.join(OUTPUT_DIR_FULL, file)
#         filepaths_list.append(filepath)
#         lvl_g = check_FG_now(filepath=filepath)
        
        

# if __name__ == "__main__":
#     sub_dir = "qwen_1_7B_Mar12" # trial1x_week5
#     # run_stats(sub_dir=sub_dir)
    
#     # check_FG(sub_dir=sub_dir)
#     # file_path = "/mount/studenten/projects/takamamo/outputs/sample_runs_week2-5"
#     # files = os.listdir(path=file_path)
    
#     # TODO: function that compares all 3 runs 
#     # single_file_stats()