import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stats.stat_functions import (single_file_stats, 
                                      plot_all_boxplots,
                                      save_stats_to_file,
                                      generate_heatmaps)
from src.stats.plots_heatmaps import bar_plots_stacked
from src.utils.helper_functions import (OUTPUT_DIR, OUTPUT_DIR_STATS, combine_num_lists, _check_dir_exist)


def run_stats(sub_dir:str):
    
    topics = ["career", "daily", "knowledge", "life", "simple"]

    OUTPUT_DIR_subdirs = os.path.join(OUTPUT_DIR, sub_dir)
    # should be: 5 topics + single_run 
    subdirs = sorted(os.listdir(path=OUTPUT_DIR_subdirs))
    
    report_dir = os.path.join(OUTPUT_DIR_STATS, datetime.now().strftime("%b%d"), sub_dir)
    _check_dir_exist(report_dir)
    
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
        
        # filepaths_list = []
        # 3 files each topic
        statlists_dict_each_run = {1: {}, 2: {}, 3: {}}
    ###-------------------------------------------------------------------###
        # each run json file
        if dir in topics:
            print("-"*30)
            print(f"\n*** TOPIC: {dir} ***")
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
                statlists_dict_each_run[int(model_topic_run[-1])] = statlists_dict
                # 5xfirst runs files 
                if int(model_topic_run[-1]) == 1:
                    statlists_dict_single[dir] = statlists_dict
    ###-------------------------------------------------------------------###
        ## PER FILE:
                # only for each run of each topic
                # get heatmaps, boxplots, overall txt
                save_stats_to_file(statlists_dict, os.path.join(report_dir, f"{FILENAME}.txt"), mode="single", part=0)
                generate_heatmaps(statlists_dict, FILENAME, report_dir, mode="single")
                # plot_all_boxplots(lists=statlists_dict, part=0)
                
                print(f"[PER TOPIC] Created plots & txt files as {FILENAME}")
    
    ###-------------------------------------------------------------------###
        # 3xRuns for topic -> {topic: {1: {[],[]}, 2: {[],[]}, 3: {[],[]}}}
        statlists_dict_three[dir] = statlists_dict_each_run
    ###-------------------------------------------------------------------###
    ## PER TOPIC: 3x per topic
    for topic in topics:
        if topic in statlists_dict_three:
            # e.g. qwen_1_7B_career_stats_3xrun
            topic_FILENAME_3x = f"{sub_dir}_{topic}_stats_3xrun"
            
            # This uses 'mode="three_runs"' which usually handles the {1:..., 2:..., 3:...} structure
            # to show individual run stats followed by their internal comparison.
            save_stats_to_file(
                statlists_dict_three[topic], 
                os.path.join(report_dir, f"{topic_FILENAME_3x}.txt"), 
                mode="three_runs", 
                part=0
            )
            generate_heatmaps(statlists_dict_three[topic], topic_FILENAME_3x, report_dir, mode="three_runs")
            print(f"Created Topic-level 3x summary: {topic_FILENAME_3x}")
    
    # 1x -> statlists_dict_single
    # 3x  -> statlists_dict_three
    ###---------------------------------------------------###
    ## OVERALL:
    print(f"--------------------------------\nOverall:")
    print(f"\n--- Combining all topics per run ---")

    # Data structure to hold: { 1: [all stats for run1], 2: [all stats for run2], 3: [all stats for run3] }
    # len(run_id) -> 5, inside 5x180
    all_topics_per_run = {1: [], 2: [], 3: []}

    for topic in topics:
        if topic in statlists_dict_three:
            for run_id in [1, 2, 3]:
                if run_id in statlists_dict_three[topic]:
                    all_topics_per_run[run_id].append(statlists_dict_three[topic][run_id])

    # Use combine_num_lists to merge the 5 files for each run
    final_run_comparison = {}
    for run_id, list_of_dicts in all_topics_per_run.items():
        if list_of_dicts:
            final_run_comparison[run_id] = combine_num_lists(list_of_dicts)

    # 3. Save the final "Run 1 vs 2 vs 3" report
    FILENAME_final_comp = "overall_run_comparison_5topics"
    save_stats_to_file(
        final_run_comparison, 
        os.path.join(report_dir, f"{FILENAME_final_comp}.txt"), 
        mode="three_runs", 
        part=1 # part=1 calculates the "Total Avg" of all three runs combined
    )
    generate_heatmaps(final_run_comparison, FILENAME_final_comp, report_dir, mode="three_runs")
    print(f"Final Verification Complete: {FILENAME_final_comp}.txt created.")
    # Overall for ALL runs (3 runs x 5 topics)
    
    # print_save_stats(lists=all_overall_lists, part=1, filename=FILENAME_over, to_plot=False)
    # get boxplot for basic stats
    # plot_all_boxplots(all_overall_lists, part=1)
    print(f"#############################END####\n")
    



# sub_dir = 'jp_stable_7B'
sub_dir = 'qwen_4B'
run_stats(sub_dir=sub_dir)