"""
    This file is only used for renaming q_id for the 20 sample questions x 5 topics sample data.
    Original q_id is e.g. q12_life_[...] For the second run the letter 'a' is added like q12a_life[...], 
    while for the third run the letter 'b' is added like q12b_life[...].
    
    The purpose is to keep the q_ids unique, because the q_ids all start with 1 in each run. 
    
    TODO (15.03.26): check functions and adjust or delete if not needed.
"""


import os
import json
import subprocess
from pathlib import Path


OUTPUT_DIR = "/mount/studenten/projects/takamamo/outputs"
files_trial2 = [i for i in os.listdir(path=OUTPUT_DIR) if "TRIAL2" in i]    # second run files
files_trial3 = [i for i in os.listdir(path=OUTPUT_DIR) if "TRIAL3" in i]    # third run files


# change file_trial<>
for file in files_trial3:
    filepath = os.path.join(OUTPUT_DIR, file)
    print(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    old_dict = data_list[0]
    new_dict = {}

    for q_id, info in old_dict.items():
        # new q_id 
        id_split = q_id.split('_')
        # a for second round, b for third round
        id_first = str(id_split[0])+"b_"
        rest_id = "_".join(i for i in id_split[1:])
        new_id = id_first+rest_id
        # print(new_id)
        
        new_dict[new_id] = info
    #     break
    # break
    # Overwrite 
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump([new_dict], f, ensure_ascii=False, indent=2)
        
        


# TODO: revise and check if ok !
def combine_run_files(path_runs_dir:str):
    
    # umbrella dirctory, e.g. /mount/studenten/projects/takamamo/outputs/qwen_1_7B_Mar13
    root = Path(path_runs_dir)
    # sub dir of roots => topics
    target_subdirs = ['career', 'daily', 'knowledge', 'life', 'simple']
    
    output_dir = root / "three_runs"
    output_dir.mkdir(exist_ok=True)

    for subdir_name in target_subdirs:
        subdir = root / subdir_name
        
        if not subdir.exists():
            print(f"Warning: {subdir_name} not found. Skipping.")
            continue

        # 2. Identify the three files based on their suffix
        # Adjust glob patterns if your naming convention differs slightly
        file1 = list(subdir.glob("*_1.json"))[0]
        file2 = list(subdir.glob("*_2.json"))[0]
        file3 = list(subdir.glob("*_3.json"))[0]

        # 3. Construct the jq command
        # We use map_values or walk to find q_ids and append "a" or "b" 
        # based on your specific requirements.
        jq_filter = (
            '([.[0][]] | map(.q_id |= .)) + ' # Run 1: keep as is
            '([.[1][]] | map(.q_id |= sub("q(?<num>\\\\d+)"; "q\\(.num)a"))) + ' # Run 2: q3 -> q3a
            '([.[2][]] | map(.q_id |= sub("q(?<num>\\\\d+)"; "q\\(.num)b"))) | ' # Run 3: q3 -> q3b
            '[add]' # Wrap in final array if needed by your schema
        )

        output_filename = f"run3x_{subdir_name}.json"
        output_path = output_dir / output_filename

        # 4. Execute the bash command
        # We pass the files into jq as a positional array using --slurp (-s)
        cmd = f"jq -s '{jq_filter}' {file1} {file2} {file3} > {output_path}"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully created: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {subdir_name}: {e}")

# Usage
# combine_run_files('./my_data_folder')