"""
    This file is only used for renaming q_id for the 20 sample questions x 5 topics sample data.
    Original q_id is e.g. q12_life_[...] For the second run the letter 'a' is added like q12a_life[...], 
    while for the third run the letter 'b' is added like q12b_life[...].
    
    The purpose is to keep the q_ids unique, because the q_ids all start with 1 in each run. 
"""



import os
import json

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