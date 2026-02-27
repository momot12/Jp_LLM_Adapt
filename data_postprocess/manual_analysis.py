import os
import json
import random


def get_random_samples(file_path, sample_size=10):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)[0]
        
        random_ids = random.sample(list(data.keys()), sample_size)
        sampled_data = {q_id: data[q_id] for q_id in random_ids}
        
        return sampled_data
            
        

            
   
# TODO: get 10 random samples from each topic (get labels too)  
path = "/mount/studenten/projects/takamamo/outputs/trial3x_week5/"
dir_files = sorted(os.listdir(path=path))

for file in dir_files:
    file = os.path.join(path, file)
    samples = get_random_samples(file, 10)
    print(json.dumps(samples, indent=4, ensure_ascii=False))
    print()