import os
import csv
from src.data_preprocess.convert_formality_gender import FormalityConverter


def run_data_preprocessing(data_in:str="/mount/arbeitsdaten/studenten4/takamamo/data/raw_qs_samples/",
                           data_out:str="/mount/studenten/projects/takamamo/data"):
    # init converter
    converter = FormalityConverter()
    
    # headers for new csv 
    headers = ["q_id", "question", "topic", "formality", "gender"]

    # read raw questions
    raw_files = sorted(os.listdir(path=data_in))
    # print(raw_files)
    
    for i,f in enumerate(raw_files):
        file_path = os.path.join(data_in, f)
        
        start_id = 1
        topic = f[10:-4]
        
        converted_per_topic = []
        with open(file_path, 'r') as file:
            file = file.readlines()
            for q in file:
                q = q.strip()
                # convert
                converted = converter.convert_all(q)
                converted_per_topic.extend(converted)
                
        # save in new csv file
        # NOTE (9.3.26): /mount/arbeitsdaten/studenten4/takamamo/data says no space anymore
        out_file = os.path.join(data_out, f"conv_qs_20_{topic}.csv")
        with open(out_file, "w", newline="", encoding ="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=headers)
            if start_id == 1:
                writer.writeheader()
    
            for i, item in enumerate(converted_per_topic):
                                        
                question = item["question"]
                formality = item["formality"]
                gender = item["gender"]
                
                writer.writerow({
                    "q_id": f"q{start_id}_{topic}_{formality}_{gender}",
                    "question": question,
                    "topic": topic,
                    "formality": formality,
                    "gender": gender,
                })
                start_id +=1