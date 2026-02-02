import os
import csv
from convert_formality_gender import FormalityConverter


def main():
    # init converter
    converter = FormalityConverter()
    
    # headers for new csv 
    headers = ["q_id", "question", "topic", "formality", "gender"]

    # read raw questions
    path_dir = "/mount/arbeitsdaten/studenten4/takamamo/data/raw_qs_samples/"
    raw_files = sorted(os.listdir(path=path_dir))
    # print(raw_files)
    
    for i,f in enumerate(raw_files):
        file_path = os.path.join(path_dir, f)
        
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
        out_file = os.path.join(path_dir, f"conv_qs_20_{topic}.csv")
        with open(out_file, "a", newline="", encoding ="utf-8") as out:
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
        # break



if __name__ == "__main__":
    main()