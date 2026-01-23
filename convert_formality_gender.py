import csv


original_data = "/mount/arbeitsdaten/studenten4/takamamo/sample_data.csv"


def read_csv(data_path:str):
    with open(data_path, 'r') as f:
        data = f.read()
    
    return data

print(read_csv(original_data))


# Create ending mapping neutral
# create gender ending mapping 
# create pronoun mapping (1st+2nd sg)