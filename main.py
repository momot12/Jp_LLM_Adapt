# Step 1: Data Preprocessing
## - read files (raw data)
## - lingusitic variatios
## - save

# Step 2: Run models
## - run each model with specific params
## - save output

# Step 3: Output Processing
## - determine F, G
## - get basic stats
## - get psycholing stats
## - PLOT stats
## - save stats

# Step 4: Additional Analysis
## - Calc Convergence (conv(t)) or sth similar
## - Correlations between e.g. skippable vs all particles

from scripts.run_data_preprocess import run_data_preprocessing
from scripts.run_models import run_model
from scripts.run_statistics import run_stats


if __name__=="__main__":
    
    #TODO: collect all params from yaml, so only need to set params in yaml
    
    # Step 1:
    # data_in_path = "/mount/arbeitsdaten/studenten4/takamamo/data/raw_qs_samples/"
    # data_out_path = "/mount/arbeitsdaten/studenten4/takamamo/data"
    # run_data_preprocessing(data_in=data_in_path, data_out=data_out_path)
    
    # Step 2:
    # config_yaml = ["config", "config_stable", "config_qwen"]
    # dir_saved = run_model(config_file=config_yaml[2])
    
    # Sanity Check
    # check if output is Japanese, if not take only Japanese output
    # TODO or Keep lang check in run_model()?
    # count = {counts_jp: x, counts_non_jp: x, counts_all: x}
    # only_jp, counts = lang_valid()
    
    # Step 3:
    # subdir = dir_saved
    # sub_dir = "qwen_1_7B_Mar12"
    # run_stats(sub_dir=subdir)
    
    # Step 4:
    # run_add_analysis()
    pass