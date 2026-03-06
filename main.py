# Step 1: Data Preprocessing
## - read files
## - variatios
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

from scripts.run_models import run_model


if __name__=="__main__":
    # Step 1:
    # run_data_preprocessing()
    
    # Step 2:
    config_yaml = ["config", "config_stable", "config_qwen"]
    run_model(config_file=config_yaml[0])
    
    # Step 3:
    # run_stats()
    
    # Step 4:
    # run_add_analysis()
    # pass