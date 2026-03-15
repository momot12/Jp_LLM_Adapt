# Configs

### YAML files:
- [config_stable.yaml](config_stable.yaml): Contains parameters for Japanese StableLM models. 
- [config_qwen.yaml](config_qwen.yaml): Contains parameters for Multilingual Qwen models.
- [temp_config.yaml](temp_config.yaml): A temporary yaml configuration to dynamically run all data files and multiple runs.
- [config.yaml](config.yaml): A general config file to try out all variables.

### What to change:
- `paths`: Adjust paths if necessary. Make sure directories are available.
- `files`: Input data files and runs change dynamically (temp_config.yaml). Files are hard-coded in [run_all.sh](../run_all.sh).
- `pipeline`: Set true or false to run/skip specific steps.
- `model`: Choose a model, and set cuda if necessary.
    - **Available models**: 
        - *qwen_1_7B*
        - *qwen_4B*
        - *jp_stable_1_6B*
        - *jp_stable_7B*
- `parameters`: Change system prompt if desired.
    - **Default** `system_prompt`: "一文のみでの回答。" (*"only one sentence reply."*)
    - **Alternatives**:
        - TBD: gender prompts


##### NOTE: The model parameters are set to their default values. They do not need to be changed, but can be modified if desired.
