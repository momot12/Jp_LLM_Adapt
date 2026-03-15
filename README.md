# Jp_LLM_adapt
Masterthesis

TBD WHAT is this repo about

### GitHub structure 

- configs (yaml files)
- data_sample (raw data + conversion as example)
- src: preprocessing, models, stats, analysis (additional), utils
- scripts: 1_preprocess, 2_run_models, 3_run_stats, 4_run_analysis
- main.py
- requirements.txt (NOTE: 2 venv needed --> explain in README)

### Model sources

## Setup

### Requirements:
- python3.11
- pip

### Step 1: Create virtual environments

```bash
python3 --version
python3.11 -m venv venv_name
source venv_name/bin/activate
```
```bash
pip install --no-cache-dir -r requirements.txt
```
##### OR (TODO: add reqs)
```bash
pip install --no-cache-dir -r requi
```

## How to run

### 1. Set parameters in yaml file
- [configs](configs): Choose yaml file based on model.
    - Available configs:
        - [config_stable.yaml](configs/config_stable.yaml)
        - [config_qwen.yaml](configs/config_qwen.yaml)
##### Note: For details how to set parameters, read [README.md](configs/README.md) in [configs](configs).

### 2. Run bash script:
```bash
bash run_all.sh
```
**Note:** You will be asked to choose a config file.
Choose *1* for config_qwen, and *2* for config_stable.

- Script: [run_all.sh](run_all.sh)
