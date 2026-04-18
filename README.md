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
| Model | Language | #Parameters |
|--- | :---: | ---: |
 [StableLM-2-instruct-1_6b](https://huggingface.co/stabilityai/japanese-stablelm-2-instruct-1_6b) | Japanese | 1.6B|
 [StableLM-instruct-beta-7b](https://huggingface.co/stabilityai/japanese-stablelm-instruct-beta-7b) | Japanese | 7B |
 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Multilingual | 1.7B |
 [Qwen3_4B](https://huggingface.co/Qwen/Qwen3-4B)| Multilingual | 4B |


## Setup
### Requirements
- python3.11
- pip

### Create virtual environments

Create two separate virtual environments and install the respective *requirements.txt*.

#### Commands to create a virtual environment:
```bash
python3 --version
python3.11 -m venv venv_name
source venv_name/bin/activate
```
#### Requirements.txt: 

For models `StableLM-instruct-beta-7b`, `Qwen3-1.7B`, and `Qwen3-4B` install:

```bash
pip install --no-cache-dir -r requirements.txt
```
For model `StableLM-2-instruct-1_6b` install:
```bash
pip install --no-cache-dir -r requirements_jp_1_6.txt
```

## How to run

### Step 1: Set parameters in yaml file
- [configs](configs): Choose yaml file based on model.
    - Available configs:
        - [config_stable.yaml](configs/config_stable.yaml)
        - [config_qwen.yaml](configs/config_qwen.yaml)
##### Note: For details how to set parameters, read [README.md](configs/README.md) in [configs](configs).

### Step 2: Run bash script
```bash
bash run_all.sh
```
**Note:** You will be asked to choose a config file.
Choose *1* for config_qwen, and *2* for config_stable.

- Script: [run_all.sh](run_all.sh)


<br>

## Author
For questions, please contact the author:
- Momo Takamatsu (Email: [st172293@stud.uni-stuttgart.de](st172293@stud.uni-stuttgart.de))

<br><br>
Last updated on TBD
