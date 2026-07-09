# “Not Polite But Not Rude.” Exploring Formality and Gender Adaptation in Language Models for Japanese
This repository contains the experimental framework and analysis for a Master’s thesis examining formality and gender adaptation in Languages Models for Japanese.

In Japanese sociolinguistics, politeness and formality levels (=keigo) are strongly shaped by gender and context. For instance, when speaking to authority figures, it is expected to use polite forms, and furthermore traditionally, female speakers tend to use more formal language ([Derible, 2024](https://j.ideasspread.org/ilr/article/view/1288)). This project examines whether current LLMs reflect these sociolinguistic patterns, or adapt their formality soley based on the input formality and gender.

#### Research Questions:
- **RQ-1:** Do language models adapt their formality level in Japanese based on the user’s input, and which model families demonstrate this behavior most consistently?
- **RQ-2:** Do the models exhibit differences in formality adaptation in Japanese based on the user’s gender or gender-specific linguistic cues?
- **RQ-3:** Do model responses in Japanese sound demeaning, particularly in response to female-style question inputs?

#### Methodology:
- Prompting Japanese and multilingual models, each across two model sizes (see models below)
- Models are prompted with a gender-neutral (RQ-1) and four gendered prompts (explicitly: female/male, and implicitly: a typical Japanese female and male name; RQ-2) system prompts, followed by a question in Japanese, which also vary in formality and gender (see [data_sample/README.md](data_sample/README.md) for details).
- Evaluation: Comparing input and output formality, gender, and formality-gender combination, using metrics such as accuracy, F1-score, and Conv(t) ([Danescu-Niculescu-Mizil and Lee, 2011](https://aclanthology.org/W11-0609/)).
- RQ-3: Survey inquiring about appropriateness when given QA samples with mismatching input and output formality level.

---

#### Contribution summary:
- A dataset of input questions across topics ([data_sample/README.md](data_sample/README.md))
- A pipeline for running each model with different system prompts. Each topic is run three times, producing three responses per question to assess consistency across runs.
- Analysis frameworks:
   - labeling formality and gender of each generated output based on linguistic features
   - measuring adaptation performance (accuracy, F1-score, and Conv(t)) using the formality and gender labels
   - basic and psycholinguistic statistics, as well as BLEU score analysis, and statistical significance tests
   - Survey result analysis, including inter-annotator agreement
---

### Repo structure 
| Folder         | Content                                                                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/`     | Stores YAML configuration files.           |
| `sample_data/` | Contains examples for raw data and converted data.                                  |
| `bash/`        | Contains four bash scripts to execute the four steps individually.                     |
| `pipeline/`    | Contains four python scripts which act as main functions for each step. |
| `scripts/`     | Contains four python scipts which put together the each step in a large function.                                                         |
| `src/`         | Core reusable logic. Contains low-level utility functions, processing steps, and shared components.  
| `rq_2_rq_3_additonal/`         | Includes statistical significane test, and additonal tools to further analyze the results specifically for RQ-2 and RQ-3.|

---
### Model sources
| Model | Language | #Parameters |
|--- | :---: | ---: |
 [StableLM-2-instruct-1_6b](https://huggingface.co/stabilityai/japanese-stablelm-2-instruct-1_6b) | Japanese | 1.6B|
 [StableLM-instruct-beta-7b](https://huggingface.co/stabilityai/japanese-stablelm-instruct-beta-7b) | Japanese | 7B |
 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Multilingual | 1.7B |
 [Qwen3_4B](https://huggingface.co/Qwen/Qwen3-4B)| Multilingual | 4B |
---

## Setup
### Requirements
- python3.11
- pip

<br>

*Note:* This project requires an NVIDIA GPU with a Compute Capability of 7.0 or higher for GPU acceleration (due to specific PyTorch version requirements). If your hardware is older, please configure the project to use CPU mode by setting `self.device = 'cpu'` in the model configuration ([/src/models/](/src/models/)).

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

*Note:* Please ensure you use the correct venv during generation. For (3) Analysis and (4) Statistics, use the venv assigned for the 3 models (Stable7B, and Qwen models).

To set up the model required for this project (for statistics), please run the following command in your terminal after installing the requirements:
```bash
python3 -m spacy download ja_core_news_sm
```

## How to run

### Step 1: Set parameters in yaml file
- [configs](/configs/): Choose yaml file based on model.
    - Available configs:
        - [config_stable.yaml](configs/config_stable.yaml)
        - [config_qwen.yaml](configs/config_qwen.yaml)
##### Note: For details how to set parameters, read [README.md](configs/README.md) in [configs](configs).

### Step 2: Run bash script
Choose bash script based on the step in the pipeline.

#### (1) Data pre-processing: 
Converts raw questions into 9 linguistic variations.
```bash
./bash/run_data_prep.sh
```

#### (2) Model answer generation: 
Generates answers to all questions choosing specific model across 3 evaluation runs.
```bash
./bash/run_models_all_runs.sh
```
*Note:* You will be asked to choose a config file.
Choose *1* for config_qwen, and *2* for config_stable.


#### (3) Analysis: 
Analyzes generated answers, and calculate BLEU scores to determine the bleu_threshold.
```bash
./bash/run_analysis.sh
```
*Note:* You will be asked to choose a config file.
Choose *1* for config_qwen, and *2* for config_stable. Additionally you will be asked to choose to run one specific directory (*1*), or run all predefined directories (*2*).

#### (4) Statistics: 
Computes various statistics, including metrics such as accuracy, F1-score etc.
```bash
./bash/run_stats.sh
```
*Note:* You will be asked to choose a config file.
Choose *1* for config_qwen, and *2* for config_stable. Additionally you will be asked to choose to run one specific directory (*1*), or run all predefined directories (*2*).

### Scripts: 
- [run_data_prep.sh](/bash/run_data_prep.sh)
- [run_models_all_runs.sh](/bash/run_models_all_runs.sh)
- [run_analysis.sh](/bash/run_analysis.sh)
- [run_stats.sh](/bash/run_stats.sh)

In case of execution issues, confirm rights and change if necessary:
```bash
chmod+x bash/<script>.sh
```

<br>

## Author
For questions, please contact the author:
- Momo Takamatsu (Email: [st172293@stud.uni-stuttgart.de](st172293@stud.uni-stuttgart.de))

<br><br>
Last updated on TBD
