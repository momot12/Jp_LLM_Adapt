# Data Creation

### Definitions:
- **Formality levels:**
    - **3**: formal (highest)
    - **2**: polite 
    - **1**: informal/casual

- **Genders:**
    - Neutral
    - Female
    - Male

- **5 Topics:** Career, daily, knowledge, life, simple

### Files:
- [raw_data.txt](https://github.tik.uni-stuttgart.de/st172293/Jp_LLM_adapt/blob/revision/sample_data/raw_data.txt): Questions for each 5 topis in neutral and simple, formality 1 form

- [converted_qs.csv](https://github.tik.uni-stuttgart.de/st172293/Jp_LLM_adapt/blob/revision/sample_data/converted_qs.csv): Variations of gender and formality combinations (3x3=9) per question.    
**Note:** The actual data for each topic is saved in a separate csv file.

### CSV:
- **Each file:**
    - Header: `q_id,question,topic,formality,gender`
    - comma-separated

- **Q_ID** consists of: `q1_work_3_neutral` (example)
    - `q1`: prefix *q* + a counter
    - `work`: topic
    - `3`: formality level 
    - `neutral`: gender