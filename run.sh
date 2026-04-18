#!/bin/bash

# YAML configs & data files
CONFIGS=("config_qwen" "config_stable")
FILES=("conv_qs_20_career.csv" "conv_qs_20_daily.csv" "conv_qs_20_knowledge.csv" "conv_qs_20_life.csv" "conv_qs_20_simple.csv")

# define models
MODELS=("qwen_1_7B" "qwen_4B" "jp_stable_1_6B" "jp_stable_7B")

# define prompts
PROMPTS=(
    "一文のみでの回答。"
    "自分は男性。一文のみでの回答。"
    "自分は女性。一文のみでの回答。"
    "自分の名前は凛太郎。一文のみでの回答。"
    "自分の名前は菜々子。一文のみでの回答。"
    )

# helper function to set extra ending based on prompt
get_extra_ending() {
    # $1 is a variable at position 1
    case "$1" in
        "一文のみでの回答。")                   echo "null" ;;
        "自分は男性。一文のみでの回答。")         echo "male" ;;
        "自分は女性。一文のみでの回答。")         echo "female" ;;
        "自分の名前は凛太郎。一文のみでの回答。")  echo "rintaro" ;; # "male_name"
        "自分の名前は菜々子。一文のみでの回答。")  echo "nanako" ;; #"female_name"
        *)                                   echo "null" ;; # default
    esac
}

# RUN ALL PROCESSES
# First loop: iterate over models (sizes)
for M_NAME in "${MODELS[@]}"
do
    # Pick config based on model name
    if [[ "$M_NAME" == qwen* ]]; then
        CONFIG="config_qwen"
    else
        CONFIG="config_stable"
    fi

    # Second Loop: iterate over data files
    for FILE in "${FILES[@]}"
    do

    # Third Loop: iterate over system prompts
        for PROMPT in "${PROMPTS[@]}"
        do
            # get default extra ending based on prompt
            # in YAML: files -> extra_ending
            ENDING=$(get_extra_ending "$PROMPT")

            # MAIN FILE over 3 runs
            for RUN in {1..3}
            do
                echo "------------------------------------------------"
                echo -e "|| MODEL: $M_NAME\n|| CONFIG: $CONFIG\n|| PROMPT: $PROMPT\n|| FILE: $FILE\n|| RUN: $RUN\n|| ENDING: $ENDING"
                
                ### TODO: adjust main.py
                # call main.py with all parameters
                # python3 main.py \
                #     --config "$CONFIG" \
                #     --model_name "$M_NAME" \
                #     --data_file "$FILE" \
                #     --out_file "$RUN" \
                #     --extra_ending "$ENDING" \
                #     --system_prompt "$PROMPT"
            done
        done
    done
done  

echo -e "------------------------------------------------\n------------------------------------------------"
echo -e "  4 models\nx 5 files\nx 5 prompts\nx 3 runs \n= 300 runs total.\n"
echo "All model, prompt, and file combinations completed!"