#!/bin/bash

# stop at errors
set -e

# interactive selection of config
echo
echo "Choose a configuration:"
echo "1) config_qwen"
echo "2) config_stable"
read -p "Type 1 or 2 and press Enter: " choice

# assign the chosen config to a variable
case $choice in
    1) CONFIG="config_qwen" ;;
    2) CONFIG="config_stable" ;;
    *) echo "Invalid selection. Exiting."; exit 1 ;;
esac

echo "-------------------------------------------"
echo "Using configuration: $CONFIG"
echo "[Note] Check BLEU threshold in config."
echo "-------------------------------------------"
echo

# choose individual directory or all directories
echo "Choose mode:"
echo "1) Run single directory (manual input)"
echo "2) Run all predefined directories (check bash script)"
read -p "Enter 1 or 2 and press Enter: " mode
echo

# predefined directories
DIRS=(
    "jp_stable_1_6B"
    "jp_stable_1_6B_female"
    "jp_stable_1_6B_male"
    "jp_stable_1_6B_nanako"
    "jp_stable_1_6B_rintaro"
    "jp_stable_7B"
    "jp_stable_7B_female"
    "jp_stable_7B_male"
    "jp_stable_7B_nanako"
    "jp_stable_7B_rintaro"
    "qwen_1_7B"
    "qwen_1_7B_female"
    "qwen_1_7B_male"
    "qwen_1_7B_nanako"
    "qwen_1_7B_rintaro"
    "qwen_4B"
    "qwen_4B_female"
    "qwen_4B_male"
    "qwen_4B_nanako"
    "qwen_4B_rintaro"
)

# individual (manually entered) directory
if [ "$mode" = "1" ]; then
    echo "Example: jp_stable_7B_female, qwen_1_7B etc."
    read -p "Enter directory name and press Enter: " sub_dir

    python3 -m pipeline.main_stats \
        --config "$CONFIG" \
        --subdir "$sub_dir"


# all predefined directories
elif [ "$mode" = "2" ]; then
    echo "[INFO] Run ALL predefined directories:"

    for sub_dir in "${DIRS[@]}";
    do

        python3 -m pipeline.main_stats \
            --config "$CONFIG" \
            --subdir "$sub_dir"
    done

else
    echo "Invalid selection"
    exit 1
fi

echo
echo "Stats completed!"