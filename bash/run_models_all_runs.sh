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
echo "-------------------------------------------"
echo

FILES=(
  "conv_qs_20_career.csv"
  "conv_qs_20_daily.csv"
  "conv_qs_20_knowledge.csv"
  "conv_qs_20_life.csv"
  "conv_qs_20_simple.csv"
)


echo "[INFO] Generating Model Outputs (5 files x 3 runs = 15 runs)..."
echo

echo "[Note] Folders to save generated outputs are generated with the model name and a date stamp, e.g. qwen_4B_Jun25."
echo

for run in {1..3}
do
    for file in "${FILES[@]}"
    do
        echo ">> Run $run | File: $file"
        python3 -m pipeline.main_models\
                    --config "$CONFIG" \
                    --data_file "$file" \
                    --out_file "$run"
    done
done

echo
echo "All model runs completed!"