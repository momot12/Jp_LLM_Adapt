#!/bin/bash

# Interactive selection of config
echo "Choose a configuration:"
echo "1) config_qwen"
echo "2) config_stable"
read -p "Type 1 or 2 and press Enter: " choice

# Assign the chosen config to a variable
case $choice in
    1) CONFIG="config_qwen" ;;
    2) CONFIG="config_stable" ;;
    *) echo "Invalid selection. Exiting."; exit 1 ;;
esac

echo
echo "Using configuration: $CONFIG"
echo "-------------------------------------------"
echo

# check if run_model is true in yaml
RUN_MODEL_ENABLED=$(grep "run_model:" ./configs/$CONFIG.yaml | awk '{print $2}')

if [ "$RUN_MODEL_ENABLED" = "true" ]; then
    echo "[INFO] Models running (5 files x 3 runs = 15 runs)..."
    echo
    #FILES=("onesample.csv" "onesample2.csv")
    FILES=("conv_qs_20_career.csv" "conv_qs_20_daily.csv" "conv_qs_20_knowledge.csv" "conv_qs_20_life.csv" "conv_qs_20_simple.csv")

    for run in {1..3}
    do
       for file in "${FILES[@]}"
       do
          echo ">> Run $run | File: $file"
          python3 main.py --config "$CONFIG" --data_file "$file" --out_file "$run"
       done
    done
else
    echo "[INFO] Model runs disabled. Running other chosen processes..."
    python3 main.py --config "$CONFIG"
fi

echo "All tasks completed!"