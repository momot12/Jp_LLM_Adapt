#!/bin/bash

# Interactive selection of config
echo "Choose a configuration:"
echo "1) config_qwen"
echo "2) config_stable"
read -p "Enter 1 or 2: " choice

# Assign the chosen config to a variable
case $choice in
    1)
        CONFIG="config_qwen"
        ;;
    2)
        CONFIG="config_stable"
        ;;
    *)
        echo "Invalid selection. Exiting."
        exit 1
        ;;
esac

echo "Using configuration: $CONFIG"
echo "--------------------------"

# Array of your 5 files
#FILES=("onesample.csv" "onesample2.csv")
FILES=("conv_qs_20_career.csv" "conv_qs_20_daily.csv" "conv_qs_20_knowledge.csv" "conv_qs_20_life.csv" "conv_qs_20_simple.csv")

# Loop 3 times for each file
for run in {1..3}
do
   for file in "${FILES[@]}"
   do
      echo "Starting Run $run for $file"
      # Use --data_file and --out_file as flags if your argparse supports them
      # OR create a temporary yaml here.
      python3 main.py --config "$CONFIG" --data_file "$file" --out_file "$run"
   done
done
echo "All runs completed!"