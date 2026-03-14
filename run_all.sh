#!/bin/bash

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
      python3 main.py --config config_stable --data_file "$file" --out_file "$run"
   done
done
echo "All runs completed!"