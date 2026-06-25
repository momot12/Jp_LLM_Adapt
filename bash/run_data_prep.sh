#!/bin/bash

# stop at errors
set -e

CONFIG="config"

echo "-------------------------------------------"
echo "Using configuration: $CONFIG"
echo "-------------------------------------------"
echo

# Run preprocessing
python3 -m pipeline.main_data_preprocess --config "$CONFIG"

echo
echo "Data preprocessing completed!"