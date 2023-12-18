#!/bin/bash

input_variables=("1" "2" "3" "4" "5")

# Loop through the input variables and run Python scripts
for var in "${input_variables[@]}"; do
    echo "Running Python script with input: $var"
    python3 features_extraction.py --dataset D2 --portion "$var" --save
    python3 features_extraction.py --dataset D2 --portion "$var" --mirrored --save
done

input_variables=("6" "7" "8" "9" "10")

# Loop through the input variables and run Python scripts
for var in "${input_variables[@]}"; do
    echo "Running Python script with input: $var"
    python3 features_extraction.py --dataset D3 --portion "$var" --sav --skip_annotated_frames
    python3 features_extraction.py --dataset D3 --portion "$var" --mirrored --save --skip_annotated_frames
done


