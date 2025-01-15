#!/bin/bash

# Seeds to iterate over
seeds=(69 420 80085)

# Replay weight keys
weights=("vog" "learning_speed" "predictive_entropy" "mutual_information" "variation_ratio" "mean_std_deviation")

# Output file
output_file="results.txt"

# Clear the output file before running
> "$output_file"

# run preprocessing
PYTHONUNBUFFERED=1 python -m preprocessing.cifar_100 --n 2

# Outer loop: Iterate over seeds
for seed in "${seeds[@]}"; do
    echo "Running experiments for seed: $seed" | tee -a "$output_file"

    # Inner loop: Iterate over replay weights
    for weight in "${weights[@]}"; do
        # Create a JSON string with the current weight set to 1.0 and all others to 0.0
        replay_weights=$(jq -n --arg w "$weight" '{
            "vog": 0.0,
            "learning_speed": 0.0,
            "predictive_entropy": 0.0,
            "mutual_information": 0.0,
            "variation_ratio": 0.0,
            "mean_std_deviation": 0.0,
            "mc_variance": 0.0,
            ($w): 1.0
        }')

        echo "Running with replay weight: $weight set to 1.0" | tee -a "$output_file"

        # Force unbuffered output
        PYTHONUNBUFFERED=1 python -m models.cifar.train \
            --n 2 \
            --classes 100 \
            --epochs 100 \
            --replay-buffer weighted_mean \
            --buffer-size 10 \
            --cutoff-lower 20 \
            --cutoff-upper 20 \
            --replay-weights "$replay_weights" \
            --use_checkpoint \
            --seed "$seed" | tee -a "$output_file"
    done
done

python extract_results.py
