#!/bin/bash

# Seeds to iterate over
seeds=(31 42 69 420 80085)

# Replay weight keys
weights=("vog" "learning_speed" "mc_entropy" "mc_mutual_information" "mc_variation_ratio" "mc_mean_std")

# Output file
output_file="results.txt"

# Clear the output file before running
> "$output_file"

# Outer loop: Iterate over seeds
for seed in "${seeds[@]}"; do
    echo "Running experiments for seed: $seed" | tee -a "$output_file"

    python -m models.cifar.train \
        --n 2 \
        --classes 100 \
        --epochs 50 \
        --replay-buffer uniform \
        --wandb \
        --seed "$seed" >> "$output_file" 2>&1

    # Inner loop: Iterate over replay weights
    for weight in "${weights[@]}"; do
        # Create a JSON string with the current weight set to 1.0 and all others to 0.0
        replay_weights=$(jq -n --arg w "$weight" '{
            "vog": 0.0,
            "learning_speed": 0.0,
            "mc_entropy": 0.0,
            "mc_mutual_information": 0.0,
            "mc_variation_ratio": 0.0,
            "mc_mean_std": 0.0,
            "mc_variance": 0.0,
            ($w): 1.0
        }')

        echo "Running with replay weight: $weight set to 1.0" | tee -a "$output_file"

        # Run the Python command
        python -m models.cifar.train \
            --n 2 \
            --classes 100 \
            --epochs 50 \
            --replay-buffer simple_sorted \
            --wandb \
            --replay_weights "$replay_weights" \
            --seed "$seed" >> "$output_file" 2>&1
    done
done
