#!/bin/bash

# Seeds to iterate over
seeds=(31 42 69 420 80085)

lower_cutoffs=(20 35)
upper_cutoffs=(20 35 50)

# Replay weight keys
weights=("vog" "learning_speed" "mc_variance")

# Output file
output_file="ewf_results.txt"

# Clear the output file before running
> "$output_file"

# run preprocessing
PYTHONUNBUFFERED=1 python -m preprocessing.ewf

# Outer loop: Iterate over seeds
for seed in "${seeds[@]}"; do

    for lower in "${lower_cutoffs[@]}"; do
        for upper in "${upper_cutoffs[@]}"; do
        echo "Running experiments for seed: $seed, lower: $lower, upper: $upper" | tee -a "$output_file"
            # Force unbuffered output
            PYTHONUNBUFFERED=1 python -m models.ewf.train \
                --epochs 50 \
                --replay-buffer uniform \
                --buffer-size 10 \
                --cutoff-lower "$lower" \
                --cutoff-upper "$upper" \
                --seed "$seed" | tee -a "$output_file"

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

                # Run the Python command
                PYTHONUNBUFFERED=1 python -m models.ewf.train \
                    --epochs 50 \
                    --replay-buffer weighted_mean \
                    --replay-weights "$replay_weights" \
                    --buffer-size 10 \
                    --cutoff-lower "$lower" \
                    --cutoff-upper "$upper" \
                    --seed "$seed" | tee -a "$output_file"
            done
        done
    done
done

python extract_results_ewf.py
