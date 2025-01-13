#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=2880

# Access the variables passed via sbatch --export
seed=${seed}    # Default value if not provided
lower=${lower} # Default value if not provided
upper=${upper} # Default value if not provided

command bash single_experiment.sh -s $seed -l $lower -u $upper