#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=2880

command bash experiment_ewf.sh