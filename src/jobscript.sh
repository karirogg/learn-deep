#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --gpus=a100_80gb
#SBATCH --mem-per-cpu=200G
#SBATCH --time=2880

command bash experiment_cifar.sh