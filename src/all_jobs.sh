#!/bin/bash

sbatch --export=ALL seed=31 lower=35 upper=50 jobscript.sh
sbatch --export=ALL seed=42 lower=35 upper=50 jobscript.sh
sbatch --export=ALL seed=69 lower=35 upper=50 jobscript.sh
sbatch --export=ALL seed=420 lower=35 upper=50 jobscript.sh
sbatch --export=ALL seed=80085 lower=35 upper=50 jobscript.sh
