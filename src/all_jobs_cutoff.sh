#!/bin/bash

sbatch --export=ALL seed=420 lower=20 upper=20 jobscript.sh
sbatch --export=ALL seed=420 lower=35 upper=20 jobscript.sh
sbatch --export=ALL seed=420 lower=20 upper=35 jobscript.sh
sbatch --export=ALL seed=420 lower=35 upper=35 jobscript.sh
sbatch --export=ALL seed=420 lower=20 upper=50 jobscript.sh
sbatch --export=ALL seed=420 lower=35 upper=50 jobscript.sh
