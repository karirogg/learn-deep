#!/bin/bash

export seed=420
export lower=20
export upper=20
sbatch --export=ALL jobscript.sh
# export lower=35
# sbatch --export=ALL seed=420 lower=35 upper=20 jobscript.sh

# export lower=20
# export upper=35
# sbatch --export=ALL seed=420 lower=20 upper=35 jobscript.sh
# export lower=35
# sbatch --export=ALL seed=420 lower=35 upper=35 jobscript.sh
# sbatch --export=ALL seed=420 lower=20 upper=50 jobscript.sh
# sbatch --export=ALL seed=420 lower=35 upper=50 jobscript.sh
