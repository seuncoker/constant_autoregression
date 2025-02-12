#!/bin/bash
# use current working directory
#$ -cwd

# Request runtime
#$ -l h_rt=00:120:00

# Run on feps-cpu ARC4
#$ -P feps-gpu

#$ -N testing

# Request resource v 100
#$ -l coproc_v100=1

# define license and load module
module load anaconda/2023.03
module load cuda/11.1.1

# Launch the executable
source activate le_pde
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/env_phd/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/le_pde/lib/


python /nobackup/scoc/variable_autoregression/main.py
