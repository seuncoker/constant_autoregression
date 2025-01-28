#!/bin/bash
# use current working directory
#$ -cwd

# Request runtime
#$ -l h_rt=00:30:00

# Run on feps-cpu ARC4
# $ -P feps-gpu
# $ -P feps-cpu

#$ -N generate

# Request resource v 100
#$ -l coproc_v100=1

# define license and load module
module load anaconda/2023.03
module load cuda/11.1.1


source activate benchmark_dataset


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/benchmark_dataset/lib/

python3 reaction_diffusion_multi_solution_Hydra.py +multi=Rho1e0_Nu5e-1.yaml
