#!/bin/bash


#SBATCH --job-name=test_job   # Job name
#SBATCH --time=00:05:00         # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request a single GPU
#SBATCH --cpus-per-task=8          # Request 4 CPU cores
#SBATCH --mem-per-cpu=8G           # Request 8GB memory per CPU core


# define license and load module
module load miniforge/24.7.1
module load cuda/12.6.2

# Launch the executable
source activate env_phd


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/env_phd/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/le_pde/lib/


python /mnt/scratch/scoc/constant_autoregression/main.py --argument_file=arguments_test
