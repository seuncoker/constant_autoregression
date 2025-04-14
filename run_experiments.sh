#!/bin/bash


#SBATCH --job-name=test_job   # Job name
#SBATCH --time=00:10:00         # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request a single GPU
#SBATCH --cpus-per-task=1          # Request 4 CPU cores
#SBATCH --mem-per-cpu=8G           # Request 8GB memory per CPU core



# define license and load module
module load miniforge/24.7.1
module load cuda/12.6.2

# Launch the executable
source activate env_phd


python /mnt/scratch/scoc/constant_autoregression/automate_jobs.py 

