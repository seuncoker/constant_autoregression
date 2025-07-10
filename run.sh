#!/bin/bash


#SBATCH --job-name=test_job   # Job name
#SBATCH --time=00:30:00         # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request a single GPU
#SBATCH --cpus-per-task=2          # Request 4 CPU cores
#SBATCH --mem-per-cpu=8G           # Request 8GB memory per CPU core



# define license and load module
module load miniforge/24.7.1
module load cuda/12.6.2

# Launch the executable
source activate env_phd

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/env_phd/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/le_pde/lib/

# python /mnt/scratch/scoc/constant_autoregression/create_dir.py > dir_path__.txt

# echo "argument name: $1"

python /mnt/scratch/scoc/constant_autoregression/create_dir.py --argument_file=argument_vcnef >> dir_path_output.txt

dir_path=$(cat argument_vcnef.txt)
last_element=$(basename "$dir_path")

echo "created directory"

python /mnt/scratch/scoc/constant_autoregression/main.py --argument_file=argument_vcnef >& "$dir_path/$last_element.txt"

