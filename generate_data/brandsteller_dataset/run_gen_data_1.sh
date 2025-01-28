#!/bin/bash
#$ -cwd
#$ -P feps-cpu
#$ -l h_rt=20:00:00
#$ -l nodes=1
#$ -l h_vmem=4G

# define license and load module
module load anaconda/2023.03

# Launch the executable
source activate le_pde

python /nobackup/scoc/variable_autoregression/generate_data/generate_1.py --experiment=KS --L=64 --nt=1000 --nt_effective=640 --end_time=200
