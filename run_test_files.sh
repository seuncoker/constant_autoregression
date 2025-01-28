#!/bin/bash
# use current working directory
#$ -cwd

# Request runtime
#$ -l h_rt=01:00:00

# Run on feps-cpu ARC4
#$ -P feps-gpu

#$ -N test_args


# Request resource v 100
#$ -l coproc_v100=1

# define license and load module
module load anaconda/2023.03
module load cuda/11.1.1

# Launch the executable
source activate le_pde
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/env_phd/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/home02/scoc/.conda/envs/le_pde/lib/



for filename in $(cat /nobackup/scoc/variable_autoregression/test_files_1.txt); do
  python /nobackup/scoc/variable_autoregression/change_argument.py --argument_loc="/nobackup/scoc/variable_autoregression/arguments_test.json" --key="test_only_path" --new_value="$filename"  # Pass filename as argument
  python /nobackup/scoc/variable_autoregression/main.py
done

