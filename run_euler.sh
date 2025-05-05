#!/bin/bash

#SBATCH --output=/cluster/home/lbarberi/PAI/project_1/PAI_project_1/task4/logs.txt

#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu=16000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g

# # Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

echo "Running on node: $(hostname)"

# Binary or script to execute
python checker_client.py

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
