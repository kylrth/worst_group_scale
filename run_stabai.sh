#!/bin/bash
# This is the script for running on StabilityAI's compute environment.
# remember to run env_stabai.sh once first
#
#SBATCH --partition=compute-od-gpu
#SBATCH --job-name=scaling-wgp
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=0

set -e

# change this to wherever you have this repo downloaded
PROJECT_HOME=~/worst_group_scale

cd $PROJECT_HOME
source .env/bin/activate

python $PROJECT_HOME/run.py $SLURM_ARRAY_TASK_ID \
    --model-cache ~/torch_cache \
    --tensorboard $PROJECT_HOME/runs \
    --checkpoints $PROJECT_HOME/checkpoints
