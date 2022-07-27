#!/bin/bash
# This is the script for running on StabilityAI's compute environment.
# remember to run env_stabai.sh once first
#
#SBATCH --partition=compute-od-gpu
#SBATCH --job-name=scaling-wgp
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=0-2

set -e

# change this to wherever you have this repo downloaded
PROJECT_HOME=~/worst_group_scale

cd $PROJECT_HOME
source .env/bin/activate

for n in {0..19}
do
    task=$(($SLURM_ARRAY_TASK_ID+3*$n))
    python $PROJECT_HOME/run.py $task \
        --model-cache ~/torch_cache \
        --tensorboard $PROJECT_HOME/runs \
        --checkpoints ~/checkpoints \
        --wilds ~/data
done
