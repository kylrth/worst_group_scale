#!/bin/bash
# remember to run slurm_env.sh once first
#
#SBATCH --time=0-08:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=0

set -e

# change this to wherever you have this repo downloaded
PROJECT_HOME=~/projects/def-irina/$USER/worst_group_scale

# create virtualenv from wheels and requirements.txt
module load python/3.9 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r slurm_requirements.txt
pip install wheels/*

# copy WILDS data from local (extract only celebA)
cd $SLURM_TMPDIR
mkdir -p work/data
cd work
tar -xf $PROJECT_HOME/data.tar --directory ./data/ --checkpoint=20000 ./celebA_v1.0

python $PROJECT_HOME/run.py $SLURM_ARRAY_TASK_ID \
    --model-cache ~/scratch/torch_cache \
    --tensorboard ~/scratch/worst_group_scale/runs \
    --checkpoints ~/scratch/worst_group_scale/checkpoints
