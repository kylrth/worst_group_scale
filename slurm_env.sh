#!/bin/bash

# This needs to be run once to get wheels downloaded before submitting any jobs, since the compute
# nodes don't have internet access.

set -e

module load python/3.9 scipy-stack

# download torch, wilds, and deps as wheels so they can be installed on compute nodes
# (We use torch 1.11 and torchvision 0.12 in order to use the pretrained ViT models.)
mkdir -p wheels
pushd wheels
pip download --no-deps torch==1.11.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip download --no-deps torchvision==0.12+cu113 -f https://download.pytorch.org/whl/torchvision/
pip download --no-deps \
    littleutils==0.2.2 \
    ogb==1.3.3 \
    outdated==0.2.1 \
    wilds==2.0.0
popd

# cache pretrained torchvision models
pip install --no-index wheels/*
python -c '
import os
import torch
import torchvision.models as models

torch.hub.set_dir(os.path.expanduser("~/scratch/torch_cache"))

for factory in [
    models.resnet18,
    models.resnet34,
    models.resnet50,
    models.vit_b_32,
    models.vit_l_32,
]:
    model = factory(pretrained=True)
'

# cache WILDS data in a tarfile to copy to compute nodes
python -c '
from wilds import get_dataset

get_dataset("celebA", root_dir="./data", download=True)
get_dataset("waterbirds", root_dir="./data", download=True)
'
pushd data
tar -cf ../data.tar .
popd
rm -rf data
