#!/bin/bash

set -e

# download torch, wilds, and deps as wheels so they can be installed on compute nodes
# (We use torch 1.11 and torchvision 0.12 in order to use the pretrained ViT models.)
python3.8 -m venv .env
source .env/bin/activate
pip install -U pip
pip install \
    torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    torchvision==0.12+cu113 -f https://download.pytorch.org/whl/torchvision/ \
    wilds==2.0.0
pip install -r slurm_requirements.txt

# cache pretrained torchvision models
python -c '
import os
import torch
import torchvision.models as models

torch.hub.set_dir(os.path.expanduser("~/torch_cache"))

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
