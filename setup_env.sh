#!/bin/bash		

# create conda env with required packages		
conda create --name robust python=3.6 scipy numpy jupyter pytorch torchvision cudatoolkit=10.1 statsmodels -c pytorch -c conda-forge		
conda activate robust		
# install pip packages		
pip install tqdm

# install deepiv		
pip install -e .
